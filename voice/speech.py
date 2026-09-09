# AI_OWNED
"""The network side of the voice front end, against the OpenAI-shaped gateway (docs/VOICE.md):
Whisper for what was said, Qwen3-TTS for what to say.

Nothing here touches audio hardware, so it runs on a box with no sound card too. The TTS
plays into a "sink" with `play(bytes)`, `stop()`, `busy()` and an `enqueued` byte counter:
client.Speaker is one; the tests use a fake.
"""

from __future__ import annotations

import hashlib
import io
import json
import ssl
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import certifi
import httpx
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

from voice.turns import dense_script, speech_seconds, speech_text, speech_utterances

DEFAULT_API = "https://api.lemontree.media/v1"
STT_MODEL = "large-v3-turbo"
TTS_MODEL = "qwen3-tts"
STT_RATE = 16_000
TTS_RATE = 24_000
# The delivery. The timbre is voice/robot.py's job; this is what the synthesizer is told
# about how to read, and it is the same for a cached filler and a live answer.
DEFAULT_INSTRUCTIONS = (
	"You are the voice of a small, cheerful male robot, like a friendly toy robot in a film: "
	"bright, quick, expressive and a little bouncy, with lively pitch movement and crisp "
	"consonants. Sound curious and warm, never sleepy and never sarcastic. "
	"Read only the supplied words exactly as written. Never laugh, giggle, chuckle, snort, "
	"sigh or add any vocal sound that is not a word."
)
DEFAULT_VOICE = "aiden"  # a bright male preset; the robot is built on top of it
TTS_LANGUAGES = {
	"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
	"de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
	"es": "Spanish", "it": "Italian",
}
# What the gateway may answer a synthesis request with. An allowlist, not a check for the
# error shapes we have seen: an HTML error page or a WAV is not audio we can enqueue.
PCM_CONTENT_TYPES = {"audio/pcm", "application/octet-stream", "audio/l16", "audio/x-pcm"}
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def stt_language(language: str) -> str:
	"""Whisper's language hint. Empty means "detect it from the audio", the only setting
	under which a bilingual speaker is transcribed rather than translated."""
	name = language.strip().lower().replace("_", "-")
	return "" if name in ("", "auto") else name.split("-")[0]


def tts_language(language: str) -> str:
	"""Resolve Whisper's ISO code (or an explicit TTS name) once at startup."""
	name = language.strip().lower().replace("_", "-")
	if name == "auto":
		return "Auto"
	if name.split("-")[0] in TTS_LANGUAGES:
		return TTS_LANGUAGES[name.split("-")[0]]
	for supported in TTS_LANGUAGES.values():
		if name == supported.lower():
			return supported
	raise ValueError(f"unsupported TTS language: {language}; use --tts-lang to choose one")


@dataclass(frozen=True)
class VoiceProfile:
	"""The same synthesis conditioning for cached status lines and live answers."""
	voice: str
	language: str = "English"
	instructions: str = DEFAULT_INSTRUCTIONS

	def payload(self) -> dict:
		return {"model": TTS_MODEL, **asdict(self), "response_format": "pcm"}


def wav_bytes(pcm: bytes, rate: int) -> bytes:
	buf = io.BytesIO()
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(rate)
		w.writeframes(pcm)
	return buf.getvalue()


def ensure_silero(cache: Path) -> Path:
	"""The VAD model file, fetched once into the cache."""
	path = cache / "silero_vad.onnx"
	if not path.exists():
		cache.mkdir(parents=True, exist_ok=True)
		r = httpx.get(SILERO_URL, follow_redirects=True, timeout=60)
		r.raise_for_status()
		path.write_bytes(r.content)
	return path


class STT:
	def __init__(self, api: str = DEFAULT_API, language: str = "auto", prompt: str = ""):
		self.client = httpx.Client(base_url=api, timeout=30)
		self.language, self.prompt = stt_language(language), prompt

	def transcribe(self, pcm16k: bytes) -> str:
		# Whisper works in 30-second windows, so a minute of speech is several passes and a
		# two-megabyte upload: a fixed 30-second timeout gives up on a long turn just as it
		# is about to come back. Wait in proportion to what was said.
		seconds = len(pcm16k) / (STT_RATE * 2)
		data = {"model": STT_MODEL, "prompt": self.prompt,
		        "response_format": "json", "temperature": "0"}
		if self.language:
			# Pinned to one language, Whisper renders anything else into it: Mandarin
			# under `en` comes back as English paraphrase, or as nothing at all. Sending
			# no hint at all is what makes a turn in another language transcribable.
			data["language"] = self.language
		r = self.client.post(
			"/audio/transcriptions",
			files={"file": ("turn.wav", wav_bytes(pcm16k, STT_RATE), "audio/wav")},
			data=data,
			timeout=max(30.0, seconds * 1.5),
		)
		r.raise_for_status()
		return r.json().get("text", "")


class Clips:
	"""Short fixed lines (fillers, tool announcements): synthesized once, kept as PCM on disk."""

	def __init__(self, api: str, voice: str, language: str, cache: Path,
	             instructions: str = DEFAULT_INSTRUCTIONS):
		self.client = httpx.Client(base_url=api, timeout=30)
		self.api, self.dir = api.rstrip("/"), cache
		self.profile = VoiceProfile(voice, tts_language(language), instructions)
		self.lock = threading.Lock()  # prefetch and playback can ask for the same clip
		self.dir.mkdir(parents=True, exist_ok=True)

	def get(self, text: str) -> bytes:
		payload = {**self.profile.payload(), "input": text}
		identity = {"version": 2, "api": self.api, "rate": TTS_RATE, **payload}
		key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
		path = self.dir / f"{key}.pcm"
		with self.lock:
			if path.exists():
				pcm = path.read_bytes()
				if self._valid(pcm):
					return pcm
			r = self.client.post("/audio/speech", json=payload)
			r.raise_for_status()
			kind = r.headers.get("content-type", "").split(";", 1)[0].strip().lower()
			if kind not in PCM_CONTENT_TYPES or not self._valid(r.content):
				raise RuntimeError(f"invalid or excessive PCM in status clip (as {kind!r})")
			# Atomic publication: no reader can hear a partially written cache file.
			tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
			tmp.write_bytes(r.content)
			tmp.replace(path)
			return r.content

	@staticmethod
	def _valid(pcm: bytes) -> bool:
		return (0 < len(pcm) <= TTS_RATE * 2 * 8 and len(pcm) % 2 == 0
		        and not pcm.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")))

	def prefetch(self, texts) -> threading.Thread:
		def run():
			for t in texts:
				try:
					self.get(t)
				except Exception:  # noqa: BLE001 - a missing filler is a silence, not a crash
					pass
		thread = threading.Thread(target=run, daemon=True)
		thread.start()
		return thread


class TTS:
	"""One WebSocket for the session: text in, 24 kHz PCM out as it is decoded."""

	def __init__(self, api: str, voice: str, language: str = "English",
	             instructions: str = DEFAULT_INSTRUCTIONS):
		# The OpenAI-shaped path: the backend serves only this one; the gateway's /tts/stream alias
		# is an nginx rewrite that does not carry the websocket upgrade.
		self.url = api.rstrip("/").replace("http", "ws", 1) + "/audio/speech/stream"
		self.profile = VoiceProfile(voice, tts_language(language), instructions)
		self.config = {"type": "session.config", **self.profile.payload(), "stream_audio": True}
		self.idle_timeout_s = 15.0
		self.ws = None

	def _retune(self, text: str) -> None:
		"""With the language left to the server, settle it once for the whole answer.

		Detection runs per request, and a request is one utterance out of several: a
		mostly-Latin stretch of a Chinese answer gets taken for English and read back as
		noise. The answer as a whole is never ambiguous, so it decides for its parts.
		"""
		if self.profile.language != "Auto":
			return
		language = "Chinese" if dense_script(speech_text(text)) else "Auto"
		if language != self.config["language"]:
			self.config = {**self.config, "language": language}
			self.close()  # the config is sent once, at connect, so this needs a new socket

	def _connect(self):
		# httpx verifies TLS against certifi's bundle; websockets would use Python's default store,
		# which is empty in a uv-managed Python on macOS. Same bundle for both, so both work.
		tls = ssl.create_default_context(cafile=certifi.where()) if self.url.startswith("wss://") else None
		ws = connect(self.url, open_timeout=5, close_timeout=1, max_size=4 * 1024 * 1024, ssl=tls)
		try:
			ws.send(json.dumps(self.config))
		except Exception:
			ws.close()
			raise
		return ws

	def close(self) -> None:
		ws, self.ws = self.ws, None
		if ws is not None:
			try:
				ws.close()
			except Exception:  # noqa: BLE001
				pass

	def _send(self, text: str) -> None:
		last_exc: Exception | None = None
		for _ in range(2):
			try:
				self.ws = self.ws or self._connect()
				self.ws.send(json.dumps({"type": "input.text", "text": text}))
				self.ws.send(json.dumps({"type": "input.done"}))
				return
			except (OSError, WebSocketException, TimeoutError) as exc:
				last_exc = exc
				self.close()
		raise RuntimeError(f"tts unreachable at {self.url}: {last_exc}")

	def speak(self, text: str, sink, cancel: threading.Event, on_sentence=None, sentences=None) -> list:
		"""Stream `text` into `sink` until it has all been played, or `cancel` is set.

		Short answers stay together to preserve context and prosody. `sentences` receives
		[segment_text, first_byte, last_byte] in absolute `sink.enqueued` space, using whatever
		segments the server emits. Never reuse a stream after cancellation or a protocol error.
		"""
		sentences = [] if sentences is None else sentences
		self._retune(text)
		try:
			for utterance in speech_utterances(text):
				if cancel.is_set():
					break
				self._send(utterance)
				current, remainder, received = None, b"", 0
				# A guard against a stream that will not stop, not a check on the estimate:
				# loose enough that a slow reading of an awkward line is not mistaken for a
				# runaway, tight enough that a runaway is still caught.
				max_audio_s = min(90.0, max(12.0, speech_seconds(utterance) * 3.0))
				last_audio = time.monotonic()
				deadline = last_audio + max(30.0, max_audio_s * 2)
				while not cancel.is_set():
					if time.monotonic() - last_audio > self.idle_timeout_s or time.monotonic() > deadline:
						raise TimeoutError("TTS stream stalled")
					try:
						msg = self.ws.recv(timeout=0.1)
					except TimeoutError:
						continue
					except ConnectionClosed as exc:
						raise RuntimeError("TTS stream closed before the answer finished") from exc
					if cancel.is_set():
						break
					if isinstance(msg, (bytes, bytearray)):
						if current is None:
							raise RuntimeError("TTS sent audio outside audio.start/audio.done")
						received += len(msg)
						if received > max_audio_s * TTS_RATE * 2:
							raise RuntimeError("TTS exceeded the audio duration limit")
						pcm = remainder + bytes(msg)
						usable = len(pcm) - len(pcm) % 2
						if usable:
							sink.play(pcm[:usable])
						remainder = pcm[usable:]
						if msg:
							last_audio = time.monotonic()
						continue
					ev = json.loads(msg)
					kind = ev.get("type")
					if kind == "error" or (kind == "audio.done" and ev.get("error")):
						raise RuntimeError(f"TTS failed: {ev.get('message') or ev.get('error')}")
					if kind == "audio.start":
						if (current is not None or ev.get("format", "pcm") != "pcm"
						    or ev.get("sample_rate", TTS_RATE) != TTS_RATE
						    or ev.get("channels", 1) != 1):
							raise RuntimeError("TTS must send framed 24 kHz mono PCM")
						current = [ev.get("sentence_text") or utterance, sink.enqueued, None]
						sentences.append(current)
						if on_sentence:
							on_sentence(current[0])
					elif kind == "audio.done":
						if current is None or remainder or sink.enqueued == current[1]:
							raise RuntimeError("TTS sent incomplete or empty PCM")
						if ev.get("total_bytes", sink.enqueued - current[1]) != sink.enqueued - current[1]:
							raise RuntimeError("TTS audio byte count mismatch")
						current[2] = sink.enqueued
						current = None
					elif kind == "session.done":
						if current is not None or not received:
							raise RuntimeError("TTS ended without complete audio")
						break
			while sink.busy() and not cancel.is_set():
				time.sleep(0.02)
		except Exception:
			sink.stop()
			self.close()
			raise
		finally:
			if cancel.is_set():
				sink.stop()
				self.close()  # no audio from this utterance can leak into the next one
		return sentences
