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
import math
import tempfile
import threading
import time
import wave
from pathlib import Path

import httpx

from voice.turns import speech_text, split_utterances

DEFAULT_API = "https://api.lemontree.media/v1"
STT_MODEL = "large-v3-turbo"
TTS_MODEL = "qwen3-tts"
STT_RATE = 16_000
TTS_RATE = 24_000
DEFAULT_INSTRUCTIONS = (
	"Speak with a neutral, conversational tone, a steady natural pitch, "
	"an even volume, and a normal speaking pace."
)
TTS_LANGUAGES = {
	"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
	"de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
	"es": "Spanish", "it": "Italian",
}
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def voice_config(voice: str, language: str, instructions: str | None, seed: int) -> dict:
	"""Identical conditioning for cached lines and streamed answers.

	The installed vLLM-Omni WebSocket schema ignores seed and extra_params; the HTTP
	speech endpoint applies them, including when streaming raw PCM.
	"""
	return {"model": TTS_MODEL, "voice": voice, "language": language,
	        "instructions": DEFAULT_INSTRUCTIONS if instructions is None else instructions,
	        "seed": seed, "extra_params": {"temperature": 0.5}, "response_format": "pcm"}


def audio_limit(text: str) -> int:
	"""Generous byte budget for normal speech; stops a runaway codec generation."""
	seconds = min(90, max(8, len(text) * 0.25))
	return math.ceil(seconds * TTS_RATE) * 2


def speech_request(config: dict, text: str) -> dict:
	return {**config, "input": text,
	        "max_new_tokens": math.ceil(audio_limit(text) / (TTS_RATE * 2) * 12.5)}


def check_pcm(pcm: bytes) -> None:
	if not pcm or len(pcm) % 2 or pcm.startswith((b"RIFF", b"OggS", b"ID3", b"fLaC")):
		raise ValueError("TTS returned invalid raw PCM audio")


def check_audio_response(response: httpx.Response) -> None:
	response.raise_for_status()
	content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
	if content_type not in {"audio/pcm", "application/octet-stream"}:
		raise ValueError(f"TTS returned {content_type!r}, expected raw PCM")


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
	def __init__(self, api: str = DEFAULT_API, language: str = "en", prompt: str = ""):
		self.client = httpx.Client(base_url=api, timeout=30)
		self.language, self.prompt = language, prompt

	def transcribe(self, pcm16k: bytes) -> str:
		r = self.client.post(
			"/audio/transcriptions",
			files={"file": ("turn.wav", wav_bytes(pcm16k, STT_RATE), "audio/wav")},
			data={"model": STT_MODEL, "language": self.language, "prompt": self.prompt,
			      "response_format": "json", "temperature": "0"},
		)
		r.raise_for_status()
		return r.json().get("text", "")


class Clips:
	"""Short fixed lines (fillers, tool announcements): synthesized once, kept as PCM on disk."""

	def __init__(self, api: str, voice: str, language: str, cache: Path,
	             instructions: str | None = None, seed: int = 42):
		self.client = httpx.Client(base_url=api, timeout=30)
		self.config = voice_config(voice, language, instructions, seed)
		self.api, self.dir = api.rstrip("/"), cache
		self.lock = threading.Lock()
		self.dir.mkdir(parents=True, exist_ok=True)

	def get(self, text: str) -> bytes:
		text = speech_text(text)
		if not text:
			return b""
		request = speech_request(self.config, text)
		key = hashlib.sha256(json.dumps({"version": 2, "api": self.api, **request},
		                               sort_keys=True).encode()).hexdigest()
		path = self.dir / f"{key}.pcm"
		with self.lock:
			if path.exists():
				pcm = path.read_bytes()
				try:
					check_pcm(pcm)
					if len(pcm) < audio_limit(text):
						return pcm
				except ValueError:
					pass
			r = self.client.post("/audio/speech", json=request)
			check_audio_response(r)
			check_pcm(r.content)
			if len(r.content) >= audio_limit(text):
				raise ValueError("TTS filler exceeded its audio budget")
			with tempfile.NamedTemporaryFile(dir=self.dir, suffix=".tmp", delete=False) as f:
				tmp = Path(f.name)
				try:
					f.write(r.content)
					f.close()
					tmp.replace(path)
				finally:
					tmp.unlink(missing_ok=True)
			return r.content

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
	"""HTTP raw PCM streaming with stable conditioning and bounded generation."""

	def __init__(self, api: str, voice: str, language: str = "English",
	             instructions: str | None = None, seed: int = 42):
		self.client = httpx.Client(base_url=api, timeout=httpx.Timeout(15, connect=5, read=5))
		self.config = voice_config(voice, language, instructions, seed)
		self.response = None

	def close(self) -> None:
		response, self.response = self.response, None
		if response is not None:
			try:
				response.close()
			except Exception:  # noqa: BLE001
				pass

	def speak(self, text: str, sink, cancel: threading.Event, on_sentence=None, sentences=None) -> list:
		"""Stream `text` into `sink` until it has all been played, or `cancel` is set.

		Short answers stay in one utterance so the model can keep a continuous delivery.
		`sentences` receives [utterance_text, first_byte, last_byte] in absolute sink bytes.
		"""
		sentences = [] if sentences is None else sentences
		try:
			for utterance in split_utterances(text):
				if cancel.is_set():
					break
				request = {**speech_request(self.config, utterance), "stream": True, "stream_format": "audio"}
				with self.client.stream("POST", "/audio/speech", json=request) as response:
					self.response = response
					check_audio_response(response)
					current = [utterance, sink.enqueued, None]
					total, pending, started = 0, b"", False
					deadline = time.monotonic() + 30
					for chunk in response.iter_bytes():
						if cancel.is_set():
							break
						if time.monotonic() > deadline:
							raise TimeoutError("TTS generation timed out")
						total += len(chunk)
						if total >= audio_limit(utterance):
							raise ValueError("TTS exceeded its audio budget")
						pending += chunk
						if not started and len(pending) < 4:
							continue
						pcm, pending = pending[:len(pending) // 2 * 2], pending[len(pending) // 2 * 2:]
						if not pcm:
							continue
						if not started:
							check_pcm(pcm)
							started = True
							sentences.append(current)
							if on_sentence:
								on_sentence(utterance)
						sink.play(pcm)
					if not cancel.is_set():
						if pending or not started:
							raise ValueError("TTS returned empty or truncated PCM")
						current[2] = sink.enqueued
				self.response = None
		except Exception:
			sink.stop()
			if not cancel.is_set():
				raise
		finally:
			self.close()
		while sink.busy() and not cancel.is_set():
			time.sleep(0.02)
		if cancel.is_set():
			sink.stop()
		return sentences
