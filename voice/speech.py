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
from pathlib import Path

import certifi
import httpx
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

from voice.turns import split_sentences

DEFAULT_API = "https://api.lemontree.media/v1"
STT_MODEL = "large-v3-turbo"
TTS_MODEL = "qwen3-tts"
STT_RATE = 16_000
TTS_RATE = 24_000
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


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

	def __init__(self, api: str, voice: str, language: str, cache: Path):
		self.client = httpx.Client(base_url=api, timeout=30)
		self.voice, self.language, self.dir = voice, language, cache
		self.dir.mkdir(parents=True, exist_ok=True)

	def get(self, text: str) -> bytes:
		key = hashlib.sha1(f"{self.voice}|{self.language}|{text}".encode()).hexdigest()
		path = self.dir / f"{key}.pcm"
		if path.exists():
			return path.read_bytes()
		r = self.client.post("/audio/speech", json={
			"model": TTS_MODEL, "input": text, "voice": self.voice,
			"language": self.language, "response_format": "pcm"})
		r.raise_for_status()
		path.write_bytes(r.content)
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
	"""One WebSocket for the session: text in, 24 kHz PCM out as it is decoded."""

	def __init__(self, api: str, voice: str, language: str = "Auto", instructions: str | None = None):
		# The OpenAI-shaped path: the backend serves only this one; the gateway's /tts/stream alias
		# is an nginx rewrite that does not carry the websocket upgrade.
		self.url = api.replace("http", "ws", 1) + "/audio/speech/stream"
		self.config = {"type": "session.config", "voice": voice, "language": language,
		               "stream_audio": True, "response_format": "pcm"}
		if instructions:
			self.config["instructions"] = instructions
		self.ws = None

	def _connect(self):
		# httpx verifies TLS against certifi's bundle; websockets would use Python's default store,
		# which is empty in a uv-managed Python on macOS. Same bundle for both, so both work.
		tls = ssl.create_default_context(cafile=certifi.where()) if self.url.startswith("wss://") else None
		ws = connect(self.url, open_timeout=5, max_size=None, ssl=tls)
		ws.send(json.dumps(self.config))
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

		One utterance per sentence, in turn on the same socket: the server synthesizes about
		six times faster than real time, so the next sentence's audio lands while the previous
		one is still playing, and a cut-off is known to the sentence. `sentences` (a list,
		appended live) receives [sentence_text, first_byte, last_byte] in `sink.enqueued` space.
		"""
		sentences = [] if sentences is None else sentences
		for sentence in split_sentences(text):
			if cancel.is_set():
				break
			self._send(sentence)
			current = None
			while not cancel.is_set():
				try:
					msg = self.ws.recv(timeout=0.1)
				except TimeoutError:
					continue
				except ConnectionClosed:
					self.ws = None
					break
				if isinstance(msg, (bytes, bytearray)):
					sink.play(bytes(msg))
					continue
				ev = json.loads(msg)
				kind = ev.get("type")
				if kind == "audio.start":
					current = [ev.get("sentence_text") or sentence, sink.enqueued, None]
					sentences.append(current)
					if on_sentence:
						on_sentence(current[0])
				elif kind == "audio.done" and current is not None:
					current[2] = sink.enqueued
					current = None
				elif kind == "session.done":
					break
		if cancel.is_set():
			sink.stop()
			self.close()  # the server may still be sending; a fresh socket is cheaper than draining
			return sentences
		while sink.busy() and not cancel.is_set():
			time.sleep(0.02)
		if cancel.is_set():
			sink.stop()
		return sentences
