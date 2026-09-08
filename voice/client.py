# AI_OWNED
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sounddevice>=0.5",
#   "numpy>=1.26",
#   "onnxruntime>=1.17",
#   "httpx>=0.27",
#   "websockets>=13",
#   "certifi",
# ]
# ///
"""Talk to the agent from the terminal (docs/VOICE.md).

	./voice.sh [--voice ryan] [--lang en] [--api https://api.lemontree.media/v1]

The one program in the repo that runs on the host, because a container on macOS cannot
reach the microphone. It does audio and nothing else: the agent still runs in the
container, started through ./run.sh with voice/bridge.py as the entrypoint, and this
process talks to it over stdin/stdout.

	mic -> Silero VAD -> end of turn -> Whisper -> bridge -> agent_loop
	                                                 \\-> [tool] events -> "Alright, I'm using the file read tool."
	answer -> Qwen3-TTS websocket -> speaker, cut the moment the user talks over it

Keys: Enter interrupts; a typed line is a turn without the microphone; Ctrl-C quits.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from voice.speech import DEFAULT_API, STT, TTS, TTS_RATE, Clips, ensure_silero  # noqa: E402
from voice.turns import (  # noqa: E402
	ACKS, REASSURE, TOOL_LINE, BargeIn, EndpointConfig, Endpointer, Narrator, heard_text, usable_transcript,
)
from voice.vad import FRAME, SAMPLE_RATE, SileroVAD  # noqa: E402

CACHE = Path(os.environ.get("AGENT_VOICE_CACHE", Path.home() / ".cache" / "agent-voice"))
STT_PROMPT = "A developer talking to a coding agent about files, tests, git, Python, and the shell."
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def say(text: str) -> None:
	print(text, flush=True)


def dim(text: str) -> None:
	print(f"{DIM}{text}{RESET}", flush=True)


def pcm16(frames: list[np.ndarray]) -> bytes:
	return (np.clip(np.concatenate(frames), -1, 1) * 32767).astype("<i2").tobytes()


# --- audio in and out ---------------------------------------------------------------------

class Mic:
	"""16 kHz mono float32 frames of 512 samples on a queue; 48 kHz decimated by 3 if the
	device will not do 16 kHz."""

	def __init__(self, device=None):
		self.frames: queue.Queue[np.ndarray] = queue.Queue()
		self.decimate = 1
		try:
			self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
			                             blocksize=FRAME, device=device, callback=self._on_audio)
		except sd.PortAudioError:
			self.decimate = 3
			self.stream = sd.InputStream(samplerate=SAMPLE_RATE * 3, channels=1, dtype="float32",
			                             blocksize=FRAME * 3, device=device, callback=self._on_audio)
		self.stream.start()

	def _on_audio(self, indata, nframes, t, status):
		x = indata[:, 0]
		if self.decimate > 1:
			x = x.reshape(-1, self.decimate).mean(axis=1)
		self.frames.put(x.copy())

	def close(self) -> None:
		self.stream.stop()
		self.stream.close()


class Speaker:
	"""24 kHz PCM out of one device. One queue; stop() empties it at once.

	`enqueued`, `played` and `dropped` count bytes for the life of the process, so a caller
	can mark where an utterance began (`enqueued`) and later ask how far playback got
	(`consumed`) before it was cut.
	"""

	def __init__(self, device=None):
		self.lock = threading.Lock()
		self.buf = bytearray()
		self.enqueued = self.played = self.dropped = 0
		self.last_sound = 0.0
		self.stream = sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16",
		                                 blocksize=480, device=device, callback=self._on_pull)
		self.stream.start()

	def _on_pull(self, outdata, nframes, t, status):
		want = nframes * 2
		with self.lock:
			chunk = bytes(self.buf[:want])
			del self.buf[:want]
			self.played += len(chunk)
		if chunk:
			self.last_sound = time.monotonic()
		outdata[:len(chunk)] = chunk
		if len(chunk) < want:
			outdata[len(chunk):] = bytes(want - len(chunk))

	@property
	def consumed(self) -> int:
		return self.played + self.dropped

	def play(self, pcm: bytes) -> None:
		with self.lock:
			self.buf += pcm
			self.enqueued += len(pcm)

	def stop(self) -> None:
		with self.lock:
			self.dropped += len(self.buf)
			self.buf.clear()

	def busy(self) -> bool:
		with self.lock:
			queued = len(self.buf) > 0
		return queued or time.monotonic() - self.last_sound < 0.06

	def close(self) -> None:
		self.stream.stop()
		self.stream.close()


# --- the agent, and what we say around it ------------------------------------------------

class Agent:
	"""The agent in its container: ./run.sh with voice/bridge.py as entrypoint, JSON lines both ways."""

	def __init__(self):
		env = {**os.environ, "AGENT_ENTRYPOINT": "python"}
		self.proc = subprocess.Popen(
			[str(REPO / "run.sh"), "-m", "voice.bridge"], cwd=REPO, env=env, text=True, bufsize=1,
			stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		self.events: queue.Queue[dict] = queue.Queue()
		threading.Thread(target=self._read_stdout, daemon=True).start()
		threading.Thread(target=self._read_stderr, daemon=True).start()

	def _read_stdout(self) -> None:
		for line in self.proc.stdout:
			line = line.strip()
			if not line:
				continue
			try:
				self.events.put(json.loads(line))
			except ValueError:
				dim(f"  {line}")
		self.events.put({"type": "exit", "code": self.proc.wait()})

	def _read_stderr(self) -> None:
		for line in self.proc.stderr:
			line = line.rstrip("\n")
			if line and not TOOL_LINE.match(line):  # tool lines arrive as events already
				dim(f"  {line}")

	def send(self, obj: dict) -> None:
		self.proc.stdin.write(json.dumps(obj) + "\n")
		self.proc.stdin.flush()

	def close(self) -> None:
		try:
			self.proc.stdin.close()
		except OSError:
			pass
		try:
			self.proc.wait(timeout=15)
		except subprocess.TimeoutExpired:
			self.proc.terminate()


class Speech:
	"""One answer on its way out: a thread feeding the TTS stream into the speaker."""

	def __init__(self, tts: TTS, speaker: Speaker, text: str):
		self.speaker = speaker
		self.cancel = threading.Event()
		self.sentences: list = []
		self.start: int | None = None
		self.thread = threading.Thread(target=self._run, args=(tts, text), daemon=True)
		self.thread.start()

	def _run(self, tts: TTS, text: str) -> None:
		deadline = time.monotonic() + 0.6  # let a filler finish rather than clip it, briefly
		while self.speaker.busy() and time.monotonic() < deadline and not self.cancel.is_set():
			time.sleep(0.02)
		self.speaker.stop()
		self.start = self.speaker.enqueued
		try:
			tts.speak(text, self.speaker, self.cancel, sentences=self.sentences,
			          on_sentence=lambda s: say(f"{BOLD}agent>{RESET} {s}"))
		except Exception as exc:  # noqa: BLE001
			say(f"{BOLD}agent>{RESET} {text}")
			dim(f"  (no speech: {exc})")

	def done(self) -> bool:
		return not self.thread.is_alive()

	def interrupt(self) -> str:
		"""Cut playback and return what was heard, so the agent's history can say so."""
		heard = "" if self.start is None else heard_text(self.sentences, self.speaker.consumed - self.start)
		self.cancel.set()
		self.speaker.stop()
		return heard


class Narration:
	"""Plays the short lines the Narrator picks, from the clip cache, off the audio thread."""

	def __init__(self, clips: Clips, speaker: Speaker, answer_playing):
		self.clips, self.speaker, self.answer_playing = clips, speaker, answer_playing
		self.q: queue.Queue[str | None] = queue.Queue()
		threading.Thread(target=self._run, daemon=True).start()

	def offer(self, text: str) -> None:
		self.q.put(text)

	def mute(self) -> None:
		while not self.q.empty():
			try:
				self.q.get_nowait()
			except queue.Empty:
				break

	def close(self) -> None:
		self.q.put(None)

	def _run(self) -> None:
		while True:
			text = self.q.get()
			if text is None:
				return
			if self.answer_playing():
				continue
			try:
				pcm = self.clips.get(text)
			except Exception as exc:  # noqa: BLE001
				dim(f"  (filler failed: {exc})")
				continue
			if not self.answer_playing():
				dim(f"  {text}")
				self.speaker.play(pcm)


# --- the loop -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="talk to the agent")
	ap.add_argument("--api", default=os.environ.get("VOICE_API", DEFAULT_API))
	ap.add_argument("--voice", default=os.environ.get("VOICE", "ryan"))
	ap.add_argument("--lang", default="en", help="what you speak, as an ISO code, for Whisper")
	ap.add_argument("--tts-lang", default="Auto", help="Qwen3-TTS language name, or Auto")
	ap.add_argument("--instructions", default=None, help="TTS style hint, e.g. 'casual and warm'")
	ap.add_argument("--endpoint-ms", type=int, default=600, help="silence that ends your turn")
	ap.add_argument("--mic", default=None, help="input device name or index")
	ap.add_argument("--speaker", default=None, help="output device name or index")
	ap.add_argument("--list-devices", action="store_true")
	args = ap.parse_args(argv)
	if args.list_devices:
		print(sd.query_devices())
		return 0

	vad = SileroVAD(ensure_silero(CACHE))
	speaker = Speaker(args.speaker)
	clips = Clips(args.api, args.voice, args.tts_lang, CACHE / "clips")
	clips.prefetch(list(ACKS) + list(REASSURE))  # tool lines are made on first use
	tts = TTS(args.api, args.voice, args.tts_lang, args.instructions)
	stt = STT(args.api, args.lang, STT_PROMPT)
	pool = ThreadPoolExecutor(max_workers=2)

	dim("starting the agent container ...")
	agent = Agent()
	while True:
		ev = agent.events.get()
		if ev.get("type") == "ready":
			break
		if ev.get("type") == "exit":
			say(f"the agent did not start (exit {ev.get('code')})")
			return 1
	mic = Mic(args.mic)  # macOS asks for microphone permission here, once per terminal app
	say(f"listening as {args.voice}. Enter interrupts, a typed line is a turn, Ctrl-C quits.")

	typed: queue.Queue[str] = queue.Queue()
	threading.Thread(target=lambda: [typed.put(l) for l in sys.stdin], daemon=True).start()

	endpointer = Endpointer(EndpointConfig(end_ms=args.endpoint_ms))
	barge = BargeIn()
	narrator = Narrator()
	speech: Speech | None = None
	narration = Narration(clips, speaker, lambda: speech is not None and not speech.done())

	preroll: deque[np.ndarray] = deque(maxlen=12)  # ~380 ms before the VAD agrees it is speech
	utterance: list[np.ndarray] | None = None
	speculation: tuple[int, Future] | None = None
	transcriptions: deque[Future] = deque()  # in flight, in order
	pending: deque[str] = deque()            # turns waiting for the agent to finish the last one
	to_speak: deque[str] = deque()
	busy = False
	started_while_playing = barged = False

	def submit(text: str) -> None:
		nonlocal busy
		if busy:
			pending.append(text)
			say(f"{BOLD}you>{RESET} {text} {DIM}(queued){RESET}")
			return
		say(f"{BOLD}you>{RESET} {text}")
		agent.send({"type": "prompt", "text": text})
		busy = True
		narrator.submitted(time.monotonic())

	def interrupt() -> None:
		narration.mute()
		if speech is not None and not speech.done():
			heard = speech.interrupt()
			agent.send({"type": "heard", "text": heard})
			dim(f"  (cut off; heard: {heard!r})")
		speaker.stop()

	try:
		while True:
			now = time.monotonic()

			try:
				line = typed.get_nowait().strip()
			except queue.Empty:
				pass
			else:
				if line in ("exit", "quit"):
					return 0
				if line:
					submit(line)
				else:
					interrupt()

			while True:
				try:
					ev = agent.events.get_nowait()
				except queue.Empty:
					break
				kind = ev.get("type")
				if kind == "tool":
					dim(f"  ⚙ {ev['name']}")
					narrator.tool(ev["name"], now)
				elif kind == "answer":
					busy = False
					narrator.answered()
					narration.mute()
					text = ev.get("text") or ""
					to_speak.append(text if ev.get("ok") else f"Hmm, I hit a problem. {text}")
					if pending:
						submit(pending.popleft())
				elif kind == "exit":
					say(f"the agent exited ({ev.get('code')})")
					return 1
				elif kind == "error":
					dim(f"  bridge: {ev.get('text')}")

			if speech is not None and speech.done():
				speech = None
			if speech is None and to_speak:
				speech = Speech(tts, speaker, to_speak.popleft())

			while transcriptions and transcriptions[0].done():
				fut = transcriptions.popleft()
				try:
					text = usable_transcript(fut.result())
				except Exception as exc:  # noqa: BLE001
					dim(f"  (transcription failed: {exc})")
					continue
				if text:
					submit(text)
				else:
					dim("  (nothing heard)")

			try:
				frame = mic.frames.get(timeout=0.05)
			except queue.Empty:
				frame = None
			if frame is not None:
				p = vad(frame)
				playing = speaker.busy()
				if barge.feed(p, playing):
					barged = True
					interrupt()
				ev = endpointer.feed(p)
				if ev == "start":
					utterance = list(preroll) + [frame]
					started_while_playing, barged, speculation = playing, False, None
				elif utterance is not None:
					utterance.append(frame)
				preroll.append(frame)
				if ev == "speculate" and utterance is not None:
					speculation = (endpointer.speech_frames, pool.submit(stt.transcribe, pcm16(utterance)))
				elif ev == "end" and utterance is not None:
					if started_while_playing and not barged:
						dim("  (you spoke over me without cutting in: ignored)")
					elif speculation and speculation[0] == endpointer.speech_frames:
						transcriptions.append(speculation[1])
					else:
						transcriptions.append(pool.submit(stt.transcribe, pcm16(utterance)))
					utterance = speculation = None
				elif ev == "abort":
					utterance = speculation = None

			line = narrator.tick(now, speaking=(speech is not None) or speaker.busy() or endpointer.in_speech)
			if line:
				narration.offer(line)
	except KeyboardInterrupt:
		return 0
	finally:
		if speech is not None:
			speech.cancel.set()
		speaker.stop()
		narration.close()
		tts.close()
		agent.close()
		mic.close()
		speaker.close()
		pool.shutdown(wait=False)


if __name__ == "__main__":
	raise SystemExit(main())
