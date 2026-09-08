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

	./voice.sh [--voice ryan] [--lang auto] [--api https://api.lemontree.media/v1]

The one program in the repo that runs on the host, because a container on macOS cannot
reach the microphone. It does audio and nothing else: the agent still runs in the
container, started through ./run.sh with voice/bridge.py as the entrypoint, and this
process talks to it over stdin/stdout.

	mic -> Silero VAD -> end of turn -> Whisper -> bridge -> agent_loop
	                                                 \\-> [tool] events -> "I'm reading the file."
	answer -> Qwen3-TTS websocket -> speaker, cut the moment the user talks over it

Keys: Enter interrupts; a typed line is a turn without the microphone; Ctrl-C quits.

Every run is logged in full to logs/voice/<timestamp>/ (voice/log.py): what was said and
heard, every VAD and endpoint decision, the agent's stderr, each turn's messages, timings.
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

from voice.log import RunLog  # noqa: E402
from voice.speech import (  # noqa: E402
	DEFAULT_API, DEFAULT_INSTRUCTIONS, STT, TTS, TTS_RATE, Clips, ensure_silero, stt_language,
	tts_language,
)
from voice.turns import (  # noqa: E402
	MAX_SPOKEN_S, STATUS_LINES, TOOL_LINE, BargeIn, EndpointConfig, Endpointer, Narrator,
	heard_text, spoken_answer, status_language, usable_transcript,
)
from voice.vad import FRAME, SAMPLE_RATE, SileroVAD  # noqa: E402

CACHE = Path(os.environ.get("AGENT_VOICE_CACHE", Path.home() / ".cache" / "agent-voice"))
LOG_DIR = Path(os.environ.get("VOICE_LOG_DIR", REPO / "logs" / "voice"))
LOG: RunLog | None = None  # the run's log, once main() has opened it
# Whisper decodes in the vocabulary it is primed with, so it is told the subject. The prompt
# must be in the language of the audio; with the language left to detection it is given both,
# because a prompt in one language alone pulls the transcript towards that language.
STT_PROMPTS = {
	"en": "A developer talking to a coding agent about files, tests, git, Python, and the shell.",
	"zh": "一位开发者在和编程助手讨论文件、测试、git、Python 和终端命令。",
}
STT_PROMPTS["auto"] = f"{STT_PROMPTS['en']} {STT_PROMPTS['zh']}"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def say(text: str) -> None:
	print(text, flush=True)
	if LOG:
		LOG.line("tty", text)


def dim(text: str) -> None:
	print(f"{DIM}{text}{RESET}", flush=True)
	if LOG:
		LOG.line("tty", text)


def log(kind: str, **fields) -> None:
	if LOG:
		LOG.event(kind, **fields)


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
				ev = json.loads(line)
			except ValueError:
				dim(f"  {line}")
				continue
			if ev.get("type") == "trace":  # the turn in full: for the log, not the loop
				if LOG:
					LOG.turn(ev)
				continue
			self.events.put(ev)
		self.events.put({"type": "exit", "code": self.proc.wait()})

	def _read_stderr(self) -> None:
		for line in self.proc.stderr:
			line = line.rstrip("\n")
			if not line:
				continue
			if LOG:
				LOG.line("agent.err", line)
			if not TOOL_LINE.match(line):  # tool lines arrive as events already
				dim(f"  {line}")

	def send(self, obj: dict) -> None:
		log("send", **obj)
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

	def __init__(self, tts: TTS, speaker: Speaker, text: str, spoken: str | None = None):
		self.speaker = speaker
		self.cancel = threading.Event()
		self.sentences: list = []
		self.start: int | None = None
		self.thread = threading.Thread(target=self._run, args=(tts, text, spoken or text), daemon=True)
		self.thread.start()

	def _run(self, tts: TTS, text: str, spoken: str) -> None:
		deadline = time.monotonic() + 0.6  # let a filler finish rather than clip it, briefly
		while self.speaker.busy() and time.monotonic() < deadline and not self.cancel.is_set():
			time.sleep(0.02)
		self.speaker.stop()
		if self.cancel.is_set():
			log("speak.skipped", text=text)
			return
		self.start = self.speaker.enqueued
		say(f"{BOLD}agent>{RESET} {text}")  # preserve code and formatting in the text answer
		if spoken != text:
			dim("  (too long to read out; it is above in full)")
		t0 = time.monotonic()
		log("speak", text=spoken, truncated=spoken != text)
		try:
			tts.speak(spoken, self.speaker, self.cancel, sentences=self.sentences,
			          on_sentence=lambda s: log("tts.sentence", after_s=round(time.monotonic() - t0, 3), text=s))
			log("speak.cut" if self.cancel.is_set() else "speak.done",
			    seconds=round(time.monotonic() - t0, 3), sentences=len(self.sentences))
		except Exception as exc:  # noqa: BLE001
			if not self.cancel.is_set():
				dim(f"  (speech stopped: {exc})")
				if LOG:
					LOG.exception("tts", exc)

	def done(self) -> bool:
		return not self.thread.is_alive()

	def interrupt(self) -> str:
		"""Cut playback and return what was heard, so the agent's history can say so."""
		heard = "" if self.start is None else heard_text(self.sentences, self.speaker.consumed)
		self.cancel.set()
		self.speaker.stop()
		return heard


class Narration:
	"""Plays the short lines the Narrator picks, from the clip cache, off the audio thread."""

	def __init__(self, clips: Clips, speaker: Speaker, answer_playing):
		self.clips, self.speaker, self.answer_playing = clips, speaker, answer_playing
		self.q: queue.Queue[tuple[int, str] | None] = queue.Queue()
		self.lock = threading.Lock()
		self.generation = 0
		self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def offer(self, text: str) -> None:
		with self.lock:
			self.q.put((self.generation, text))

	def mute(self) -> None:
		with self.lock:
			self.generation += 1  # invalidate in-flight synthesis as well as queued lines
			while not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					break

	def close(self) -> None:
		self.mute()
		self.q.put(None)

	def _run(self) -> None:
		while True:
			item = self.q.get()
			if item is None:
				return
			generation, text = item
			with self.lock:
				if generation != self.generation or self.answer_playing():
					continue
			try:
				pcm = self.clips.get(text)
			except Exception as exc:  # noqa: BLE001
				dim(f"  (filler failed: {exc})")
				if LOG:
					LOG.exception("filler", exc)
				continue
			with self.lock:
				if generation == self.generation and not self.answer_playing():
					dim(f"  {text}")
					self.speaker.play(pcm)


# --- the loop -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="talk to the agent")
	ap.add_argument("--api", default=os.environ.get("VOICE_API", DEFAULT_API))
	ap.add_argument("--voice", default=os.environ.get("VOICE", "ryan"))
	ap.add_argument("--lang", default=os.environ.get("VOICE_LANG", "auto"),
	                help="what you speak, as an ISO code (en, zh, ...); auto detects it each turn")
	ap.add_argument("--tts-lang", default=None, help="TTS language; defaults to --lang, or explicitly Auto")
	ap.add_argument("--stt-prompt", default=None, help="words to prime Whisper with, in the language you speak")
	ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS, help="style for both answers and status clips")
	ap.add_argument("--no-narration", action="store_true", help="speak answers only, without status clips")
	ap.add_argument("--max-answer-s", type=float, default=MAX_SPOKEN_S,
	                help="answers longer than this are left on screen instead of read out")
	ap.add_argument("--endpoint-ms", type=int, default=600, help="silence that ends your turn")
	ap.add_argument("--max-turn-ms", type=int, default=60_000, help="longest turn held before it is taken as said")
	ap.add_argument("--mic", default=None, help="input device name or index")
	ap.add_argument("--speaker", default=None, help="output device name or index")
	ap.add_argument("--list-devices", action="store_true")
	ap.add_argument("--log-dir", default=LOG_DIR, type=Path,
	                help="where each run's logs and traces go (default: logs/voice under the repo)")
	ap.add_argument("--no-log-audio", action="store_true", help="do not keep each utterance's WAV in the log")
	args = ap.parse_args(argv)
	if args.list_devices:
		print(sd.query_devices())
		return 0
	global LOG
	LOG = RunLog(args.log_dir, audio=not args.no_log_audio)
	dim(f"logging to {LOG.dir}")
	try:
		code = run(args, ap)
	except KeyboardInterrupt:
		LOG.close("ctrl-c")
		return 0
	except SystemExit as exc:
		LOG.close(f"exit {exc.code}")
		raise
	except BaseException as exc:
		LOG.exception("main", exc)
		LOG.close(f"crash: {type(exc).__name__}")
		raise
	LOG.close(f"exit {code}")
	return code


def run(args, ap) -> int:
	log("start", argv=sys.argv[1:], args={k: str(v) for k, v in vars(args).items()},
	    cwd=os.getcwd(), python=sys.version.split()[0], pid=os.getpid(),
	    provider=os.environ.get("PROVIDER", ""), model=os.environ.get("MODEL", ""))
	try:
		language = tts_language(args.tts_lang or args.lang)
	except ValueError as exc:
		ap.error(str(exc))
	heard_lang = stt_language(args.lang)
	prompt = args.stt_prompt if args.stt_prompt is not None else STT_PROMPTS.get(heard_lang or "auto", "")
	# Status lines follow whatever the answers are synthesized as; with detection on, they
	# follow the script of each turn instead, so a Chinese turn is not answered by an
	# English "Let me check." over the top of a Chinese answer.
	spoken_status = {"English": "en", "Chinese": "zh"}.get(language, None if language == "Auto" else "en")

	vad = SileroVAD(ensure_silero(CACHE))
	speaker = Speaker(args.speaker)
	clips = Clips(args.api, args.voice, language, CACHE / "clips", args.instructions)
	if not args.no_narration:
		wanted = STATUS_LINES if spoken_status is None else {spoken_status: STATUS_LINES[spoken_status]}
		clips.prefetch(list(dict.fromkeys(l for lines in wanted.values() for l in lines.values())))
	tts = TTS(args.api, args.voice, language, args.instructions)
	stt = STT(args.api, args.lang, prompt)
	log("config", api=args.api, voice=args.voice, tts_language=language, stt_language=heard_lang or "auto",
	    stt_prompt=prompt, status_language=spoken_status, instructions=args.instructions, cache=str(CACHE))

	def transcribe(pcm: bytes, uid: str | None) -> str:
		"""stt.transcribe, timed and logged. Runs on the pool, so only when it was not discarded."""
		t0 = time.monotonic()
		try:
			text = stt.transcribe(pcm)
		except Exception as exc:  # noqa: BLE001
			log("stt.error", utterance=uid, seconds=round(len(pcm) / (SAMPLE_RATE * 2), 2),
			    latency_s=round(time.monotonic() - t0, 3), error=f"{type(exc).__name__}: {exc}")
			raise
		log("stt", utterance=uid, seconds=round(len(pcm) / (SAMPLE_RATE * 2), 2),
		    latency_s=round(time.monotonic() - t0, 3), text=text)
		return text
	# One speculation running, one stale one still winding down, and the turn itself.
	pool = ThreadPoolExecutor(max_workers=3)

	dim("starting the agent container ...")
	t0 = time.monotonic()
	agent = Agent()
	while True:
		ev = agent.events.get()
		if ev.get("type") == "ready":
			log("agent.ready", after_s=round(time.monotonic() - t0, 3), provider=ev.get("provider"),
			    model=ev.get("model"))
			break
		if ev.get("type") == "exit":
			say(f"the agent did not start (exit {ev.get('code')})")
			return 1
	mic = Mic(args.mic)  # macOS asks for microphone permission here, once per terminal app
	for kind, dev in (("input", args.mic), ("output", args.speaker)):
		try:
			log("device", kind=kind, name=sd.query_devices(dev, kind)["name"],
			    decimate=mic.decimate if kind == "input" else 1)
		except Exception as exc:  # noqa: BLE001 - a log line, not a requirement
			log("device", kind=kind, error=str(exc))
	say(f"listening as {args.voice} (hearing {heard_lang or 'any language'}, speaking {language}). "
	    "Enter interrupts, a typed line is a turn, Ctrl-C quits.")

	typed: queue.Queue[str] = queue.Queue()
	threading.Thread(target=lambda: [typed.put(l) for l in sys.stdin], daemon=True).start()

	endpointer = Endpointer(EndpointConfig(end_ms=args.endpoint_ms, max_turn_ms=args.max_turn_ms))
	barge = BargeIn()
	narrator = Narrator(language=spoken_status or "en")
	speech: Speech | None = None
	narration = Narration(clips, speaker, lambda: speech is not None and not speech.done())

	preroll: deque[np.ndarray] = deque(maxlen=12)  # ~380 ms before the VAD agrees it is speech
	utterance: list[np.ndarray] | None = None
	speculation: tuple[int, Future] | None = None
	transcriptions: deque[Future] = deque()  # in flight, in order
	pending: deque[str] = deque()            # turns waiting for the agent to finish the last one
	to_speak: deque[tuple[str, str]] = deque()
	busy = False
	started_while_playing = barged = False
	turns = 0
	submitted_at = 0.0
	utterance_id: str | None = None

	def submit(text: str) -> None:
		nonlocal busy, turns, submitted_at
		if busy:
			pending.append(text)
			say(f"{BOLD}you>{RESET} {text} {DIM}(queued){RESET}")
			log("prompt.queued", text=text, waiting=len(pending))
			return
		say(f"{BOLD}you>{RESET} {text}")
		turns += 1
		log("prompt", turn=turns, text=text)
		agent.send({"type": "prompt", "text": text})
		busy = True
		submitted_at = time.monotonic()
		if spoken_status is None:
			narrator.language = status_language(text)
		narrator.submitted(submitted_at)

	def discard(spec) -> None:
		"""Drop a speculative transcription the turn has outgrown, freeing its worker if it
		has not started. A minute of speech pauses a dozen times; without this, each pause
		leaves a full upload of everything said so far in front of the one that matters."""
		if spec is not None:
			spec[1].cancel()

	def interrupt(why: str) -> None:
		narration.mute()
		if speech is not None and not speech.done():
			heard = speech.interrupt()
			log("interrupt", why=why, heard=heard)
			agent.send({"type": "heard", "text": heard})
			dim(f"  (cut off; heard: {heard!r})")
		else:
			log("interrupt", why=why, heard=None)
		speaker.stop()

	try:
		while True:
			now = time.monotonic()

			try:
				line = typed.get_nowait().strip()
			except queue.Empty:
				pass
			else:
				log("typed", text=line)
				if line in ("exit", "quit"):
					return 0
				if line:
					submit(line)
				else:
					interrupt("enter")

			while True:
				try:
					ev = agent.events.get_nowait()
				except queue.Empty:
					break
				kind = ev.get("type")
				if kind == "tool":
					dim(f"  ⚙ {ev['name']}")
					log("tool", name=ev["name"], after_s=round(now - submitted_at, 3))
					narrator.tool(ev["name"], now)
				elif kind == "answer":
					busy = False
					narrator.answered()
					narration.mute()
					text = ev.get("text") or ""
					log("answer", turn=turns, ok=ev.get("ok"), steps=ev.get("steps"),
					    stop_reason=ev.get("stop_reason"), wait_s=round(now - submitted_at, 3), text=text)
					if not ev.get("ok"):
						dim(f"  agent error: {text}")
						text = "I hit a problem. The details are in the terminal."
					# The whole answer is printed either way; only what is said out loud is
					# cut, and the listener is told where the rest of it is.
					answer_lang = spoken_status or status_language(text)
					to_speak.append((text, spoken_answer(text, answer_lang, args.max_answer_s)))
					if pending:
						submit(pending.popleft())
				elif kind == "exit":
					say(f"the agent exited ({ev.get('code')})")
					return 1
				elif kind == "error":
					dim(f"  bridge: {ev.get('text')}")
					log("bridge.error", text=ev.get("text"))
				else:
					log("agent.event", **ev)

			if speech is not None and speech.done():
				speech = None
			if speech is None and to_speak:
				speech = Speech(tts, speaker, *to_speak.popleft())

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
					log("stt.empty")

			try:
				frame = mic.frames.get(timeout=0.05)
			except queue.Empty:
				frame = None
			if frame is not None:
				p = vad(frame)
				playing = speaker.busy()
				if barge.feed(p, playing):
					barged = True
					interrupt("barge-in")
				ev = endpointer.feed(p)
				if speculation and speculation[0] != endpointer.speech_frames:
					log("stt.stale", speech_frames=speculation[0])
					discard(speculation)  # more was said since: that transcript is stale now
					speculation = None
				if ev == "start":
					narration.mute()
					utterance = list(preroll) + [frame]
					started_while_playing, barged, speculation = playing, False, None
					utterance_id = LOG.utterance() if LOG else None
					log("vad.start", utterance=utterance_id, playing=playing, p=round(float(p), 3))
				elif utterance is not None:
					utterance.append(frame)
				preroll.append(frame)
				if ev == "speculate" and utterance is not None:
					log("stt.speculate", speech_frames=endpointer.speech_frames,
					    ms=len(utterance) * FRAME * 1000 // SAMPLE_RATE)
					speculation = (endpointer.speech_frames, pool.submit(transcribe, pcm16(utterance), utterance_id))
				elif ev == "end" and utterance is not None:
					pcm = pcm16(utterance)
					log("vad.end", utterance=utterance_id, ms=len(pcm) * 1000 // (SAMPLE_RATE * 2),
					    file=LOG.audio(utterance_id, pcm, SAMPLE_RATE) if LOG else None,
					    speech_frames=endpointer.speech_frames, started_while_playing=started_while_playing,
					    barged=barged, speculated=speculation is not None)
					if started_while_playing and not barged:
						dim("  (you spoke over me without cutting in: ignored)")
						log("stt.ignored", utterance=utterance_id, why="spoke over playback without barging in")
						discard(speculation)
					elif speculation and speculation[0] == endpointer.speech_frames:
						transcriptions.append(speculation[1])
					else:
						transcriptions.append(pool.submit(transcribe, pcm, utterance_id))
					utterance = speculation = None
				elif ev == "abort":
					log("vad.abort", utterance=utterance_id, frames=len(utterance) if utterance else 0)
					utterance = speculation = None

			line = narrator.tick(now, speaking=(speech is not None) or speaker.busy() or endpointer.in_speech)
			if line and not args.no_narration:
				log("narrate", text=line)
				narration.offer(line)
	except KeyboardInterrupt:
		log("signal", name="ctrl-c")
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
