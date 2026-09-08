# AI_OWNED
"""Per-run logs and traces for the voice front end (CLAUDE.md, "Debugging the voice front end").

Every ./voice.sh run writes to logs/voice/<timestamp>/ on the host. Nothing there is ever
committed: the directory carries its own .gitignore. logs/voice/latest points at the
newest run.

	session.log     one timestamped line per thing that happened: what reached the terminal
	                and what did not (the agent's full stderr, the VAD's decisions, every
	                timing, every traceback). Read this first.
	events.jsonl    the same, as one JSON record per line, for grepping and scripting.
	turns/NNN.json  each agent turn as the bridge traced it: the prompt, every message the
	                loop added (tool calls, tool outputs), steps, stop reason, timing, usage.
	audio/uNNN.wav  each utterance exactly as it was sent to Whisper.

The client is the only writer. The bridge runs in the container, which mounts nothing, so
its trace rides back over stdout as events and lands here.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _render(fields: dict) -> str:
	return " ".join(f"{k}={v!r}" if not isinstance(v, str) else f"{k}={v}" for k, v in fields.items())


def _wav(pcm: bytes, rate: int) -> bytes:
	buf = io.BytesIO()
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(rate)
		w.writeframes(pcm)
	return buf.getvalue()


class RunLog:
	"""One run's directory. Every method is safe to call from any thread."""

	def __init__(self, root: Path, audio: bool = True):
		self.started = time.monotonic()
		root = Path(root)
		root.mkdir(parents=True, exist_ok=True)
		ignore = root / ".gitignore"
		if not ignore.exists():
			ignore.write_text("*\n")
		stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		self.dir = root / stamp
		n = 1
		while self.dir.exists():
			n += 1
			self.dir = root / f"{stamp}-{n}"
		(self.dir / "turns").mkdir(parents=True)
		self.audio_kept = audio
		if audio:
			(self.dir / "audio").mkdir()
		latest = root / "latest"
		try:
			if latest.is_symlink() or latest.exists():
				latest.unlink()
			os.symlink(self.dir.name, latest)
		except OSError:
			pass
		self.lock = threading.Lock()
		self.text = open(self.dir / "session.log", "a", encoding="utf-8")
		self.jsonl = open(self.dir / "events.jsonl", "a", encoding="utf-8")
		self.utterances = 0
		self.closed = False

	# Everything below takes the lock, and goes quiet once the run is closed.

	def event(self, kind: str, line: str | None = None, **fields) -> None:
		"""One record: `fields` in full in events.jsonl; `line` (or a rendering of the
		fields) in session.log."""
		t = time.monotonic() - self.started
		now = datetime.now()
		rec = {"t": round(t, 3), "ts": now.isoformat(timespec="milliseconds"), "kind": kind, **fields}
		text = _render(fields) if line is None else line
		with self.lock:
			if self.closed:
				return
			self.jsonl.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
			self.jsonl.flush()
			self.text.write(f"{now.strftime('%H:%M:%S.%f')[:-3]} {t:9.3f} {kind:<13} {text}\n")
			self.text.flush()

	def line(self, source: str, text: str) -> None:
		"""A raw line of output, from the terminal or the agent's pipes, as it was."""
		text = ANSI.sub("", text).rstrip("\n")
		self.event(source, line=text, text=text)

	def exception(self, where: str, exc: BaseException) -> None:
		tb = "".join(traceback.format_exception(exc))
		self.event("error", line=f"{where}: {type(exc).__name__}: {exc}", where=where,
		    error=f"{type(exc).__name__}: {exc}", traceback=tb)
		with self.lock:
			if not self.closed:
				self.text.write(tb if tb.endswith("\n") else tb + "\n")
				self.text.flush()

	def turn(self, trace: dict) -> Path:
		"""The bridge's trace of one agent turn, pretty-printed on its own, and summarized
		in the log with the file's name."""
		n = int(trace.get("turn") or 0)
		path = self.dir / "turns" / f"{n:03d}.json"
		with self.lock:
			path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str) + "\n")
		summary = {k: trace.get(k) for k in ("turn", "ok", "steps", "stop_reason", "elapsed_s")}
		summary["messages"] = len(trace.get("messages") or [])
		summary["usage"] = trace.get("usage")
		if trace.get("error"):
			summary["error"] = trace["error"]
		self.event("turn", line=f"{_render(summary)} -> {path.relative_to(self.dir)}",
		    file=str(path.relative_to(self.dir)), **summary)
		return path

	def utterance(self) -> str:
		"""Reserve the next utterance id, at the moment speech starts."""
		with self.lock:
			self.utterances += 1
			return f"u{self.utterances:03d}"

	def audio(self, uid: str, pcm16: bytes, rate: int) -> str | None:
		"""Keep one utterance as WAV; returns the file's name relative to the run, or None
		when audio logging is off."""
		if not self.audio_kept:
			return None
		path = self.dir / "audio" / f"{uid}.wav"
		try:
			path.write_bytes(_wav(pcm16, rate))
		except OSError as exc:
			self.event("error", where="audio", error=str(exc))
			return None
		return str(path.relative_to(self.dir))

	def close(self, reason: str = "") -> None:
		self.event("exit", reason=reason, seconds=round(time.monotonic() - self.started, 3))
		with self.lock:
			self.closed = True
			self.text.close()
			self.jsonl.close()
