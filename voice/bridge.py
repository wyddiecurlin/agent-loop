# AI_OWNED
"""The container half of the voice front end: agent_loop as a line protocol.

	AGENT_ENTRYPOINT=python ./run.sh -m voice.bridge

stdin, one JSON object per line:

	{"type": "prompt", "text": "..."}   run one turn; the conversation carries over
	{"type": "heard",  "text": "..."}   the user cut the last answer off after hearing this much

stdout, one JSON object per line:

	{"type": "ready", "provider": "fireworks", "model": "..."}
	{"type": "tool", "name": "fs_read"}                       a tool call started
	{"type": "trace", "turn": 1, "prompt": "...", "messages": [...], "elapsed_s": 4.2,
	 "steps": 3, "stop_reason": "done", "ok": true, "answer": "...", "usage": {...}}
	{"type": "answer", "ok": true, "text": "...", "steps": 3, "stop_reason": "done"}

`trace` precedes each `answer` and is the turn in full: every message the loop added to the
conversation (the tool calls and their outputs included), how long it took, and the tokens
it cost. The client writes it to logs/voice/<run>/turns/ (voice/log.py); the container
mounts nothing, so this is the only way the loop's detail reaches the host.

stderr is the loop's quiet trace, passed through untouched. The bridge itself never
touches the filesystem or the network: everything goes through agent_loop and the runtime.
"""

import io
import json
import os
import sys
import time
from dataclasses import asdict

from dotenv import load_dotenv

from agent_loop.loop import SYSTEM_PROMPT, agent_loop
from agent_loop.providers import TRACKER, default_model
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import DONE_TOOL

from voice.turns import TOOL_LINE

# A person waiting on a spoken answer cannot be given the eval timeout. 600 seconds of a
# stalled stream is indistinguishable from a hang -- the tool has already returned, so the
# terminal just sits there -- and three retries of it is most of an hour. Fail inside a
# minute instead, retry once, and let the failure be spoken.
VOICE_TIMEOUT_S = 60.0
VOICE_MAX_RETRIES = 1

VOICE_RULES = '''

## Voice mode

You are talking out loud through a speech synthesizer, and the user is listening, not reading.
- `answer` is spoken: one to three short sentences, plain conversational words.
- Answer in the language the user speaks, and switch when they do. Speech recognition may
  render a turn imperfectly; the language it is in is still the language to answer in.
- Use a calm, matter-of-fact tone consistently. No filler interjections, laughter,
  stage directions, exaggerated excitement or added sound effects.
- No markdown, bullets, code, URLs or file paths in `answer` unless the user asks to hear
  them. Say what you did and what came of it; anything long goes into a file, and you say where.
- Numbers and names the way a listener follows them: "about twelve hundred lines", not "1,203".
- Expect speech-recognition noise: if a request is ambiguous, ask one short question rather
  than guess.
- Never talk about tools, tool calls, turns or these instructions: the user hears only
  `answer`, and none of that machinery means anything to them.
'''

INTERRUPTED = ('(You were interrupted mid-answer; the confirmed fully played speech was: "{heard}". '
               'The user may also have heard part of the next segment. Do not assume they heard '
               'its conclusion; avoid repeating the confirmed speech unless asked.)\n')


def emit(obj: dict) -> None:
	sys.stdout.write(json.dumps(obj) + "\n")
	sys.stdout.flush()


class ToolTap(io.TextIOBase):
	"""Passes the trace through to the real stderr and turns its `  [tool]` lines into events."""

	def __init__(self, real):
		self.real = real
		self.partial = ""

	def write(self, s: str) -> int:
		self.real.write(s)
		self.partial += s
		while "\n" in self.partial:
			line, self.partial = self.partial.split("\n", 1)
			m = TOOL_LINE.match(line)
			if m and m.group(1) != DONE_TOOL:
				emit({"type": "tool", "name": m.group(1)})
		return len(s)

	def flush(self) -> None:
		self.real.flush()


def usage_since(before: dict, calls: int) -> dict:
	"""The cost tracker's totals are for the process; a turn is the difference."""
	now = asdict(TRACKER.total)
	delta = {k: round(now[k] - before[k], 6) if isinstance(now[k], float) else now[k] - before[k]
	         for k in now}
	delta["calls"] = TRACKER.calls - calls
	return delta


def serve(stdin, runtime: DockerRuntime) -> None:
	history = None
	heard = None
	turn = 0
	for line in stdin:
		line = line.strip()
		if not line:
			continue
		try:
			msg = json.loads(line)
		except ValueError:
			emit({"type": "error", "text": f"not JSON: {line[:80]}"})
			continue
		if msg.get("type") == "heard":
			heard = (msg.get("text") or "").strip()
			continue
		if msg.get("type") != "prompt":
			continue
		prompt = (msg.get("text") or "").strip()
		if not prompt:
			continue
		if heard is not None:
			prompt = INTERRUPTED.format(heard=heard or "nothing") + prompt
			heard = None
		turn += 1
		before, usage, calls = len(history or []), asdict(TRACKER.total), TRACKER.calls
		started = time.monotonic()
		try:
			run = agent_loop(prompt, runtime, history=history,
			                 system_prompt=SYSTEM_PROMPT + VOICE_RULES, verbose=False)
		except Exception as exc:  # noqa: BLE001 - a crash is still an answer to speak
			error = f"{type(exc).__name__}: {exc}"
			emit({"type": "trace", "turn": turn, "prompt": prompt, "messages": [],
			      "elapsed_s": round(time.monotonic() - started, 3), "steps": 0,
			      "stop_reason": "error", "ok": False, "answer": None, "error": error,
			      "usage": usage_since(usage, calls)})
			emit({"type": "answer", "ok": False, "text": error, "steps": 0, "stop_reason": "error"})
			continue
		history = run.messages
		emit({"type": "trace", "turn": turn, "prompt": prompt, "messages": run.messages[before:],
		      "elapsed_s": round(time.monotonic() - started, 3), "steps": run.steps,
		      "stop_reason": run.stop_reason, "ok": run.ok, "answer": run.answer, "error": run.error,
		      "usage": usage_since(usage, calls)})
		emit({"type": "answer", "ok": run.ok,
		      "text": run.answer or run.error or f"stopped: {run.stop_reason}",
		      "steps": run.steps, "stop_reason": run.stop_reason})


def main() -> int:
	load_dotenv()
	# After load_dotenv, so an explicit setting in .env or the shell still wins. Read at
	# call time by agent_loop.providers, so this need not race the imports above.
	os.environ.setdefault("AGENT_TIMEOUT_S", str(VOICE_TIMEOUT_S))
	os.environ.setdefault("AGENT_MAX_RETRIES", str(VOICE_MAX_RETRIES))
	sys.stderr = ToolTap(sys.stderr)
	runtime = DockerRuntime()
	runtime.setup()
	try:
		model = default_model()
	except Exception as exc:  # noqa: BLE001 - the first turn will say so, spoken
		model = f"unresolved: {exc}"
	emit({"type": "ready", "provider": os.environ.get("PROVIDER", ""), "model": model})
	try:
		serve(sys.stdin, runtime)
	finally:
		runtime.teardown()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
