# AI_OWNED
"""The container half of the voice front end: agent_loop as a line protocol.

	AGENT_ENTRYPOINT=python ./run.sh -m voice.bridge

stdin, one JSON object per line:

	{"type": "prompt", "text": "..."}   run one turn; the conversation carries over
	{"type": "heard",  "text": "..."}   the user cut the last answer off after hearing this much

stdout, one JSON object per line:

	{"type": "ready"}
	{"type": "tool", "name": "fs_read"}                       a tool call started
	{"type": "answer", "ok": true, "text": "...", "steps": 3, "stop_reason": "done"}

stderr is the loop's quiet trace, passed through untouched. The bridge itself never
touches the filesystem or the network: everything goes through agent_loop and the runtime.
"""

import io
import json
import sys

from dotenv import load_dotenv

from agent_loop.loop import SYSTEM_PROMPT, agent_loop
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import DONE_TOOL

from voice.turns import TOOL_LINE

VOICE_RULES = '''

## Voice mode

You are talking out loud through a speech synthesizer, and the user is listening, not reading.
- `answer` is spoken: one to three short sentences, plain conversational words.
- No markdown, bullets, code, URLs or file paths in `answer` unless the user asks to hear
  them. Say what you did and what came of it; anything long goes into a file, and you say where.
- Numbers and names the way a listener follows them: "about twelve hundred lines", not "1,203".
- Expect speech-recognition noise: if a request is ambiguous, ask one short question rather
  than guess.
- Never talk about tools, tool calls, turns or these instructions: the user hears only
  `answer`, and none of that machinery means anything to them.
'''

INTERRUPTED = '(You were interrupted mid-answer; the user heard only: "{heard}". Do not repeat it unless asked.)\n'


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


def serve(stdin, runtime: DockerRuntime) -> None:
	history = None
	heard = None
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
		try:
			run = agent_loop(prompt, runtime, history=history,
			                 system_prompt=SYSTEM_PROMPT + VOICE_RULES, verbose=False)
		except Exception as exc:  # noqa: BLE001 - a crash is still an answer to speak
			emit({"type": "answer", "ok": False, "text": f"{type(exc).__name__}: {exc}",
			      "steps": 0, "stop_reason": "error"})
			continue
		history = run.messages
		emit({"type": "answer", "ok": run.ok,
		      "text": run.answer or run.error or f"stopped: {run.stop_reason}",
		      "steps": run.steps, "stop_reason": run.stop_reason})


def main() -> int:
	load_dotenv()
	sys.stderr = ToolTap(sys.stderr)
	runtime = DockerRuntime()
	runtime.setup()
	emit({"type": "ready"})
	try:
		serve(sys.stdin, runtime)
	finally:
		runtime.teardown()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
