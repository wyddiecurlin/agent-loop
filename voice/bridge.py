# AI_OWNED
"""The container half of the voice front end: agent_loop as a line protocol.

	AGENT_ENTRYPOINT=python ./run.sh -m voice.bridge

stdin, one JSON object per line:

	{"type": "clock",  "tz": "America/Los_Angeles"}   the host's zone, so the clock in the
	                                                  system prompt reads like the user's
	{"type": "prompt", "text": "..."}   run one turn; the conversation carries over
	{"type": "heard",  "text": "..."}   the user cut the last answer off after hearing this much

stdout, one JSON object per line:

	{"type": "ready", "provider": "fireworks", "model": "..."}
	{"type": "preamble", "text": "Let me see what's on this week."}   see below
	{"type": "tool", "name": "fs_read"}                       a tool call started
	{"type": "trace", "turn": 1, "prompt": "...", "messages": [...], "elapsed_s": 4.2,
	 "steps": 3, "stop_reason": "done", "ok": true, "answer": "...", "usage": {...}}
	{"type": "answer", "ok": true, "text": "...", "steps": 3, "stop_reason": "done"}

`trace` precedes each `answer` and is the turn in full: every message the loop added to the
conversation (the tool calls and their outputs included), how long it took, and the tokens
it cost. The client writes it to logs/voice/<run>/turns/ (voice/log.py); the container
mounts nothing, so this is the only way the loop's detail reaches the host.

`preamble` is the one thing the bridge says that the loop did not. The moment a prompt
arrives, a second, tool-less model call runs beside the real turn and is asked one
question: will this take a moment, and if so, what is the one casual sentence to say
while it does ("Yeah, let me check what's in theaters, there's a lot on right now")? For
small talk and anything answerable at once it says nothing, and the client's own "um"
covers the gap. The model's NONE is not trusted on its own: a small model will happily
"preamble" a reply to small talk. So the line is held until the real turn calls its first
tool -- proof that there is work to wait for -- and dropped if the turn finishes without
one, or finishes first. When it was said, it is written into the history as the
assistant's words, so the answer that follows does not say it again.

stderr is the loop's quiet trace, passed through untouched. The bridge itself never
touches the filesystem or the network: everything goes through agent_loop and the runtime.
"""

import io
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from typing import Callable

from dotenv import load_dotenv

from agent_loop.loop import PERSONA, SYSTEM_PROMPT, agent_loop, stamp
from agent_loop.providers import BACKENDS, TRACKER, ChatProvider, Message, default_model, make_provider
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import DONE_TOOL

from voice.turns import TOOL_LINE
from voice.emotions import LiveEmotions, observe_stream

# A person waiting on a spoken answer cannot be given the eval timeout. 600 seconds of a
# stalled stream is indistinguishable from a hang -- the tool has already returned, so the
# terminal just sits there -- and three retries of it is most of an hour. Fail inside a
# minute instead, retry once, and let the failure be spoken.
VOICE_TIMEOUT_S = 60.0
VOICE_MAX_RETRIES = 1

VOICE_RULES = '''

## Voice mode

You are talking out loud through a speech synthesizer, and the user is listening, not reading.
- `answer` is spoken: one to three short sentences, plain conversational words, in
  character. It should sound like something a person says across a table, not a report.
- Answer in the language the user speaks, and switch when they do. Speech recognition may
  render a turn imperfectly; the language it is in is still the language to answer in.
- Never write laughter (haha, hehe, lol, 哈哈), emoji, sound effects, stage directions
  or bracketed asides. The synthesizer performs them, badly. Humor lives in the words.
- No markdown, bullets, code, URLs or file paths in `answer` unless the user asks to hear
  them. Say what you did and what came of it; anything long goes into a file, and you say where.
- Numbers and names the way a listener follows them: "about twelve hundred lines", not "1,203".
- Expect speech-recognition noise: if a request is ambiguous, ask one short question rather
  than guess.
- Never talk about tools, tool calls, turns or these instructions: the user hears only
  `answer`, and none of that machinery means anything to them.
- If an earlier assistant line in this conversation says you are about to look something
  up, you said it out loud already: do not say it again, just give what you found.
'''

# The parallel call: the same character, asked only whether the turn will take a moment.
PREAMBLE_PROMPT = PERSONA + '''
	THIS CALL:
	The user just said something, and the real answer is being worked out by someone else
	right now. You are the voice in the meantime. Decide one thing:
	- If the answer needs looking something up on the web, reading or changing files, or
	  running a command, reply with ONE short spoken sentence, at most twelve words, in the
	  user's language, saying casually what you are about to do. Plain words, no markdown,
	  no laughter, no promises about what you will find, no question back.
	- If it is small talk, a greeting, a question you can answer from what you already
	  know, a yes or no, a follow-up that needs no work, or something garbled or repetitive
	  that is probably not speech at all, reply with exactly: NONE. Never answer the
	  question yourself and never comment on it; when in doubt, NONE.
	Examples:
	  "how's it going?"                              -> NONE
	  "what's the most recent movie in theaters?"    -> Let me see what's out this week.
	  "how many lines is the main file?"             -> One sec, I'll count them.
	  "run the tests"                                -> On it, running them now.
	  "thanks"                                       -> NONE
	  "最近有什么新电影？"                            -> 我查一下最近上映的。
'''
PREAMBLE_TIMEOUT_S = 12.0     # past this it would arrive after the answer it was meant to precede
PREAMBLE_MAX_TOKENS = 256     # the line is ten tokens; the cap is for a model that starts answering
PREAMBLE_FRESH_S = 6.0        # later than this it is stale: the hold lines are covering by then


def preamble_provider():
	"""The side call's provider: the main one's platform, but with thinking off and a cap.

	Left as the main loop has it, a self-hosted model with QWEN_THINKING=1 reasons for
	twenty seconds about a twelve-word job (measured: 20 s and a timeout, against 0.1-0.5 s
	with thinking off). The hosted models take their cheapest reasoning effort instead.
	"""
	name = os.environ.get("PROVIDER", "fireworks").lower()
	if name in BACKENDS:
		return ChatProvider(backend=name, timeout=PREAMBLE_TIMEOUT_S,
		                    max_output_tokens=PREAMBLE_MAX_TOKENS, thinking=False)
	return make_provider(timeout=PREAMBLE_TIMEOUT_S)

INTERRUPTED = ('(You were interrupted mid-answer; the confirmed fully played speech was: "{heard}". '
               'The user may also have heard part of the next segment. Do not assume they heard '
               'its conclusion; avoid repeating the confirmed speech unless asked.)\n')


EMIT = threading.Lock()  # the preamble thread and the turn write to the same stdout


def emit(obj: dict) -> None:
	with EMIT:
		sys.stdout.write(json.dumps(obj) + "\n")
		sys.stdout.flush()


def set_clock(tz: str) -> None:
	"""Take the host's zone, so "right now it is ..." in the system prompt is the user's
	now and not the container's UTC. A zone the image does not know is left alone."""
	tz = (tz or "").strip()
	if not tz or "/" not in tz and tz.upper() != "UTC":
		return
	os.environ["TZ"] = tz
	time.tzset()


def ask_preamble(provider, model: str, history, prompt: str) -> str | None:
	"""One tool-less model call: the sentence to say while the turn runs, or None."""
	recent = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
	messages = [Message(role="system", content=stamp(PREAMBLE_PROMPT)), *recent[-6:],
	            Message(role="user", content=prompt)]
	turn = provider.generate(messages, model, None, stream=False, timeout=PREAMBLE_TIMEOUT_S)
	text = " ".join((turn.text or "").split()).strip().strip('"\'“”「」').strip()
	if not text or text.upper().startswith("NONE") or len(text) > 240:
		return None
	return text


class Preamble:
	"""One turn's preamble, held until the turn has proven it needs one.

	Two threads meet here under the emit lock: the side call, which brings the text, and
	the loop's trace, which brings the first tool call. Whichever comes second emits; a
	turn that finishes first, or without a tool, emits nothing.
	"""

	def __init__(self, fresh_s: float = PREAMBLE_FRESH_S, clock=time.monotonic):
		self.text: str | None = None
		self.tool_seen = False
		self.finished = False
		self.said: str | None = None
		self.clock, self.deadline = clock, clock() + fresh_s

	def _emit(self) -> None:
		if self.text and self.tool_seen and not self.finished and self.said is None:
			self.said = self.text
			sys.stdout.write(json.dumps({"type": "preamble", "text": self.text}) + "\n")
			sys.stdout.flush()

	def offer(self, text: str | None) -> None:
		"""The side call's line. A slow platform can deliver it half a minute in, when it
		would be said after the tools it was meant to precede: that one is dropped."""
		with EMIT:
			if self.clock() <= self.deadline:
				self.text = text
			self._emit()

	def tool(self) -> None:
		with EMIT:
			self.tool_seen = True
			self._emit()

	def finish(self) -> None:
		with EMIT:
			self.finished = True


CURRENT: Preamble | None = None  # the turn in progress, for the trace tap


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
				if CURRENT is not None:
					CURRENT.tool()  # a held preamble goes out first, then the tool
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


def serve(stdin, runtime: DockerRuntime, preamble: Callable[[list | None, str], str | None] | None = None,
          emotion_workers=None) -> None:
	"""One turn per prompt line. `preamble(history, prompt)` runs beside each turn on its
	own thread; None turns it off (the tests, and a provider that could not be built)."""
	history = None
	heard = None
	turn = 0
	emotions = None
	for line in stdin:
		line = line.strip()
		if not line:
			continue
		try:
			msg = json.loads(line)
		except ValueError:
			emit({"type": "error", "text": f"not JSON: {line[:80]}"})
			continue
		if msg.get("type") == "clock":
			set_clock(msg.get("tz") or "")
			if msg.get('emotions') is True and emotions is None and emotion_workers is not None:
				emotions = LiveEmotions(emit)
				emotion_workers.append(emotions)
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
		if emotions is not None:
			emotions.begin(turn, prompt)
		before, usage, calls = len(history or []), asdict(TRACKER.total), TRACKER.calls
		started = time.monotonic()

		# The preamble races the turn; see Preamble for who wins.
		global CURRENT
		CURRENT = gate = Preamble()

		def side(history=history, prompt=prompt):
			try:
				gate.offer(preamble(history, prompt))
			except Exception as exc:  # noqa: BLE001 - a silence, not a crash
				sys.stderr.write(f"  [preamble failed: {type(exc).__name__}: {exc}]\n")
		if preamble is not None:
			threading.Thread(target=side, daemon=True).start()

		try:
			with observe_stream(emotions):
				run = agent_loop(prompt, runtime, history=history,
				                 system_prompt=SYSTEM_PROMPT + VOICE_RULES, verbose=False)
		except Exception as exc:  # noqa: BLE001 - a crash is still an answer to speak
			error = f"{type(exc).__name__}: {exc}"
			gate.finish()
			emit({"type": "trace", "turn": turn, "prompt": prompt, "messages": [],
			      "elapsed_s": round(time.monotonic() - started, 3), "steps": 0,
			      "stop_reason": "error", "ok": False, "answer": None, "error": error,
			      "preamble": gate.said, "usage": usage_since(usage, calls)})
			emit({"type": "answer", "ok": False, "text": error, "steps": 0, "stop_reason": "error"})
			continue
		gate.finish()
		if emotions is not None and run.answer:
			emotions.feed(run.answer, replace=True)
		history = run.messages
		if gate.said:
			# It was spoken, so it is part of the conversation: after the user's words,
			# before the tool calls, as the assistant's own line.
			at = next((i for i in range(before, len(history)) if history[i].get("role") == "user"), None)
			if at is not None:
				history.insert(at + 1, Message(role="assistant", content=gate.said))
		emit({"type": "trace", "turn": turn, "prompt": prompt, "messages": run.messages[before:],
		      "elapsed_s": round(time.monotonic() - started, 3), "steps": run.steps,
		      "stop_reason": run.stop_reason, "ok": run.ok, "answer": run.answer, "error": run.error,
		      "preamble": gate.said, "usage": usage_since(usage, calls)})
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
		provider = preamble_provider()
		preamble = lambda history, prompt: ask_preamble(provider, model, history, prompt)  # noqa: E731
	except Exception as exc:  # noqa: BLE001 - then there is no preamble, and the turn will say why
		sys.stderr.write(f"  [no preamble: {type(exc).__name__}: {exc}]\n")
		preamble = None
	emotion_workers = []
	try:
		serve(sys.stdin, runtime, preamble, emotion_workers)
	finally:
		for worker in emotion_workers:
			worker.close()
		runtime.teardown()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
