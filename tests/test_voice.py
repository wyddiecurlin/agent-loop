# AI_OWNED
"""Voice front end, the parts that run anywhere: turn-taking logic and the bridge protocol.

	AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_voice

The audio and network halves (voice/vad.py, voice/speech.py, voice/client.py) need a sound
card and the gateway; docs/VOICE.md says how they were exercised by hand.
"""

import io
import json
import sys

from agent_loop.loop import AgentRun, log

from voice import bridge
from voice.turns import (
	ACKS, REASSURE, TOOL_LINE, BargeIn, Endpointer, Narrator, frames, heard_text, spoken_tool,
	split_sentences, usable_transcript,
)


def check(label: str, passed, detail: str = "") -> bool:
	passed = bool(passed)
	print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
	return passed


def feed(ep: Endpointer, probs) -> list[tuple[int, str]]:
	return [(i, e) for i, p in enumerate(probs) if (e := ep.feed(p))]


def test_endpointer() -> bool:
	ok = True
	ev = feed(Endpointer(), [0.05] * 20 + [0.9] * 15 + [0.1] * 40)
	ok &= check("a turn opens on the third speech frame", ev[0] == (22, "start"), str(ev))
	ok &= check("speculates after 250 ms of silence", (35 + frames(250) - 1, "speculate") in ev, str(ev))
	ok &= check("ends after 600 ms of silence", (35 + frames(600) - 1, "end") in ev, str(ev))
	ok &= check("one event per phase", [e for _, e in ev] == ["start", "speculate", "end"], str(ev))

	probs = [0.9] * 10 + [0.1] * frames(250) + [0.9] * 10 + [0.1] * frames(600)
	kinds = [e for _, e in feed(Endpointer(), probs)]
	ok &= check("speech after a speculation speculates again", kinds == ["start", "speculate", "speculate", "end"], str(kinds))

	ep, snap, final = Endpointer(), None, None
	for p in probs:
		e = ep.feed(p)
		if e == "speculate" and snap is None:
			snap = ep.speech_frames
		if e == "end":
			final = ep.speech_frames
	ok &= check("speech_frames moves when speech resumed, so the early result is known stale",
	            snap is not None and final is not None and final > snap, f"{snap} -> {final}")
	probs = [0.9] * 10 + [0.1] * frames(600)
	ep, snap, final = Endpointer(), None, None
	for p in probs:
		e = ep.feed(p)
		if e == "speculate":
			snap = ep.speech_frames
		if e == "end":
			final = ep.speech_frames
	ok &= check("and stays put when nothing more was said", snap == final, f"{snap} -> {final}")

	ev = feed(Endpointer(), [0.9] * 4 + [0.05] * frames(600))
	ok &= check("a 130 ms blip is aborted, not a turn", [e for _, e in ev] == ["start", "abort"], str(ev))
	ev = feed(Endpointer(), [0.9, 0.9, 0.1, 0.9, 0.9, 0.1] * 5)
	ok &= check("two speech frames in a row never open a turn", ev == [], str(ev))
	ev = feed(Endpointer(), [0.9] * 10 + [0.4] * 30)
	ok &= check("hesitant frames between the thresholds keep the turn open", ev == [(2, "start")], str(ev))
	return ok


def test_bargein() -> bool:
	b = BargeIn()
	ok = check("silence while playing never barges", not any(b.feed(0.1, True) for _ in range(50)))
	hits = [i for i in range(10) if b.feed(0.9, True)]
	ok &= check("sustained speech while playing barges once, at 250 ms", hits == [frames(250) - 1], str(hits))
	ok &= check("speech while we are silent is not a barge-in", not any(BargeIn().feed(0.9, False) for _ in range(20)))
	b = BargeIn()
	ok &= check("a gap resets the run", not any(b.feed(p, True) for p in [0.9] * 5 + [0.2] + [0.9] * 5))
	return ok


def test_narrator() -> bool:
	n = Narrator(seed=1)
	ok = check("quiet before a turn is submitted", n.tick(0.0, False) is None and n.tick(100.0, False) is None)
	n.submitted(10.0)
	ok &= check("nothing in the first 400 ms", n.tick(10.3, False) is None)
	ack = n.tick(10.5, False)
	ok &= check("acknowledges once the agent has been quiet 400 ms", ack in ACKS, repr(ack))
	ok &= check("never talks over itself", n.tick(11.0, True) is None)
	n.tool("fs_read", 11.0)
	ok &= check("a tool line waits for the 2.5 s gap", n.tick(12.0, False) is None)
	line = n.tick(13.1, False)
	ok &= check("then announces the tool in words", line is not None and "file read tool" in line, repr(line))
	n.tool("fs_read", 14.0)
	ok &= check("the same tool again is not worth repeating", n.tick(20.0, False) is None)
	n.tool("shell_run", 20.0)
	line = n.tick(20.0, False)
	ok &= check("a different tool is announced", line is not None and "shell run tool" in line, repr(line))
	ok &= check("reassures after 12 s of silence", n.tick(32.5, False) in REASSURE)
	ok &= check("but not before the next 12 s", n.tick(40.0, False) is None)
	n.tick(44.6, False)
	n.tick(56.7, False)
	ok &= check("and at most three times", n.tick(70.0, False) is None)
	n.answered()
	ok &= check("silent once the answer is in", n.tick(90.0, False) is None)
	n = Narrator(seed=1)
	n.submitted(0.0)
	n.tool("web_search", 0.2)
	line = n.tick(0.25, False)
	ok &= check("an early tool call is announced instead of an um", line is not None and "web search tool" in line, repr(line))
	lines = set()
	for _ in range(20):
		n = Narrator(seed=None)
		n.submitted(0.0)
		lines.add(n.tick(1.0, False))
	ok &= check("acknowledgements vary", len(lines) > 1, str(lines))
	return ok


def test_heard() -> bool:
	s = [("One.", 0, 100), ("Two.", 100, 300), ("Three.", 300, None)]
	ok = check("nothing played, nothing heard", heard_text(s, 0) == "")
	ok &= check("a finished sentence counts", heard_text(s, 100) == "One.")
	ok &= check("more than half of the next counts too", heard_text(s, 210) == "One. Two.")
	ok &= check("less than half does not", heard_text(s, 140) == "One.")
	ok &= check("an unfinished sentence counts after half a second", heard_text(s, 300 + 24_000) == "One. Two. Three.")
	ok &= check("but not before", heard_text(s, 300 + 1000) == "One. Two.")
	return ok


def test_transcripts() -> bool:
	ok = check("keeps real speech", usable_transcript(" List the files, please. ") == "List the files, please.")
	ok &= check("drops Whisper's silence favourites", usable_transcript("Thank you.") is None and usable_transcript("you") is None)
	ok &= check("drops sound effects", usable_transcript("[BLANK_AUDIO]") is None and usable_transcript("(silence)") is None)
	ok &= check("strips a sound effect off real speech", usable_transcript("[music] run the tests") == "run the tests")
	ok &= check("tool names become words", spoken_tool("fs_read") == "file read"
	            and spoken_tool("get_today_date") == "get today's date" and spoken_tool("substract") == "subtract")
	ok &= check("the trace's tool line is recognised, other trace lines are not",
	            TOOL_LINE.match("  [fs_read]") and not TOOL_LINE.match("[LOG] step 1/120")
	            and not TOOL_LINE.match("tool calling: fs_read"))
	return ok


def test_sentences() -> bool:
	ok = check("splits on sentence ends", split_sentences("All done here. I wrote the file! Anything else?") == ["All done here.", "I wrote the file!", "Anything else?"])
	ok &= check("keeps decimals together", split_sentences("It took 3.5 seconds.") == ["It took 3.5 seconds."])
	ok &= check("newlines break too", split_sentences("First line\nsecond line") == ["First line", "second line"])
	ok &= check("tiny fragments ride with the next sentence",
	            split_sentences("No. It failed because the test was wrong.") == ["No. It failed because the test was wrong."])
	ok &= check("empty in, nothing out", split_sentences("  ") == [])
	return ok


def test_bridge() -> bool:
	calls = []

	def fake_loop(prompt, runtime, history=None, system_prompt="", verbose=True, **_):
		calls.append({"prompt": prompt, "history": history, "system_prompt": system_prompt, "verbose": verbose})
		log("  [fs_read]")
		log("[LOG] step 1/120 (3 items in context)")
		log("  [done]")
		if "boom" in prompt:
			raise RuntimeError("provider down")
		msgs = list(history or []) + [
			{"role": "user", "content": prompt},
			{"type": "function_call", "call_id": "c1", "name": "done", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "c1", "output": "forty two"},
			{"role": "assistant", "content": "forty two"},
		]
		return AgentRun(msgs, "done", 2)

	saved = bridge.agent_loop, sys.stdout, sys.stderr
	out, err = io.StringIO(), io.StringIO()
	bridge.agent_loop, sys.stdout, sys.stderr = fake_loop, out, bridge.ToolTap(err)
	try:
		bridge.serve(io.StringIO(
			'{"type": "prompt", "text": "what is six times seven"}\n'
			'not json\n'
			'{"type": "heard", "text": "forty"}\n'
			'{"type": "prompt", "text": "say it again"}\n'
			'{"type": "prompt", "text": "boom"}\n'
			'{"type": "prompt", "text": "   "}\n'), runtime=None)
	finally:
		bridge.agent_loop, sys.stdout, sys.stderr = saved

	events = [json.loads(line) for line in out.getvalue().splitlines()]
	kinds = [e["type"] for e in events]
	ok = check("a tool event per tool line, none for done, an answer per prompt, an error for junk",
	           kinds == ["tool", "answer", "error", "tool", "answer", "tool", "answer"], str(kinds))
	ok &= check("tool events name the tool", events[0] == {"type": "tool", "name": "fs_read"}, str(events[0]))
	ok &= check("the answer is what done returned", events[1]["text"] == "forty two" and events[1]["ok"]
	            and events[1]["stop_reason"] == "done" and events[1]["steps"] == 2, str(events[1]))
	ok &= check("the trace still reaches stderr", "[fs_read]" in err.getvalue() and "[LOG] step" in err.getvalue())
	ok &= check("voice rules ride on the system prompt, and the loop is quiet",
	            "Voice mode" in calls[0]["system_prompt"] and calls[0]["verbose"] is False)
	ok &= check("the second turn carries the first turn's history",
	            calls[1]["history"] is not None and calls[1]["history"][-1]["content"] == "forty two")
	ok &= check("an interruption is told to the model with what was heard",
	            calls[1]["prompt"].startswith("(You were interrupted") and '"forty"' in calls[1]["prompt"]
	            and calls[1]["prompt"].endswith("say it again"), calls[1]["prompt"])
	ok &= check("and only once", not calls[2]["prompt"].startswith("("))
	ok &= check("a crash is an answer with ok false", events[-1]["ok"] is False
	            and "provider down" in events[-1]["text"] and events[-1]["stop_reason"] == "error", str(events[-1]))
	ok &= check("a blank prompt is ignored", len(calls) == 3)
	return ok


def main() -> int:
	results = [test_endpointer(), test_bargein(), test_narrator(), test_heard(), test_transcripts(), test_sentences(), test_bridge()]
	print("\nvoice:", "all passed" if all(results) else "FAILURES")
	return 0 if all(results) else 1


if __name__ == "__main__":
	raise SystemExit(main())
