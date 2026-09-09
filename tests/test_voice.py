# AI_OWNED
"""Voice front end, the parts that run anywhere: turn-taking logic and the bridge protocol.

	AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_voice

The host audio and network code has a separate mocked suite in test_voice_audio.py.
"""

import io
import json
import re
import sys

from agent_loop.loop import AgentRun, log

from voice import bridge
from voice.echo import EchoConfig, EchoGate
from voice.turns import (
	MAX_UTTERANCE_S, STATUS_LINES, TOO_LONG, TOOL_LINE, BargeIn, EndpointConfig, Endpointer, Narrator,
	NarratorConfig, all_status_lines, frames, heard_text, speech_seconds, speech_text, speech_utterances,
	split_sentences, spoken_answer, status_language, status_lines, tool_family, usable_transcript,
)

EN, ZH = STATUS_LINES["en"], STATUS_LINES["zh"]


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

	# Someone can talk for a minute without a 600 ms gap in it. The buffer, the upload and
	# the transcription all have to stop somewhere, so the turn does.
	ev = feed(Endpointer(), [0.9] * (frames(60_000) + 200))
	ok &= check("a turn nobody ends is taken as said at sixty seconds",
	            ev[:2] == [(2, "start"), (frames(60_000) - 1, "end")], str(ev[:2]))
	ok &= check("and the speaker carries straight on into the next one", ev[2] == (1877, "start"), str(ev))
	ev = feed(Endpointer(EndpointConfig(max_turn_ms=1000)), [0.9] * 10 + [0.1] * 5 + [0.9] * 40)
	ok &= check("the pauses inside a turn count towards its cap too",
	            ev[:2] == [(2, "start"), (frames(1000) - 1, "end")], str(ev))
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
	n = Narrator(seed=3)
	ok = check("quiet before a turn is submitted", n.tick(0.0, False) is None and n.tick(100.0, False) is None)
	n.submitted(10.0)
	ok &= check("nothing in the first half second", n.tick(10.3, False) is None)
	filler = n.tick(10.5, False)
	ok &= check("then a filler, a sound and not a sentence", filler in EN["fillers"] and filler != "", repr(filler))
	ok &= check("and never 'let me check'", not any("check" in f.lower() for f in EN["fillers"]))
	ok &= check("only one", n.tick(11.0, False) is None and n.tick(13.0, False) is None)
	n.tool("web_search", 11.0)
	n.tool("web_fetch", 12.0)
	ok &= check("a tool call is not announced", n.tick(12.0, False) is None and n.tick(16.0, False) is None)
	line = n.tick(17.6, False)
	ok &= check("seven quiet seconds after the filler: a hold line, about the web", line == EN["hold"]["web"], repr(line))
	ok &= check("not again for fifteen", n.tick(30.0, False) is None)
	n.tool("shell_run", 31.0)
	ok &= check("and the next one follows what is running now", n.tick(32.7, False) == EN["hold"]["shell"])
	n.tick(47.8, False)
	ok &= check("at most three", n.tick(70.0, False) is None)
	n.answered()
	ok &= check("silent once the answer is in", n.tick(90.0, False) is None)

	# The preamble is the agent's own line, spoken live; while anything is heard the clock
	# for a hold line does not run, and a filler is not owed at all.
	n = Narrator(seed=3)
	n.submitted(0.0)
	n.said(0.3)
	ok &= check("a preamble means no filler", n.tick(0.5, True) is None and n.tick(2.0, True) is None)
	ok &= check("quiet is counted from when the speaking stopped", n.tick(8.0, False) is None
	            and n.tick(9.1, False) == EN["hold"][""])
	ok &= check("never talks over itself", Narrator().tick(5.0, True) is None)

	# Silence is one of the fillers, on purpose: not "um" every single time.
	picks = set()
	for seed in range(40):
		n = Narrator(seed=seed)
		n.submitted(0.0)
		picks.add(n.tick(1.0, False))
	ok &= check("fillers vary, and sometimes there is none", len(picks) >= 3 and None in picks, str(picks))
	n = Narrator(NarratorConfig(filler_after_s=0.2, hold_after_s=1.0, hold_every_s=1.0, max_hold=1), seed=1)
	n.submitted(0.0)
	n.tick(0.2, False)
	ok &= check("the timings are configuration", n.tick(1.2, False) == EN["hold"][""] and n.tick(5.0, False) is None)

	n = Narrator(language="zh", seed=1)
	n.submitted(0.0)
	n.tool("fs_read", 0.1)
	n.tick(0.5, False)
	ok &= check("status lines follow the language of the conversation", n.tick(8.0, False) == ZH["hold"]["files"])
	ok &= check("a language with no lines of its own falls back to English",
	            status_lines("fr") is EN and status_lines("zh") is ZH)
	ok &= check("the spoken language is read off the turn, not a stray character",
	            status_language("给我看一下这个文件") == "zh"
	            and status_language("I've been thinking about 健康.") == "en"
	            and status_language("run the tests") == "en")
	ok &= check("every line worth caching is listed once, and the silence is not",
	            "" not in all_status_lines() and len(all_status_lines()) == len(set(all_status_lines()))
	            and EN["hold"]["web"] in all_status_lines() and ZH["hold"][""] in all_status_lines()
	            and ZH["hold"][""] not in all_status_lines(("en",)))
	ok &= check("no line laughs, announces a tool or checks anything",
	            not any(re.search(r"(?i)tool|check|haha", line) for line in all_status_lines()))
	return ok


def test_echo_gate() -> bool:
	"""A synthetic room: the speaker's blocks every 20 ms, the microphone's frames every
	32 ms, the echo arriving 130 ms later at a quarter of the level with the room's noise
	on top, and a person talking over it for one second in the middle."""
	import random

	rng = random.Random(1)
	gate = EchoGate(EchoConfig(lag_s=0.2, gain=0.1))  # wrong on purpose: it has to learn
	true_lag, true_gain = 0.13, 0.25
	played: list[tuple[float, float]] = []

	def level(t: float) -> float:
		return 0.06 * (0.3 + abs(((t * 7) % 2) - 1)) if 1.0 <= t < 9.0 else 0.0

	events = sorted([(i * 0.02, "out") for i in range(600)] + [(i * 0.032, "mic") for i in range(375)])
	missed = leaked = idle_blocked = 0
	for t, kind in events:
		if kind == "out":
			gate.played(t, level(t))
			played.append((t, level(t)))
			continue
		src = max((r for w, r in played if abs(w - (t - true_lag)) <= 0.03), default=0.0)
		rms = src * true_gain * rng.uniform(0.6, 1.6) + rng.uniform(0.002, 0.006)
		talking = 5.0 <= t < 6.0
		if talking:
			rms += 0.12
		echo = gate.heard(t, rms)
		if 2.0 <= t < 9.0:
			if talking and echo:
				missed += 1
			if not talking and not echo:
				leaked += 1
		elif t < 1.0 and echo:
			idle_blocked += 1
	ok = check("nothing of the robot's own voice gets through", leaked == 0, f"{leaked} frames leaked")
	ok &= check("a person talking over it is still heard", missed <= 2, f"{missed} frames missed")
	ok &= check("the room is learned from wrong priors", abs(gate.lag - true_lag) < 0.02 and 0.15 < gate.gain < 0.45
	            and gate.fits >= 2, str(gate.state()))
	ok &= check("with nothing playing the microphone is heard raw", idle_blocked == 0)
	ok &= check("the quiet room's level is learned while idle", 0.002 < gate.floor < 0.007, str(gate.floor))
	ok &= check("the echo is over once the tail has rung out", not gate.active(9.0 + 0.13 + 0.3 + 0.1)
	            and gate.active(9.0 + 0.13 + 0.1))
	return ok


def test_heard() -> bool:
	s = [("One.", 0, 100), ("Two.", 100, 300), ("Three.", 300, None)]
	ok = check("nothing played, nothing heard", heard_text(s, 0) == "")
	ok &= check("a finished sentence counts", heard_text(s, 100) == "One.")
	ok &= check("half a segment does not claim its ending was heard", heard_text(s, 210) == "One.")
	ok &= check("less than half does not", heard_text(s, 140) == "One.")
	ok &= check("unfinished synthesis never claims the whole segment was heard", heard_text(s, 300 + 24_000) == "One. Two.")
	ok &= check("absolute offsets work after previous playback", heard_text([("Next turn.", 500, 700)], 700) == "Next turn.")
	ok &= check("empty audio is not heard", heard_text([("Empty.", 500, 500)], 500) == "")
	return ok


def test_transcripts() -> bool:
	ok = check("keeps real speech", usable_transcript(" List the files, please. ") == "List the files, please.")
	ok &= check("drops Whisper's silence favourites", usable_transcript("Thank you.") is None and usable_transcript("you") is None)
	ok &= check("drops sound effects", usable_transcript("[BLANK_AUDIO]") is None and usable_transcript("(silence)") is None)
	ok &= check("strips a sound effect off real speech", usable_transcript("[music] run the tests") == "run the tests")
	ok &= check("keeps real Chinese speech", usable_transcript(" 帮我看一下这个文件。 ") == "帮我看一下这个文件。")
	ok &= check("drops Whisper's Chinese silence fillers",
	            usable_transcript("谢谢观看。") is None and usable_transcript("字幕由Amara.org社群提供") is None
	            and usable_transcript("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目") is None)
	ok &= check("a phrase repeated to fill the budget is not a turn",
	            usable_transcript("一位开发者,一位开发者,一位开发者,一位开发者,一位开发者,一") is None
	            and usable_transcript("no no no no no no") is None
	            and usable_transcript("一位开发者在和编程中,一位开发者在和编程中,一位开发者,一位开发者,一位开发者,一位开发者,一") is None
	            and usable_transcript("yes, yes, yes, it worked") == "yes, yes, yes, it worked"
	            and usable_transcript("Thank you, thank you, thank you.") is None
	            and usable_transcript("no, run it again") == "no, run it again"
	            and usable_transcript("测试测试，可以听到吗") == "测试测试，可以听到吗")
	primer = ("A developer talking to a coding agent about files, tests, git, Python, and the shell. "
	          "一位开发者在和编程助手讨论文件、测试、git、Python 和终端命令。")
	ok &= check("Whisper reading its own primer back is not a turn",
	            usable_transcript("一位开发者在和编程中,一位开发者在和编程中,一位开发者在一起。", primer) is None
	            and usable_transcript("A developer talking to a coding agent.", primer) is None
	            and usable_transcript("run the tests, git commit, and ping me", primer) == "run the tests, git commit, and ping me"
	            and usable_transcript("帮我看一下测试文件", primer) == "帮我看一下测试文件"
	            and usable_transcript("一位开发者在和编程中", "") == "一位开发者在和编程中")
	ok &= check("tool names fall into families", tool_family("fs_read") == "files" and tool_family("web_fetch") == "web"
	            and tool_family("shell_run") == "shell" and tool_family("done") == "")
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


def test_speech_text() -> bool:
	ok = check("strips markdown and nonverbal directions",
	           speech_text("## Done\n**All tests passed.** [laughs] *chuckles* [Details](https://example.com).")
	           == "Done\nAll tests passed. Details.")
	ok &= check("keeps inline names, paths and numbers",
	            speech_text("Read `voice/client.py` at 3.5 seconds.") == "Read voice/client.py at 3.5 seconds.")
	ok &= check("keeps multiplication and identifiers while stripping emphasis",
	            speech_text("**Result:** 2 * 3 * 4. a*b*c and file_name. _Done._")
	            == "Result: 2 * 3 * 4. a*b*c and file_name. Done.")
	ok &= check("code blocks become a reference to the text answer",
	            speech_text("Here it is.\n```python\nprint('hello')\n```\nIt works.")
	            == "Here it is.\nThe code is in the text answer.\nIt works.")
	ok &= check("drops reasoning and model delimiters",
	            speech_text("<think>private draft</think><|im_start|>All done.<|im_end|>") == "All done.")
	ok &= check("control characters do not reach speech", speech_text("\x1b[31mHello\x1b[0m\x00 &amp; goodbye.") == "Hello & goodbye.")
	# The laughs. Written any way a model writes them, they are performed by the
	# synthesizer, and never again.
	ok &= check("written laughter never reaches the synthesizer",
	            speech_text("Haha, sure. That works, hahaha! lol. Ahahaha. Bahaha no. 哈哈哈哈，行。呵呵。 wwww ok")
	            == "sure. That works, no. 行。 ok", repr(speech_text("Haha, sure. That works, hahaha! lol. Ahahaha. Bahaha no. 哈哈哈哈，行。呵呵。 wwww ok")))
	ok &= check("words that merely contain a laugh survive",
	            speech_text("Shah met Hahn in the Bahamas for a haiku, hehe.") == "Shah met Hahn in the Bahamas for a haiku,")
	ok &= check("emoji are not read out", speech_text("Done 😄🤖 and dusted ✅.") == "Done and dusted .")
	answer = "All tests passed. I updated the file. You can try it now."
	ok &= check("short answers keep their prosodic context", speech_utterances(answer) == [answer])
	long = "This is a longer answer. " * 50
	chunks = speech_utterances(long)
	ok &= check("long answers are bounded without losing words", all(len(c) <= 500 for c in chunks)
	            and " ".join(chunks) == long.strip())
	ok &= check("even unbroken tokens are bounded", all(len(c) <= 500 for c in speech_utterances("x" * 1200)))
	ok &= check("markup alone does not trigger synthesis", speech_utterances("[laughs] <|im_end|>") == [])
	ok &= check("lines survive as places to stop, and never reach the synthesizer",
	            speech_utterances("First line\nsecond line") == ["First line second line"])

	# The bug this replaced: 50 Chinese characters is 13 seconds of speech and was budgeted
	# as if it were 6, so the answer was cut off with "TTS exceeded the audio duration limit".
	zh = "健康是一个广泛的话题。您具体想了解哪方面的健康呢？比如饮食、运动、睡眠、心理健康，还是其他方面？"
	ok &= check("a syllabic script is not measured in characters",
	            speech_seconds(zh) > 10 and speech_seconds("a" * len(zh)) < 5,
	            f"{speech_seconds(zh):.1f}s vs {speech_seconds('a' * len(zh)):.1f}s")
	ok &= check("a Chinese answer stays one utterance if it fits", speech_utterances(zh) == [zh])
	long_zh = zh * 6
	chunks = speech_utterances(long_zh)
	ok &= check("longer ones are split without losing a character", "".join(chunks) == long_zh, str(chunks[:1]))
	ok &= check("every piece fits the budget the stream allows",
	            all(speech_seconds(c) <= MAX_UTTERANCE_S for c in chunks),
	            str([round(speech_seconds(c), 1) for c in chunks]))
	ok &= check("a script without spaces breaks at its own punctuation",
	            all(c[-1] in "。？！，、；：" for c in chunks[:-1]), str(chunks))

	# The gibberish: a list of films flattened to one line has nowhere to break but the
	# middle of a title, and half a title is not something a synthesizer can read.
	films = "\n".join(f"{m} 月 {m + 6} 日：《蜘蛛人：重生日 {m}》（汤姆·霍兰德主演）" for m in range(1, 13))
	chunks = speech_utterances(films)
	lines = films.split("\n")
	ok &= check("a listed answer is broken between its lines, never inside one",
	            all(any(c.startswith(l[:6]) for l in lines) for c in chunks), str(chunks[:2]))
	ok &= check("and no line is lost or duplicated on the way",
	            " ".join(chunks) == " ".join(lines), str(chunks[:1]))
	ok &= check("a sentence is preferred to the clause break inside a title",
	            all("：《" not in c[-3:] for c in chunks), str(chunks))

	# An answer can be worth reading and not worth listening to: twelve items in, the first
	# is gone and there is no scrolling back through speech.
	ok &= check("a short answer is read out as it is", spoken_answer(answer) == answer)
	ok &= check("a long one is left on screen instead", spoken_answer(films, "zh") == TOO_LONG["zh"]
	            and spoken_answer(films) == TOO_LONG["en"])
	ok &= check("where the line is drawn is a number, not a mood",
	            spoken_answer(answer, "en", 0.1) == TOO_LONG["en"]
	            and spoken_answer(films, "en", 1000) == films)
	ok &= check("what stands in for it is short enough to be worth hearing",
	            all(speech_seconds(line) < 15 for line in TOO_LONG.values()),
	            str({k: round(speech_seconds(v), 1) for k, v in TOO_LONG.items()}))

	# Numbers are words: "97.6%" is four syllables and a 百分之, not five characters.
	ok &= check("digits and symbols are counted as spoken, not as typed",
	            speech_seconds("97.6%") > speech_seconds("abcde") * 3,
	            f"{speech_seconds('97.6%'):.1f}s vs {speech_seconds('abcde'):.1f}s")
	return ok


def test_timeouts() -> bool:
	"""A stalled model stream must not leave someone listening to silence for ten minutes."""
	import os

	from agent_loop.providers import DEFAULT_TIMEOUT_S, _env

	ok = check("the eval default is untouched", DEFAULT_TIMEOUT_S == 600.0)
	ok &= check("the voice bridge asks for a conversational one",
	            bridge.VOICE_TIMEOUT_S <= 90 and bridge.VOICE_MAX_RETRIES <= 1)
	saved = {k: os.environ.get(k) for k in ("AGENT_TIMEOUT_S", "AGENT_MAX_RETRIES")}
	try:
		os.environ["AGENT_TIMEOUT_S"] = str(bridge.VOICE_TIMEOUT_S)
		os.environ["AGENT_MAX_RETRIES"] = str(bridge.VOICE_MAX_RETRIES)
		# agent_loop() has no timeout argument and binds the eval default at import, so the
		# environment has to beat the caller, not just the default, or it never lands.
		ok &= check("the environment beats the value agent_loop passes in",
		            _env("AGENT_TIMEOUT_S", DEFAULT_TIMEOUT_S) == bridge.VOICE_TIMEOUT_S
		            and _env("AGENT_MAX_RETRIES", 3, int) == bridge.VOICE_MAX_RETRIES)
		os.environ["AGENT_TIMEOUT_S"] = "one minute please"
		ok &= check("a typo in .env is not a zero timeout",
		            _env("AGENT_TIMEOUT_S", DEFAULT_TIMEOUT_S) == DEFAULT_TIMEOUT_S)
	finally:
		for key, value in saved.items():
			os.environ.pop(key, None) if value is None else os.environ.update({key: value})
	return ok


def test_preamble() -> bool:
	"""The side call: one tool-less request, and only a sentence worth saying comes back."""
	import os
	import time

	class Provider:
		def __init__(self, reply):
			self.reply, self.calls = reply, []

		def generate(self, messages, model, tools, **kw):
			self.calls.append({"messages": messages, "model": model, "tools": tools, **kw})
			return type("Turn", (), {"text": self.reply})()

	history = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
	           {"type": "function_call", "call_id": "c", "name": "done", "arguments": "{}"},
	           {"type": "function_call_output", "call_id": "c", "output": "hey"},
	           {"role": "assistant", "content": "hey"}]
	pv = Provider(" \"Let me see what's out this week.\" ")
	line = bridge.ask_preamble(pv, "m", history, "what's in theaters?")
	ok = check("a sentence comes back bare", line == "Let me see what's out this week.", repr(line))
	call = pv.calls[0]
	ok &= check("no tools, no stream, a short timeout", call["tools"] is None and call["stream"] is False
	            and call["timeout"] <= 30 and call["model"] == "m")
	ok &= check("it sees the character, the clock, the recent talk and the prompt",
	            call["messages"][0]["role"] == "system" and "CHARACTER" in call["messages"][0]["content"]
	            and "Right now it is" in call["messages"][0]["content"]
	            and [m["content"] for m in call["messages"][1:]] == ["hi", "hey", "what's in theaters?"]
	            and not any(m.get("type") for m in call["messages"]), str(call["messages"])[:300])
	ok &= check("the prompt tells it when to say nothing, and never to laugh",
	            "NONE" in bridge.PREAMBLE_PROMPT and "no laughter" in bridge.PREAMBLE_PROMPT)
	ok &= check("NONE, blank and an essay are all silence",
	            bridge.ask_preamble(Provider("NONE"), "m", None, "hi") is None
	            and bridge.ask_preamble(Provider("none."), "m", None, "hi") is None
	            and bridge.ask_preamble(Provider("  "), "m", None, "hi") is None
	            and bridge.ask_preamble(Provider("word " * 80), "m", None, "hi") is None)

	saved = os.environ.get("TZ")
	try:
		bridge.set_clock("Asia/Tokyo")
		tokyo = time.strftime("%Z")
		bridge.set_clock("not a zone")
		ok &= check("the clock takes a named zone and ignores junk", os.environ["TZ"] == "Asia/Tokyo"
		            and tokyo == "JST", tokyo)
	finally:
		os.environ.pop("TZ", None) if saved is None else os.environ.update({"TZ": saved})
		time.tzset()
	return ok


def test_bridge() -> bool:
	import threading

	calls = []
	release = threading.Event()  # the preamble answers only when the test lets it

	def fake_loop(prompt, runtime, history=None, system_prompt="", verbose=True, **_):
		calls.append({"prompt": prompt, "history": history, "system_prompt": system_prompt, "verbose": verbose})
		if "chat" not in prompt:
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

	asked = []

	def fake_preamble(history, prompt):
		asked.append(prompt)
		if "six" in prompt:
			return "Let me work that out."   # said while the turn runs
		if "again" in prompt:
			release.wait(2)                   # arrives after the answer: too late to say
			return "Hang on."
		if "chat" in prompt:
			return "Nothing much, you?"        # a small model answering instead of NONE
		return None                           # small talk: nothing to say

	def slow_loop(prompt, runtime, **kw):
		if "six" in prompt:
			import time
			time.sleep(0.1)  # the preamble lands first
		return fake_loop(prompt, runtime, **kw)

	saved = bridge.agent_loop, sys.stdout, sys.stderr
	out, err = io.StringIO(), io.StringIO()
	bridge.agent_loop, sys.stdout, sys.stderr = slow_loop, out, bridge.ToolTap(err)
	try:
		bridge.serve(io.StringIO(
			'{"type": "clock", "tz": "UTC"}\n'
			'{"type": "prompt", "text": "what is six times seven"}\n'
			'not json\n'
			'{"type": "heard", "text": "forty"}\n'
			'{"type": "prompt", "text": "say it again"}\n'
			'{"type": "prompt", "text": "boom"}\n'
			'{"type": "prompt", "text": "   "}\n'
			'{"type": "prompt", "text": "just chat"}\n'), runtime=None, preamble=fake_preamble)
	finally:
		release.set()
		bridge.agent_loop, sys.stdout, sys.stderr = saved

	events = [json.loads(line) for line in out.getvalue().splitlines()]
	kinds = [e["type"] for e in events]
	ok = check("a preamble when there is one, a tool event per tool line, none for done, a trace "
	           "and an answer per prompt, an error for junk, and a late preamble is never emitted",
	           kinds == ["preamble", "tool", "trace", "answer", "error", "tool", "trace", "answer",
	                     "tool", "trace", "answer", "trace", "answer"], str(kinds))
	ok &= check("the preamble is asked for every turn", len(asked) == 4, str(asked))
	ok &= check("a turn that never calls a tool never gets one, whatever the side call said",
	            events[-2]["preamble"] is None and events[-2]["turn"] == 4, str(events[-2])[:200])
	now = [100.0]
	gate = bridge.Preamble(fresh_s=6.0, clock=lambda: now[0])
	gate.tool()
	now[0] = 130.0
	gate.offer("Let me look.")
	ok &= check("a preamble that took thirty seconds to arrive is stale, not spoken", gate.said is None)
	gate = bridge.Preamble(fresh_s=6.0, clock=lambda: now[0])
	now[0] = 131.0
	gate.offer("Let me look.")
	gate.tool()
	ok &= check("a fresh one waits for the tool and is then said once", gate.said == "Let me look.")
	ok &= check("tool events name the tool", events[1] == {"type": "tool", "name": "fs_read"}, str(events[1]))
	ok &= check("the preamble is the model's line", events[0] == {"type": "preamble", "text": "Let me work that out."})
	answers = [e for e in events if e["type"] == "answer"]
	traces = [e for e in events if e["type"] == "trace"]
	ok &= check("the answer is what done returned", answers[0]["text"] == "forty two" and answers[0]["ok"]
	            and answers[0]["stop_reason"] == "done" and answers[0]["steps"] == 2, str(answers[0]))
	ok &= check("each turn is traced before its answer, with what the loop added to the history",
	            [t["turn"] for t in traces] == [1, 2, 3, 4] and traces[0]["prompt"].endswith("six times seven")
	            and len(traces[0]["messages"]) == 5 and traces[0]["ok"], str(traces[0])[:200])
	ok &= check("a crashed turn is traced too, with no messages and the error on it",
	            traces[2]["messages"] == [] and traces[2]["ok"] is False
	            and "provider down" in traces[2]["error"], str(traces[2])[:200])
	ok &= check("the trace still reaches stderr", "[fs_read]" in err.getvalue() and "[LOG] step" in err.getvalue())
	ok &= check("voice rules ride on the system prompt, and the loop is quiet",
	            "Voice mode" in calls[0]["system_prompt"] and calls[0]["verbose"] is False)
	ok &= check("the second turn carries the first turn's history",
	            calls[1]["history"] is not None and calls[1]["history"][-1]["content"] == "forty two")
	spoken = [m for m in calls[1]["history"] if m.get("role") == "assistant"]
	ok &= check("what the preamble said is in that history, right after the user's words",
	            spoken[0]["content"] == "Let me work that out."
	            and calls[1]["history"][calls[1]["history"].index(spoken[0]) - 1]["role"] == "user"
	            and traces[0]["preamble"] == "Let me work that out.", str(calls[1]["history"])[:300])
	ok &= check("and a preamble that was never said is not", traces[1]["preamble"] is None
	            and not any(m.get("content") == "Hang on." for m in calls[2]["history"]))
	ok &= check("an interruption is told to the model with what was heard",
	            calls[1]["prompt"].startswith("(You were interrupted") and '"forty"' in calls[1]["prompt"]
	            and calls[1]["prompt"].endswith("say it again"), calls[1]["prompt"])
	ok &= check("and only once", not calls[2]["prompt"].startswith("("))
	ok &= check("a crash is an answer with ok false", answers[2]["ok"] is False
	            and "provider down" in answers[2]["text"] and answers[2]["stop_reason"] == "error",
	            str(answers[2]))
	ok &= check("a blank prompt is ignored", len(calls) == 4)
	return ok


def test_runlog() -> bool:
	"""One run directory per launch, never committed, with the turn files and audio in it."""
	import tempfile
	from pathlib import Path

	from voice.log import RunLog

	with tempfile.TemporaryDirectory() as tmp:
		root = Path(tmp) / "logs" / "voice"
		run = RunLog(root)
		run.line("tty", "\x1b[2mstarting\x1b[0m")
		run.event("prompt", turn=1, text="what is six times seven")
		path = run.turn({"type": "trace", "turn": 1, "prompt": "what is six times seven", "ok": True,
		                 "steps": 2, "stop_reason": "done", "elapsed_s": 0.5, "answer": "forty two",
		                 "messages": [{"role": "user", "content": "what is six times seven"}],
		                 "usage": {"calls": 1}})
		uid = run.utterance()
		wav = run.audio(uid, b"\x00\x01" * 160, 16000)
		try:
			raise RuntimeError("provider down")
		except RuntimeError as exc:
			run.exception("tts", exc)
		run.close("test")
		second = RunLog(root, audio=False)
		second.close()

		text = (run.dir / "session.log").read_text()
		records = [json.loads(l) for l in (run.dir / "events.jsonl").read_text().splitlines()]
		ok = check("the directory ignores itself", (root / ".gitignore").read_text().strip() == "*")
		ok &= check("terminal lines land without their colour codes", "starting" in text and "\x1b" not in text)
		ok &= check("events are records with a clock", [r["kind"] for r in records]
		            == ["tty", "prompt", "turn", "error", "exit"] and all("t" in r and "ts" in r for r in records),
		            str([r["kind"] for r in records]))
		ok &= check("a turn is its own pretty file, named in the log", path == run.dir / "turns" / "001.json"
		            and json.loads(path.read_text())["answer"] == "forty two" and "turns/001.json" in text)
		ok &= check("audio is kept as WAV under the utterance id", uid == "u001" and wav == "audio/u001.wav"
		            and (run.dir / wav).read_bytes().startswith(b"RIFF"))
		ok &= check("a traceback reaches the log", "provider down" in text and "Traceback" in text
		            and "Traceback" in records[3]["traceback"])
		ok &= check("closing records why", records[-1]["kind"] == "exit" and records[-1]["reason"] == "test")
		ok &= check("latest points at the newest run", (root / "latest").resolve() == second.dir.resolve()
		            and not (second.dir / "audio").exists())
	return ok


def main() -> int:
	results = [test_endpointer(), test_bargein(), test_narrator(), test_echo_gate(), test_heard(), test_transcripts(),
	           test_sentences(), test_speech_text(), test_timeouts(), test_preamble(), test_bridge(),
	           test_runlog()]
	print("\nvoice:", "all passed" if all(results) else "FAILURES")
	return 0 if all(results) else 1


if __name__ == "__main__":
	raise SystemExit(main())
