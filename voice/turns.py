# AI_OWNED
"""Turn-taking decisions with no audio or network in them, so they run and test anywhere.

Three small state machines, fed once per 32 ms frame (Silero's frame at 16 kHz):

	Endpointer   one VAD probability in -> "start" | "speculate" | "end" | "abort" | None
	BargeIn      the same probability plus "are we playing" -> True when the user talks over us
	Narrator     the clock in -> a short spoken line while the agent works, or None

plus the helpers that turn a Whisper result into a usable turn, a tool name into words,
and a cut-off answer into what the user actually heard.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

FRAME_MS = 32  # 512 samples at 16 kHz, the frame Silero VAD v5 wants

# The quiet trace prints every tool call as a line of exactly this shape (loop.py,
# verbose=False). bridge.py turns it into an event; client.py hides the duplicate.
TOOL_LINE = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*$")

# Chinese, Japanese and Korean are written one syllable per character and without spaces,
# so a character count says something completely different there than it does in a Latin
# script: 50 characters is three seconds of English and thirteen of Mandarin. Everything
# below that budgets or splits text asks this, never len().
DENSE_SCRIPT = re.compile(
	r"[\u1100-\u11ff\u2e80-\u303f\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff"
	r"\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff"
	r"\ufe30-\ufe4f\uff00-\uff60\uffe0-\uffe6\U00020000-\U0003ffff]")
DENSE_SECONDS = 0.25      # a syllabic character, read slowly enough to be a safe bound
SPARSE_SECONDS = 1 / 13   # a Latin character
DIGIT_SECONDS = 0.25      # every digit is spoken separately: "97.6" is four syllables
# Symbols nobody pronounces as one character. "%" is 百分之 or "percent", "$" is 美元 or
# "dollars": five characters of "97.6%" are two and a half seconds of speech, not four
# tenths, and a table of benchmark numbers is mostly made of them.
SPOKEN_SYMBOLS = {"%": 0.75, "$": 0.5, "€": 0.5, "£": 0.5, "¥": 0.4, "=": 0.5, "#": 0.5,
                  "/": 0.35, "&": 0.35, "+": 0.35, "@": 0.35, "~": 0.35, "×": 0.35}


def speech_seconds(text: str) -> float:
	"""Roughly how long `text` takes to say aloud.

	Only ever used to size a budget, so it errs high and needs no linguistics beyond which
	characters are whole syllables and which are words in disguise.
	"""
	total = 0.0
	for c in text:
		if c.isdigit():
			total += DIGIT_SECONDS
		elif c in SPOKEN_SYMBOLS:
			total += SPOKEN_SYMBOLS[c]
		elif DENSE_SCRIPT.match(c):
			total += DENSE_SECONDS
		else:
			total += SPARSE_SECONDS
	return total


def dense_script(text: str) -> bool:
	"""Is this mostly written in a syllabic script? A stray character is not enough."""
	body = "".join(text.split())
	return bool(body) and len(DENSE_SCRIPT.findall(body)) * 3 >= len(body)


def frames(ms: int) -> int:
	"""How many frames cover `ms` milliseconds: rounded up, never zero."""
	return max(1, -(-ms // FRAME_MS))


# --- where a turn starts and ends -------------------------------------------------------

@dataclass
class EndpointConfig:
	start_prob: float = 0.5    # a frame this likely to be speech opens a turn ...
	start_frames: int = 3      # ... once three in a row agree (~100 ms), so a click does not
	end_prob: float = 0.35     # below this a frame is silence: hysteresis against flutter
	speculate_ms: int = 250    # silence at which the transcription is fired early
	end_ms: int = 600          # silence that closes the turn
	min_speech_ms: int = 250   # anything shorter is a cough or a chair
	max_turn_ms: int = 60_000  # the longest turn we will hold before taking what we have


class Endpointer:
	"""Where a turn starts and ends, from one probability per frame.

	`speech_frames` counts the frames judged speech since the turn opened. A caller that
	fired a speculative transcription remembers the count at that moment; if it is the
	same when "end" arrives, nothing was said in between and the early result stands.

	`turn_frames` counts every frame since it opened, silence included, and closes the turn
	at `max_turn_ms` however the speaker feels about it. Without that the buffer, the upload
	and the transcription are all bounded only by how long someone is willing to keep going.
	"""

	def __init__(self, cfg: EndpointConfig | None = None):
		self.cfg = cfg or EndpointConfig()
		self.in_speech = False
		self.speech_run = 0
		self.silence_run = 0
		self.speech_frames = 0
		self.turn_frames = 0
		self.speculated = False

	def _close(self) -> str:
		self.in_speech, self.speech_run, self.silence_run = False, 0, 0
		return "end" if self.speech_frames >= frames(self.cfg.min_speech_ms) else "abort"

	def feed(self, p: float) -> str | None:
		c = self.cfg
		if not self.in_speech:
			if p < c.start_prob:
				self.speech_run = 0
				return None
			self.speech_run += 1
			if self.speech_run < c.start_frames:
				return None
			self.in_speech, self.speech_frames = True, self.speech_run
			self.turn_frames, self.silence_run, self.speculated = self.speech_run, 0, False
			return "start"
		self.turn_frames += 1
		expired = self.turn_frames >= frames(c.max_turn_ms)
		if p >= c.end_prob:
			self.speech_frames += 1
			self.silence_run = 0
			self.speculated = False  # speech resumed: any early transcription is stale
			return self._close() if expired else None
		self.silence_run += 1
		if self.silence_run >= frames(c.end_ms) or expired:
			return self._close()
		if (not self.speculated and self.silence_run >= frames(c.speculate_ms)
		    and self.speech_frames >= frames(c.min_speech_ms)):  # a blip is not worth a request
			self.speculated = True
			return "speculate"
		return None


# --- the user talking over us ------------------------------------------------------------

@dataclass
class BargeInConfig:
	prob: float = 0.6          # stricter than start_prob: a little of our own voice leaks back in
	sustain_ms: int = 250      # this long before we stop talking; a cough does not cut us off


class BargeIn:
	def __init__(self, cfg: BargeInConfig | None = None):
		self.cfg = cfg or BargeInConfig()
		self.run = 0

	def feed(self, p: float, playing: bool) -> bool:
		if not playing or p < self.cfg.prob:
			self.run = 0
			return False
		self.run += 1
		if self.run < frames(self.cfg.sustain_ms):
			return False
		self.run = 0
		return True


# --- what to say while the agent works ---------------------------------------------------

# The short lines spoken while the agent works. Only English and Chinese are written out;
# any other spoken language falls back to the English ones.
STATUS_LINES = {
	"en": {
		"ack": "Let me check.",
		"reassure": "I'm still working on that.",
		"fs_list": "I'm checking the files.",
		"fs_read": "I'm reading the file.",
		"fs_search": "I'm searching the files.",
		"fs_write": "I'm writing the file.",
		"fs_patch": "I'm updating the file.",
		"shell_run": "I'm running the command.",
		"web_search": "I'm searching the web.",
		"web_fetch": "I'm reading the page.",
		"multiply": "I'm checking the calculation.",
		"substract": "I'm checking the calculation.",
		"get_today_date": "I'm checking the date.",
	},
	"zh": {
		"ack": "我看一下。",
		"reassure": "还在处理，请稍等。",
		"fs_list": "我在看有哪些文件。",
		"fs_read": "我在读这个文件。",
		"fs_search": "我在搜索文件。",
		"fs_write": "我在写文件。",
		"fs_patch": "我在修改文件。",
		"shell_run": "我在执行命令。",
		"web_search": "我在搜索网页。",
		"web_fetch": "我在读这个网页。",
		"multiply": "我在核对计算。",
		"substract": "我在核对计算。",
		"get_today_date": "我在查日期。",
	},
}
STATUS_LANGUAGES = tuple(STATUS_LINES)


def status_line(key: str, language: str = "en") -> str:
	"""The line for `key` -- a tool name, "ack" or "reassure" -- in `language`.

	An unknown tool is worth acknowledging but not describing, so it falls back to the
	acknowledgement rather than inventing a sentence about machinery the user cannot see.
	"""
	lines = STATUS_LINES.get(language) or STATUS_LINES["en"]
	return lines.get(key) or lines["ack"]


def status_language(text: str) -> str:
	"""Which set of status lines suits a turn the user just spoke."""
	return "zh" if dense_script(text) else "en"


# Not a status line but the same idea: what to say in place of an answer nobody wants read
# to them. Kept out of STATUS_LINES because it is spoken through the answer stream, not
# prefetched as a status clip.
TOO_LONG = {
	"en": "That answer is a bit long to read out loud. It is on screen in full. "
	      "Have a look, and ask me about any of it.",
	"zh": "这个回答有点长，我就不念了。完整内容在屏幕上，你看一下，有什么问题随时问我。",
}
MAX_SPOKEN_S = 40.0


def spoken_answer(text: str, language: str = "en", max_seconds: float = MAX_SPOKEN_S) -> str:
	"""What to say for an answer that the terminal is showing in full.

	A twelve-item list is a fine answer to read and a poor one to listen to: by the third
	entry the first is gone, and there is no scrolling back through speech. Past
	`max_seconds` of it, say where the answer is instead of reading it out.
	"""
	return text if speech_seconds(speech_text(text)) <= max_seconds else (
		TOO_LONG.get(language) or TOO_LONG["en"])


SPOKEN_WORDS = {"fs": "file", "substract": "subtract", "today": "today's"}


def spoken_tool(name: str) -> str:
	"""`fs_read` -> "file read", `get_today_date` -> "get today's date"."""
	return " ".join(SPOKEN_WORDS.get(w, w) for w in name.split("_"))


@dataclass
class NarratorConfig:
	ack_after_s: float = 1.2        # quick answers need no acknowledgement or tool line
	gap_s: float = 6.0              # least time between two spoken lines
	repeat_tool_after_s: float = 15.0
	reassure_every_s: float = 15.0
	max_reassure: int = 2


class Narrator:
	"""Picks the short lines spoken while the agent is busy. Time is passed in, never read."""

	def __init__(self, cfg: NarratorConfig | None = None, language: str = "en"):
		self.cfg = cfg or NarratorConfig()
		self.language = language  # reassigned per turn when the spoken language is not fixed
		self.active = False
		self.t0 = 0.0
		self.last_spoken: float | None = None
		self.pending: str | None = None
		self.last_tool: str | None = None
		self.last_tool_at = float("-inf")
		self.reassured = 0

	def submitted(self, now: float) -> None:
		"""The user's turn went to the agent; from here on we owe them signs of life."""
		self.active, self.t0, self.last_spoken = True, now, None
		self.pending, self.last_tool, self.last_tool_at, self.reassured = None, None, float("-inf"), 0

	def answered(self) -> None:
		self.active, self.pending = False, None

	def tool(self, name: str, now: float) -> None:
		if not self.active:
			return
		if name == self.last_tool and now - self.last_tool_at < self.cfg.repeat_tool_after_s:
			return
		self.pending = name

	def tick(self, now: float, speaking: bool) -> str | None:
		if not self.active or speaking:
			return None
		c = self.cfg
		quiet_for = now - (self.t0 if self.last_spoken is None else self.last_spoken)
		if self.last_spoken is None and quiet_for < c.ack_after_s:
			return None
		if self.pending is not None and (self.last_spoken is None or quiet_for >= c.gap_s):
			name, self.pending = self.pending, None
			self.last_tool, self.last_tool_at, self.last_spoken = name, now, now
			return status_line(name, self.language)
		if self.last_spoken is None:
			if quiet_for < c.ack_after_s:
				return None
			self.last_spoken = now
			return status_line("ack", self.language)
		if quiet_for >= c.reassure_every_s and self.reassured < c.max_reassure:
			self.reassured += 1
			self.last_spoken = now
			return status_line("reassure", self.language)
		return None


# --- from an answer to utterances ----------------------------------------------------------

SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+(?=\S)|\n+")


def split_sentences(text: str) -> list[str]:
	"""The answer as sentences, to synthesize one at a time: the TTS server treats one input
	as one utterance, and a cut-off is only known to the sentence if the sentences were
	separate utterances. Fragments under eight characters ("No.") ride with the next one."""
	parts = [p.strip() for p in SENTENCE_BREAK.split(text or "") if p and p.strip()]
	merged: list[str] = []
	for part in parts:
		if merged and len(merged[-1]) < 8:
			merged[-1] = f"{merged[-1]} {part}"
		else:
			merged.append(part)
	return merged


def speech_text(text: str) -> str:
	"""Plain text for synthesis; retain the original answer separately for the terminal.

	Remove presentation markup, model delimiters and nonverbal stage directions. Keep
	names, numbers, inline code and paths: these may be what the user asked to hear, and
	keep the line breaks: a list of twelve films is twelve natural places to stop, and
	flattening it to one line is what leaves nowhere to break but the middle of a title.
	"""
	text = html.unescape(text or "")
	text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
	text = re.sub(r"<think\b[^>]*>.*?(?:</think>|$)", "", text, flags=re.S | re.I)
	text = re.sub(r"<\|[^<>]*\|>", "", text)
	text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?(?:^\s*\1\s*$|\Z)",
	              " The code is in the text answer. ", text)
	text = re.sub(r"!?\[([^\]]+)\]\(\S+?(?:\s+\"[^\"]*\")?\)", r"\1", text)
	text = re.sub(r"<https?://([^>]+)>", r"https://\1", text)
	text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
	text = re.sub(r"(?im)^[ \t]*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)", "", text)
	text = re.sub(r"(?i)(?:\[|\(|\*{1,2})\s*(?:laughs?|laughing|chuckles?|chuckling|"
	              r"giggles?|giggling|sighs?|sighing|gasps?|gasping)\s*(?:\]|\)|\*{1,2})", "", text)
	text = re.sub(r"`+([^`]+)`+", r"\1", text)
	text = re.sub(r"(?<!\w)(\*{1,3}|_{1,2}|~~)(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
	text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c.isspace())
	lines = (" ".join(line.split()) for line in text.splitlines())
	return "\n".join(line for line in lines if line)


# Where an answer may be broken, best first. A sentence or a line is a place to stop for
# breath; a clause is a compromise; between two characters is what is left when a script
# without spaces hands us a hundred characters with no punctuation in them at all.
SENTENCE_END = "。．.！!？?…\n"
CLAUSE_END = "，,、；;：:》）」』】)]"
MAX_UTTERANCE_S = 20.0
STRONG, WEAK, ANYWHERE = 2, 1, 0


def speech_segments(text: str, max_chars: int) -> list[tuple[str, int]]:
	"""`text` as pieces that concatenate back to it, each with how good an ending it makes.

	A Latin full stop only ends a sentence when a space or a line follows, so that "3.5"
	and "voice/client.py" survive; CJK punctuation needs no such company.
	"""
	segments: list[tuple[str, int]] = []
	run, strength = "", ANYWHERE
	for i, c in enumerate(text):
		run += c
		nxt = text[i + 1] if i + 1 < len(text) else " "
		if c in SENTENCE_END and (c not in ".!?" or not nxt.strip()):
			strength = STRONG
		elif c in CLAUSE_END or (c == " " and nxt.strip()):
			strength = WEAK
		else:
			# Nothing is a word boundary here, but an utterance still has to end somewhere.
			if len(run) < max_chars:
				continue
			strength = ANYWHERE
		segments.append((run, strength))
		run, strength = "", ANYWHERE
	if run:
		segments.append((run, ANYWHERE))
	return segments


def speech_utterances(text: str, max_seconds: float = MAX_UTTERANCE_S) -> list[str]:
	"""Keep short answers together for prosody; bound long ones by how long they take to say.

	The common one-to-three-sentence reply is one synthesis request, so the model gets its
	surrounding context. The bound is in seconds of speech rather than characters, because
	the same character count is a few seconds of English and half a minute of Mandarin.

	Where to break matters as much as when. Each utterance is a separate request, and the
	synthesizer reads it with no idea what came before, so a cut inside "《蜘蛛人：重生日》"
	hands it half a title and gets back something that is not words. Sentences and lines
	are used first, clauses next, and a bare character boundary only where a script offers
	nothing else.
	"""
	if max_seconds <= 0:
		raise ValueError("max_seconds must be positive")
	max_chars = max(1, int(max_seconds / DENSE_SECONDS))  # no one segment may outlast the budget
	utterances: list[str] = []
	current: list[str] = []
	stop = 0  # segments of `current` up to its last sentence ending

	def flush(upto: int) -> None:
		nonlocal current, stop
		spoken = " ".join("".join(current[:upto]).split())
		if spoken:
			utterances.append(spoken)
		current, stop = current[upto:], 0

	for segment, strength in speech_segments(speech_text(text), max_chars):
		while current and speech_seconds("".join(current) + segment) > max_seconds:
			flush(stop or len(current))  # back up to the last sentence if there was one
		current.append(segment)
		if strength == STRONG:
			stop = len(current)
	flush(len(current))
	return utterances


# --- from Whisper to a turn, and from a cut-off answer to what was heard --------------------

# Whisper's stock output for near-silence. Bracketed text is a sound effect, not speech.
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching",
                  "bye", "the end", "so", "and",
                  "谢谢", "谢谢观看", "谢谢大家", "谢谢收看", "謝謝觀看", "再见", "再見", "下次见"}
# On silence in Chinese, Whisper reaches instead for the subtitle credits and channel
# begging in its training data. These are long and endlessly varied, so they are matched by
# their giveaway words. A turn that genuinely asks about subscriptions is the price.
CREDITS = re.compile(r"amara|字幕|訂閱|订阅|打赏|打賞|点赞|點贊|明镜|明鏡|转发|轉發", re.I)


def usable_transcript(text: str) -> str | None:
	text = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", text or "").strip()
	bare = re.sub(r"[^\w\s']", "", text.lower()).strip()
	if not bare or bare in HALLUCINATIONS or CREDITS.search(bare):
		return None
	return text


def heard_text(sentences, played: int) -> str:
	"""What the user heard of an answer cut at `played` bytes.

	`sentences` are (text, first_byte, last_byte | None) in playback order, with absolute
	speaker byte offsets. Only complete, fully played segments count: without word timing
	we cannot claim the listener heard the end of an interrupted utterance.
	"""
	heard = []
	for text, start, end in sentences:
		if end is not None and end > start and played >= end:
			heard.append(text)
			continue
		break
	return " ".join(heard)
