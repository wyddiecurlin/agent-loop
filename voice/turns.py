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

import random
import re
from dataclasses import dataclass

FRAME_MS = 32  # 512 samples at 16 kHz, the frame Silero VAD v5 wants

# The quiet trace prints every tool call as a line of exactly this shape (loop.py,
# verbose=False). bridge.py turns it into an event; client.py hides the duplicate.
TOOL_LINE = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*$")


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


class Endpointer:
	"""Where a turn starts and ends, from one probability per frame.

	`speech_frames` counts the frames judged speech since the turn opened. A caller that
	fired a speculative transcription remembers the count at that moment; if it is the
	same when "end" arrives, nothing was said in between and the early result stands.
	"""

	def __init__(self, cfg: EndpointConfig | None = None):
		self.cfg = cfg or EndpointConfig()
		self.in_speech = False
		self.speech_run = 0
		self.silence_run = 0
		self.speech_frames = 0
		self.speculated = False

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
			self.silence_run, self.speculated = 0, False
			return "start"
		if p >= c.end_prob:
			self.speech_frames += 1
			self.silence_run = 0
			self.speculated = False  # speech resumed: any early transcription is stale
			return None
		self.silence_run += 1
		if self.silence_run >= frames(c.end_ms):
			self.in_speech, self.speech_run, self.silence_run = False, 0, 0
			return "end" if self.speech_frames >= frames(c.min_speech_ms) else "abort"
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

ACKS = ("Let me check that.", "I'll take a look.")
ANNOUNCE = (
	"I'm using the {tool} tool.",
)
TOOL_ANNOUNCEMENTS = {
	"fs_list": "I'm checking the files.",
	"fs_read": "I'm reading the file.",
	"fs_search": "I'm searching the files.",
	"fs_write": "I'm writing the file.",
	"fs_patch": "I'm updating the file.",
	"shell_run": "I'm running a command.",
	"web_search": "I'm searching the web.",
	"web_fetch": "I'm reading the page.",
	"multiply": "I'm checking the calculation.",
	"substract": "I'm checking the calculation.",
	"get_today_date": "I'm checking the date.",
}
REASSURE = ("I'm still working on this.", "This is taking a little longer.")

SPOKEN_WORDS = {"fs": "file", "substract": "subtract", "today": "today's"}


def spoken_tool(name: str) -> str:
	"""`fs_read` -> "file read", `get_today_date` -> "get today's date"."""
	return " ".join(SPOKEN_WORDS.get(w, w) for w in name.split("_"))


@dataclass
class NarratorConfig:
	ack_after_s: float = 1.5        # fast answers need no acknowledgement
	gap_s: float = 2.5              # least time between two spoken lines
	repeat_tool_after_s: float = 8.0  # the same tool again is worth a word only after this
	reassure_every_s: float = 12.0
	max_reassure: int = 3


class Narrator:
	"""Picks the short lines spoken while the agent is busy. Time is passed in, never read."""

	def __init__(self, cfg: NarratorConfig | None = None, seed: int | None = None):
		self.cfg = cfg or NarratorConfig()
		self.rng = random.Random(seed)
		self.active = False
		self.t0 = 0.0
		self.last_spoken: float | None = None
		self.last_line: str | None = None
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
		if self.pending is not None and (self.last_spoken is None or quiet_for >= c.gap_s):
			name, self.pending = self.pending, None
			self.last_tool, self.last_tool_at, self.last_spoken = name, now, now
			return TOOL_ANNOUNCEMENTS.get(name) or self._pick(ANNOUNCE).format(tool=spoken_tool(name))
		if self.last_spoken is None:
			if quiet_for < c.ack_after_s:
				return None
			self.last_spoken = now
			return self._pick(ACKS)
		if quiet_for >= c.reassure_every_s and self.reassured < c.max_reassure:
			self.reassured += 1
			self.last_spoken = now
			return self._pick(REASSURE)
		return None

	def _pick(self, options: tuple[str, ...]) -> str:
		fresh = [o for o in options if o != self.last_line] or list(options)
		self.last_line = self.rng.choice(fresh)
		return self.last_line


# --- from an answer to utterances ----------------------------------------------------------

SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+(?=\S)|\n+")


def split_sentences(text: str) -> list[str]:
	"""Split at sentence ends; fragments under eight characters ride with the next one."""
	parts = [p.strip() for p in SENTENCE_BREAK.split(text or "") if p and p.strip()]
	merged: list[str] = []
	for part in parts:
		if merged and len(merged[-1]) < 8:
			merged[-1] = f"{merged[-1]} {part}"
		else:
			merged.append(part)
	return merged


def speech_text(text: str) -> str:
	"""Remove synthesis cues and formatting that can leak out of a text answer."""
	text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text or "")
	text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I)
	text = re.sub(r"<\|[^<>]*\|>", "", text)
	text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?(?:^\s*\1\s*$|\Z)",
	              " The code is shown in the terminal. ", text)
	text = re.sub(r"[\[(](?:laughs?|laughing|chuckles?|chuckling|sighs?|sighing|giggles?|giggling)[\])]",
	              "", text, flags=re.I)
	text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", "", text)
	text = re.sub(r"\[([^\]]+)\]\(https?://[^\s)]+\)", r"\1", text)
	text = re.sub(r"(`+|\*{1,2})(.*?)\1", r"\2", text)
	text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
	return re.sub(r"\s+", " ", text).strip()


def split_utterances(text: str, max_chars: int = 300) -> list[str]:
	"""Keep a short answer in one generation for prosody; bound longer requests."""
	parts: list[str] = []
	for sentence in split_sentences(speech_text(text)):
		while len(sentence) > max_chars:
			cut = sentence.rfind(" ", 0, max_chars + 1)
			cut = cut if cut > 0 else max_chars
			parts.append(sentence[:cut].strip())
			sentence = sentence[cut:].strip()
		if sentence:
			parts.append(sentence)
	merged: list[str] = []
	for part in parts:
		if merged and len(merged[-1]) + len(part) + 1 <= max_chars:
			merged[-1] += " " + part
		else:
			merged.append(part)
	return merged


# --- from Whisper to a turn, and from a cut-off answer to what was heard --------------------

# Whisper's stock output for near-silence. Bracketed text is a sound effect, not speech.
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching",
                  "bye", "the end", "so", "and"}


def usable_transcript(text: str) -> str | None:
	text = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", text or "").strip()
	bare = re.sub(r"[^\w\s']", "", text.lower()).strip()
	if not bare or bare in HALLUCINATIONS:
		return None
	return text


def heard_text(sentences, played: int) -> str:
	"""Estimate the heard prefix from absolute playback offsets.

	Without word alignment a partial utterance is only an estimate. Count a proportional
	prefix, rounded down, rather than claiming the whole answer was heard halfway through.
	If generation is unfinished its duration is unknown, so omit that utterance.
	"""
	heard = []
	for text, start, end in sentences:
		if end is None:
			break
		if played >= end:
			heard.append(text)
			continue
		if played > start:
			words = text.split()
			count = int(len(words) * (played - start) / (end - start))
			if count:
				heard.append(" ".join(words[:count]))
		break
	return " ".join(heard)
