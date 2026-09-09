# AI_OWNED
"""The echo gate: is this microphone frame our own voice coming back off the room?

A container cannot reach macOS's echo canceller, and a laptop's speakers are inches from
its microphone, so without this every answer the robot gives is heard again as a turn.
The full cancellation problem (subtracting the echo from the signal) is not needed to
stop that. It is enough to know, per 32 ms frame, whether the microphone carries more
energy than the echo alone would explain -- and we know exactly what was played and
when, because the speaker callback tells us.

	played(t, rms)      every block the speaker emits, with its level (0 when idle)
	heard(t, rms)       -> True if the frame at t is explained by the echo of what was
	                       played `lag` seconds earlier, scaled by `gain`, with a margin
	active(t)           -> is the echo of something still arriving

`lag` and `gain` start from a measured prior and are re-fitted from the two envelopes
whenever playback has run for long enough: the delay through CoreAudio, the speaker, the
room and the input buffer is a property of the machine and the volume knob, not a
constant. Measured on a MacBook Pro at 55% volume: lag 130 ms, gain 0.23, and the
microphone's level within a factor of 3.6 of the prediction 95% of the time.

Pure Python, no clock of its own and no numpy, so it runs and is tested in the container
where the audio half never goes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class EchoConfig:
	lag_s: float = 0.13        # speaker block time -> microphone frame time, until re-fitted
	gain: float = 0.25         # microphone rms per played rms, likewise
	margin: float = 4.0        # more than this times the predicted echo is somebody talking
	floor: float = 0.004       # microphone rms of the quiet room, adapted while idle
	window_s: float = 0.1      # how much earlier than the lag the echo may still arrive
	tail_s: float = 0.3        # the room rings this long after the last block
	max_lag_s: float = 0.5     # how far the fit will look for the echo
	fit_every_s: float = 1.0   # how often to re-fit while playing
	fit_span_s: float = 4.0    # over how much history
	frame_s: float = 0.032     # envelope resolution of the fit


class EchoGate:
	def __init__(self, cfg: EchoConfig | None = None):
		self.cfg = cfg or EchoConfig()
		self.lag, self.gain, self.floor = self.cfg.lag_s, self.cfg.gain, self.cfg.floor
		self.out: deque[tuple[float, float]] = deque()   # (t, rms) played
		self.mic: deque[tuple[float, float]] = deque()   # (t, rms) heard
		self.last_fit = float("-inf")
		self.fits = 0
		self.suppressed = 0
		self.input_rms = self.limit_rms = 0.0

	# -- what was played --------------------------------------------------------------------

	def played(self, t: float, rms: float) -> None:
		self.out.append((t, rms))
		self._trim(self.out, t)

	def _trim(self, q: deque, t: float) -> None:
		keep = self.cfg.fit_span_s + self.cfg.max_lag_s + 1.0
		while q and q[0][0] < t - keep:
			q.popleft()

	def predicted(self, t: float) -> float:
		"""The echo's rms at t: the loudest block played between one lag ago and now, scaled.

		Up to now, not up to one lag ago: a block played ten milliseconds ago cannot be at
		the microphone yet, but when the lag is overestimated the onset of an utterance
		arrives before the window would open, and the first syllable of every answer was
		a turn. Counting the newest blocks costs a little sensitivity in the envelope's
		dips and buys a gate that closes the moment the speaker starts.
		"""
		lo = t - self.lag - self.cfg.window_s
		loud = 0.0
		for when, rms in reversed(self.out):
			if when < lo:
				break
			if rms > loud:
				loud = rms
		return loud * self.gain

	def active(self, t: float) -> bool:
		"""Is the echo of something still arriving, tail included?"""
		lo = t - self.lag - self.cfg.tail_s
		for when, rms in reversed(self.out):
			if when < lo:
				return False
			if rms > 0 and when <= t - self.lag + self.cfg.window_s:
				return True
		return False

	# -- what was heard ---------------------------------------------------------------------

	def heard(self, t: float, rms: float) -> bool:
		"""True when the frame is our own voice: silence to the endpointer and the VAD."""
		self.input_rms, self.limit_rms = rms, 0.0
		self.mic.append((t, rms))
		self._trim(self.mic, t)
		if not self.active(t):
			# The quiet room, learned while nothing plays; a slow follower, so a word does
			# not raise it and a fan that turns on does, eventually.
			self.floor += (min(rms, self.floor * 4) - self.floor) * 0.02
			return False
		self._maybe_fit(t)
		echo = self.predicted(t) * self.cfg.margin + 2 * self.floor
		self.limit_rms = echo
		if rms <= echo:
			self.suppressed += 1
			return True
		return False

	# -- learning the room ------------------------------------------------------------------

	def _envelope(self, q: deque, t0: float, n: int) -> list[float]:
		"""Both queues on one grid of frame_s bins ending now, by the loudest entry per bin."""
		bins = [0.0] * n
		for when, rms in q:
			i = int((when - t0) / self.cfg.frame_s)
			if 0 <= i < n and rms > bins[i]:
				bins[i] = rms
		return bins

	def _maybe_fit(self, t: float) -> None:
		if t - self.last_fit < self.cfg.fit_every_s:
			return
		self.last_fit = t
		c = self.cfg
		n = int(c.fit_span_s / c.frame_s)
		t0 = t - n * c.frame_s
		out, mic = self._envelope(self.out, t0, n), self._envelope(self.mic, t0, n)
		if sum(1 for v in out if v > 0) < n // 4:  # not enough playback to learn from
			return
		best = None
		for lag in range(0, int(c.max_lag_s / c.frame_s) + 1):
			a, b = out[: n - lag], mic[lag:]
			ma, mb = sum(a) / len(a), sum(b) / len(b)
			va = sum((x - ma) ** 2 for x in a)
			vb = sum((y - mb) ** 2 for y in b)
			if va <= 0 or vb <= 0:
				continue
			corr = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5
			if best is None or corr > best[0]:
				best = (corr, lag, a, b)
		if best is None or best[0] < 0.6:
			return
		corr, lag, a, b = best
		# The gain that explains the microphone with the least left over; the user talking
		# over playback pulls it up, so it moves slowly and only within reason.
		num = sum(x * y for x, y in zip(a, b))
		den = sum(x * x for x in a)
		gain = num / den if den > 0 else self.gain
		if 0.01 <= gain <= 5.0:
			self.lag += ((lag * c.frame_s) - self.lag) * 0.5
			self.gain += (gain - self.gain) * 0.5
			self.fits += 1

	def state(self) -> dict:
		return {"lag_ms": round(self.lag * 1000), "gain": round(self.gain, 3), "floor": round(self.floor, 4),
		        "fits": self.fits, "suppressed": self.suppressed,
		        "input_rms": round(self.input_rms, 5), "limit_rms": round(self.limit_rms, 5)}
