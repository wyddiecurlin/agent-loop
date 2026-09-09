# AI_OWNED
"""The robot in the voice: one fixed effect chain over every byte the speaker plays.

The synthesizer decides the words and the delivery; this decides the *timbre*, and it is
the same for a filler, a preamble and an answer, so the character never changes mid-turn.
The chain, on 24 kHz mono PCM, in order:

	band-pass    a small speaker: nothing under ~250 Hz or over ~5.5 kHz
	vibrato      a slow, shallow pitch wobble from a modulated delay: the analog warble
	ring mod     the signal times a low carrier, mixed in under the dry voice: the metal
	drive        a soft clip, for warmth and a little grit
	crush        fewer bits than the file has: the hiss and edge of an old converter

Every stage carries its state across calls, so audio fed in 4 KB chunks comes out the
same as audio fed at once (`tests/test_voice_audio.py` checks this), and the output has
exactly as many bytes as the input, so byte offsets taken before or after it agree.
`amount` scales the character from 0 (the plain voice) to 1; the defaults are the voice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotConfig:
	low_hz: float = 250.0        # band-pass edges: the speaker in the robot's chest
	high_hz: float = 5500.0
	taps: int = 255              # FIR length: odd, and the whole state the filter keeps
	vibrato_hz: float = 5.5      # the warble's rate ...
	vibrato_ms: float = 0.45     # ... and depth, as a delay swing (about +-1.5% in pitch)
	carrier_hz: float = 140.0    # ring modulator carrier: the metal in the voice
	ring_mix: float = 0.35       # how much of it, against the dry voice
	drive: float = 1.8           # soft-clip gain before the tanh
	bits: int = 9                # output word length before it goes back to 16
	hiss: float = 0.0025         # analog noise floor, linear amplitude


class RobotVoice:
	"""Stateful; call `process` with consecutive chunks of one stream, `reset` between streams."""

	def __init__(self, rate: int = 24_000, cfg: RobotConfig | None = None, amount: float = 1.0, seed: int = 7):
		self.rate, self.cfg, self.amount = rate, cfg or RobotConfig(), max(0.0, min(1.0, amount))
		self.taps = self._bandpass(self.cfg.low_hz, self.cfg.high_hz, self.cfg.taps)
		self.max_delay = int(rate * (self.cfg.vibrato_ms * 2 + 1) / 1000) + 2
		self.rng = np.random.default_rng(seed)
		self.reset()

	def reset(self) -> None:
		self.fir_tail = np.zeros(len(self.taps) - 1, dtype=np.float32)
		self.delay_tail = np.zeros(self.max_delay, dtype=np.float32)
		self.vib_phase = 0.0
		self.ring_phase = 0.0

	def _bandpass(self, low: float, high: float, taps: int) -> np.ndarray:
		"""A windowed-sinc FIR: linear phase, and vectorisable, which a biquad is not."""
		n = np.arange(taps) - (taps - 1) / 2
		lp = lambda fc: np.sinc(2 * fc / self.rate * n) * 2 * fc / self.rate  # noqa: E731
		kernel = lp(high) - lp(low)
		kernel *= np.hamming(taps)
		return (kernel / np.sum(lp(high) * np.hamming(taps))).astype(np.float32)

	def process(self, pcm: bytes) -> bytes:
		if not pcm or self.amount <= 0:
			return pcm
		usable = len(pcm) - len(pcm) % 2
		x = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
		n = len(x)
		a, c = self.amount, self.cfg

		# band-pass, with the previous chunk's tail so the convolution has no seam
		full = np.concatenate([self.fir_tail, x])
		y = np.convolve(full, self.taps, mode="valid")[:n].astype(np.float32)
		self.fir_tail = full[-(len(self.taps) - 1):]
		y = x + a * (y - x)

		# vibrato: read the signal back through a delay that sways with a slow sine
		t = np.arange(n, dtype=np.float64)
		phase = self.vib_phase + 2 * np.pi * c.vibrato_hz * t / self.rate
		self.vib_phase = float((phase[-1] + 2 * np.pi * c.vibrato_hz / self.rate) % (2 * np.pi))
		depth = self.rate * c.vibrato_ms / 1000 * a
		delay = self.rate * c.vibrato_ms / 1000 + 1 + depth * np.sin(phase)
		buf = np.concatenate([self.delay_tail, y])
		pos = len(self.delay_tail) + t - delay
		y = np.interp(pos, np.arange(len(buf)), buf).astype(np.float32)
		self.delay_tail = buf[-self.max_delay:]

		# ring modulation, mixed under the dry voice so the words stay
		phase = self.ring_phase + 2 * np.pi * c.carrier_hz * t / self.rate
		self.ring_phase = float((phase[-1] + 2 * np.pi * c.carrier_hz / self.rate) % (2 * np.pi))
		mix = c.ring_mix * a
		y = y * (1 - mix) + y * np.sin(phase).astype(np.float32) * mix

		# drive, then fewer bits, then a whisper of hiss under it all
		drive = 1 + (c.drive - 1) * a
		y = np.tanh(y * drive) / np.tanh(drive)
		bits = 16 - (16 - c.bits) * a
		step = 2.0 ** (1 - bits)
		y = np.round(y / step) * step
		if c.hiss > 0:
			y = y + self.rng.standard_normal(n).astype(np.float32) * (c.hiss * a)

		out = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
		return out + pcm[usable:]


if __name__ == "__main__":  # a listening test: voice/robot.py in.wav out.wav [amount]
	import sys
	import wave

	with wave.open(sys.argv[1], "rb") as w:
		rate, frames = w.getframerate(), w.readframes(w.getnframes())
	fx = RobotVoice(rate, amount=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
	out = b"".join(fx.process(frames[i:i + 4096]) for i in range(0, len(frames), 4096))
	with wave.open(sys.argv[2], "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(rate)
		w.writeframes(out)
