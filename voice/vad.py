# AI_OWNED
"""Silero VAD v5 through onnxruntime directly: one speech probability per 32 ms frame.

The `silero-vad` package pulls in torch for the same two megabytes of model, so this
talks to the .onnx file itself. Under a millisecond per frame on one CPU thread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
FRAME = 512      # samples per call at 16 kHz
CONTEXT = 64     # v5 wants the tail of the previous frame prepended


class SileroVAD:
	def __init__(self, model_path: Path):
		opts = ort.SessionOptions()
		opts.inter_op_num_threads = 1
		opts.intra_op_num_threads = 1
		opts.log_severity_level = 3
		self.session = ort.InferenceSession(str(model_path), sess_options=opts,
		                                    providers=["CPUExecutionProvider"])
		self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
		self.reset()

	def reset(self) -> None:
		self.state = np.zeros((2, 1, 128), dtype=np.float32)
		self.context = np.zeros((1, CONTEXT), dtype=np.float32)

	def __call__(self, frame: np.ndarray) -> float:
		"""`frame`: FRAME float32 samples in [-1, 1]. Returns P(speech)."""
		x = np.concatenate([self.context, frame.reshape(1, -1).astype(np.float32)], axis=1)
		out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
		self.context = x[:, -CONTEXT:]
		return float(out[0, 0])
