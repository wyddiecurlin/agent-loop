# AI_OWNED
"""Conversation budgets; the provider catalog owns serving context limits."""
import json
import os

from .providers import (
	DEFAULT_FALLBACK, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_PROVIDER,
	alias_for, default_model, model_spec, resolve_model,
)


def model_limits() -> tuple[int, int]:
	provider = os.getenv("PROVIDER", DEFAULT_PROVIDER).lower()
	model = default_model(provider)
	spec = model_spec(model, provider)
	# Unknown/custom deployments must not inherit a flagship's million-token window.
	context = spec.context if spec else 32_768
	output = spec.max_output if spec else DEFAULT_MAX_OUTPUT_TOKENS
	second = os.getenv("FALLBACK", DEFAULT_FALLBACK.get(provider, "")).lower()
	alias = alias_for(model, provider)
	if alias and second not in ("", "none"):
		try:
			twin = model_spec(resolve_model(alias, second), second)
		except ValueError:
			twin = None
		if twin:
			context = min(context, twin.context)
			output = max(output, twin.max_output)
	if override := os.getenv("CONTEXT_WINDOW_TOKENS"):
		context = int(override)
	if context < 1024:
		raise ValueError("CONTEXT_WINDOW_TOKENS must be at least 1024")
	output = int(os.getenv("MAX_OUTPUT_TOKENS") or os.getenv("QWEN_MAX_OUTPUT_TOKENS") or output)
	return context, output


class ContextBudget:
	def __init__(self, mode="compaction", *, window=None, output=None):
		if mode not in ("auto-clear", "compaction"):
			raise ValueError("mode must be auto-clear or compaction")
		self.mode = mode
		self.window, self.output = (window, output or DEFAULT_MAX_OUTPUT_TOKENS) if window is not None else model_limits()
		self.anchor = None

	@staticmethod
	def size(messages, tools):
		# Byte-based upper estimate handles CJK, JSON arguments, tool results and
		# schemas without assuming the English-only four-characters-per-token rule.
		return len(json.dumps([messages, tools], ensure_ascii=False).encode("utf-8")) + 256

	def estimate(self, messages, tools):
		size = self.size(messages, tools)
		if self.anchor is not None:
			old_size, tokens = self.anchor
			if size >= old_size:
				return tokens + size - old_size
		return size

	def observe(self, messages, tools, usage):
		# Input usage includes cached tokens. Never use cumulative conversation usage
		# or the concurrent preamble/emotion calls to measure the active context.
		if usage is not None and usage.input_tokens > 0:
			self.anchor = (self.size(messages, tools), usage.input_tokens)

	def prepare(self, messages, tools, current_start, max_output_tokens=None):
		if self.mode == "compaction":
			return False  # Reserved: no summarization or history deletion yet.
		reserve = max_output_tokens if max_output_tokens is not None else self.output
		threshold = int(self.window * 0.9)
		cleared = False
		if self.estimate(messages, tools) + reserve >= threshold and current_start > 1:
			# Drop whole previous turns. Current tool calls/results remain paired;
			# clearing must never cause an already executed action to run again.
			retained = [messages[0], *messages[current_start:]]
			if self.size(retained, tools) + reserve >= self.window:
				raise ValueError("This request is too large for the model's context window. Please make it smaller.")
			messages[:] = retained
			self.anchor = None
			cleared = True
		if self.estimate(messages, tools) + reserve >= self.window:
			raise ValueError("This request is too large for the model's context window. Please make it smaller.")
		return cleared
