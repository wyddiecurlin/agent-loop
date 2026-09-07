"""Providers, the wire shapes they speak, and the retry policy around them.

A Provider turns messages + tools into a ModelTurn. Everything above this module
sees only ModelTurn, so adding a backend never reaches the loop.

Two wire shapes, five backends:

  Responses API      openai
  Chat Completions   fireworks | together | baseten | qwen (self-hosted vLLM)

The four Chat Completions backends share one class. They differ in a base URL, a key,
the model ids they answer to, and how they express "do not think" - which is the whole
of `Backend` below.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, TypedDict

from openai import (
	APIConnectionError,
	APIStatusError,
	APITimeoutError,
	OpenAI,
	RateLimitError,
)

from .tools import ToolCall


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.4-nano"  # fastest OpenAI TTFT (~0.67s) with reasoning off
DEFAULT_REASONING_EFFORT = "none"  # no reasoning tokens before the first output token
# 600s, not 60. A task's slowest single request grows with how many agents share the GPU:
# at 24 concurrent containers a request queues behind the others, and a timeout there is
# not a saved second - it is a retry, which costs the GPU the whole generation twice and
# re-rolls the sample, putting variance back into a run we are trying to make repeatable.
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_RETRIES = 3

# Provider selection: PROVIDER=openai (default) | fireworks | together | baseten | qwen
DEFAULT_PROVIDER = "openai"
QWEN_BASE_URL = "http://localhost:9000/v1"
QWEN_MODEL = "qwen3.5-9b"

# Repeatability comes from the SEED, not from temperature=0. That distinction is the
# whole of this block.
#
# temperature=0 was tried and is a trap here. Greedy decoding walked this model into an
# unbounded reasoning loop: 31,360 output tokens, all of them reasoning, no text and no
# tool call, on task after task. Sampling normally breaks such a loop by taking a
# different token; greedy cannot, because the argmax that started it is the same argmax
# every time. So we sample, at the values the model card gives for precise coding, and
# fix the seed - which makes the sampling reproducible without making it degenerate.
#
# Even so this is not bitwise reproducible: under continuous batching the kernel path
# depends on what else is in the batch, so 24 concurrent agents still flip the occasional
# token. Expect much less run-to-run noise, not none, and compare two runs paired. On a
# hosted platform the batch contains *other tenants'* traffic, so the seed buys less
# there than it does on our own box - it is still worth sending, and still not a promise.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_SEED = 0

# A turn that wants more than this has stopped doing the task. The runaway above cost
# 31k tokens per step and would have repeated for every one of 60 steps; a cap turns an
# unbounded loop into a bounded one that the loop's stall check can then end.
#
# It is per-model in the catalog, because a model that cannot be told to stop reasoning
# spends this budget on thinking before it writes a single tool call, and 8192 would
# truncate it mid-thought - which reads as a broken model, not a tight cap.
DEFAULT_MAX_OUTPUT_TOKENS = 8192


# ---------------------------------------------------------------------------
# Catalog: what each platform serves, what it costs, and how it reasons
# ---------------------------------------------------------------------------
#
# The same six models are sold by three platforms at three different prices, and two of
# them serve `zai-org/GLM-5.3` under that identical id at different cached-input rates -
# so the model id alone cannot bill a call, and every price lookup is keyed by the pair.
# Getting this wrong is not a crash; it is a plausible-looking invoice that is 2x off.
#
# Prices are USD per 1M tokens, read from each platform's own pricing page on 2026-09-07
# and unverified since. `./test.sh pricing` re-reads them; if a number here disagrees
# with the page, the page is right.
#
# `reasoning` is not cosmetic. GLM 5.3 and Kimi K3 reason unconditionally: a request that
# asks either of them for effort "none" is rejected outright, so "none" is absent from
# their tuples and must stay absent. We send the first entry - the cheapest the model
# will accept - unless REASONING_EFFORT says otherwise.

@dataclass(frozen=True)
class Model:
	"""One model as one platform serves it: the id to send, the price, the reasoning floor."""
	id: str  # the exact string this platform's API wants
	input: float  # USD per 1M tokens
	cached_input: float
	output: float
	reasoning: tuple[str, ...] = ("none",)  # accepted reasoning_effort, cheapest first
	context: int = 128_000
	max_output: int = DEFAULT_MAX_OUTPUT_TOKENS


# Effort ladders, named once so the tables below stay readable.
_TOGGLEABLE = ("none", "low", "medium", "high", "max")  # reasoning can be turned off
_ALWAYS_ON = ("low", "high", "max")  # it cannot; "max" is the model's own default
_M = 1_048_576

CATALOG: dict[str, dict[str, Model]] = {
	"fireworks": {
		"deepseek-v4-pro":  Model("accounts/fireworks/models/deepseek-v4-pro",  1.32, 0.044, 3.96, _TOGGLEABLE, _M),
		"deepseek-v4-flash": Model("accounts/fireworks/models/deepseek-v4-flash", 0.22, 0.007, 0.66, _TOGGLEABLE, _M),
		"glm-5.3":          Model("accounts/fireworks/models/glm-5p3",          1.40, 0.26,  4.40, _ALWAYS_ON,  _M, 32_768),
		"glm-5.3-flash":    Model("accounts/fireworks/models/glm-5p3-flash",    0.15, 0.03,  0.50, _ALWAYS_ON,  _M, 32_768),
		"kimi-k3":          Model("accounts/fireworks/models/kimi-k3",          3.00, 0.30, 15.00, _ALWAYS_ON,  _M, 32_768),
		"qwen-3.7-plus":    Model("accounts/fireworks/models/qwen3p7-plus",     0.40, 0.08,  1.60, _TOGGLEABLE, 262_144),
		"qwen-3.8-max":     Model("accounts/fireworks/models/qwen3p8-max",      2.00, 0.25,  6.00, _TOGGLEABLE, _M),
	},
	"together": {
		"deepseek-v4-pro":  Model("deepseek-ai/DeepSeek-V4-Pro-0813",   1.32, 0.13, 3.96, _TOGGLEABLE, _M),
		"deepseek-v4-flash": Model("deepseek-ai/DeepSeek-V4-Flash-0731", 0.14, 0.03, 0.28, _TOGGLEABLE, _M),
		"glm-5.3":          Model("zai-org/GLM-5.3",                    1.40, 0.26, 4.40, _ALWAYS_ON,  _M, 32_768),
		"glm-5.3-flash":    Model("zai-org/GLM-5.3-Flash",              0.15, 0.03, 0.50, _ALWAYS_ON,  _M, 32_768),
		"kimi-k3":          Model("moonshotai/Kimi-K3",                 3.00, 0.30, 15.00, _ALWAYS_ON, _M, 32_768),
		# No cached-input rate is published for either Qwen; billed here at the full input
		# rate, which over-states the cost rather than under-stating it.
		"qwen-3.7-plus":    Model("Qwen/Qwen3.7-Plus",                  0.32, 0.32, 1.28, _TOGGLEABLE, 1_000_000),
		"qwen-3.8-max":     Model("Qwen/Qwen3.8-2.4T-A95B",             2.00, 0.25, 6.00, _TOGGLEABLE, _M),
	},
	# Baseten's Model APIs catalog carries neither Qwen tier, so a Qwen run here is a
	# KeyError rather than a silent fallback to a different model.
	"baseten": {
		"deepseek-v4-pro":  Model("deepseek-ai/DeepSeek-V4-Pro",  1.32, 0.13, 3.96, _TOGGLEABLE, _M),
		"deepseek-v4-flash": Model("deepseek-ai/DeepSeek-V4-Flash", 0.13, 0.03, 0.26, _TOGGLEABLE, _M),
		"glm-5.3":          Model("zai-org/GLM-5.3",              1.40, 0.14, 4.40, _ALWAYS_ON,  _M, 32_768),
		"glm-5.3-flash":    Model("zai-org/GLM-5.3-Flash",        0.15, 0.03, 0.50, _ALWAYS_ON,  _M, 32_768),
		"kimi-k3":          Model("moonshotai/Kimi-K3",           3.00, 0.30, 15.00, _ALWAYS_ON, _M, 32_768),
	},
	# Self-hosted vLLM on the local box: the GPU is already paid for, so per-token is 0.
	"qwen": {
		"qwen3.5-9b": Model(QWEN_MODEL, 0.0, 0.0, 0.0, _TOGGLEABLE, 32_768),
	},
	"openai": {
		"gpt-5.4-nano": Model("gpt-5.4-nano", 0.20, 0.02, 1.25, _TOGGLEABLE),
		"gpt-5-nano":   Model("gpt-5-nano",   0.05, 0.005, 0.40, _TOGGLEABLE),
		"gpt-5-mini":   Model("gpt-5-mini",   0.25, 0.025, 2.00, _TOGGLEABLE),
		"gpt-5":        Model("gpt-5",        1.25, 0.125, 10.00, _TOGGLEABLE),
	},
}

# USD per 1M tokens, keyed by (platform, id-on-the-wire). Derived, never edited: the
# catalog is the one place a price is written down.
PRICING: dict[tuple[str, str], dict[str, float]] = {
	(provider, m.id): {"input": m.input, "cached_input": m.cached_input, "output": m.output}
	for provider, models in CATALOG.items()
	for m in models.values()
}


def resolve_model(name: str, provider: str) -> str:
	"""Map a catalog alias to the id `provider` wants; pass anything else through.

	So `MODEL=kimi-k3` is portable across platforms and `MODEL=accounts/fireworks/...`
	still works for a model the catalog has never heard of.
	"""
	entry = CATALOG.get(provider, {}).get(name)
	if entry is not None:
		return entry.id
	if name in {alias for models in CATALOG.values() for alias in models}:
		known = ", ".join(sorted(CATALOG.get(provider, {})))
		raise ValueError(f"{provider!r} does not serve {name!r}; it has: {known}")
	return name


def model_spec(model_id: str, provider: str) -> Model | None:
	"""The catalog entry for an id already resolved for `provider`, or None if unlisted."""
	for entry in CATALOG.get(provider, {}).values():
		if entry.id == model_id:
			return entry
	return None


# ---------------------------------------------------------------------------
# Usage and cost
# ---------------------------------------------------------------------------

@dataclass
class Usage:
	input_tokens: int = 0
	cached_input_tokens: int = 0  # subset of input_tokens served from cache
	output_tokens: int = 0
	reasoning_tokens: int = 0  # subset of output_tokens
	cost_usd: float = 0.0

	@property
	def total_tokens(self) -> int:
		return self.input_tokens + self.output_tokens

	def __add__(self, other: "Usage") -> "Usage":
		return Usage(
			input_tokens=self.input_tokens + other.input_tokens,
			cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
			output_tokens=self.output_tokens + other.output_tokens,
			reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
			cost_usd=self.cost_usd + other.cost_usd,
		)


def compute_cost(model: str, input_tokens: int, cached_input_tokens: int,
                 output_tokens: int, provider: str = "") -> float:
	"""Return USD cost for one call, or 0.0 if that platform has no price for the model.

	`provider` is not optional in spirit: two platforms serve GLM 5.3 under the same id at
	different cached rates, so a price without a platform is a coin flip. It defaults to
	"" so an unpriced call is free rather than a crash mid-eval.
	"""
	price = PRICING.get((provider, model))
	if price is None:
		return 0.0
	uncached = max(input_tokens - cached_input_tokens, 0)
	return (
		uncached * price["input"]
		+ cached_input_tokens * price["cached_input"]
		+ output_tokens * price["output"]
	) / 1_000_000


@dataclass
class CostTracker:
	"""Accumulates usage across calls so the agent loop can report totals / enforce budgets."""
	total: Usage = field(default_factory=Usage)
	calls: int = 0

	def record(self, turn) -> None:
		"""`turn` is anything with a `.usage`; typed loosely so this module needs no
		provider import and can be shared by every backend."""
		self.calls += 1
		if turn.usage is not None:
			self.total = self.total + turn.usage

	def summary(self) -> str:
		u = self.total
		return (
			f"calls={self.calls} "
			f"input={u.input_tokens} (cached={u.cached_input_tokens}) "
			f"output={u.output_tokens} (reasoning={u.reasoning_tokens}) "
			f"total={u.total_tokens} "
			f"cost=${u.cost_usd:.6f}"
		)


TRACKER = CostTracker()


# ---------------------------------------------------------------------------
# Input message shapes (Responses API `input` items)
# ---------------------------------------------------------------------------

class Message(TypedDict):
	"""A chat turn from the user, system, or assistant."""
	role: Literal["user", "assistant", "system", "developer"]
	content: str


class FunctionCallItem(TypedDict):
	"""The assistant's tool call, echoed back into history exactly as the model emitted it."""
	type: Literal["function_call"]
	call_id: str
	name: str
	arguments: str  # JSON string


class FunctionCallOutputItem(TypedDict):
	"""Our result for a tool call; call_id must match the FunctionCallItem it answers."""
	type: Literal["function_call_output"]
	call_id: str
	output: str  # tool result serialized to a string (json.dumps for dicts)


InputItem = Message | FunctionCallItem | FunctionCallOutputItem
# `input` may be a bare string (shorthand for one user message) or an ordered list of items.
Messages = str | list[InputItem]


# ---------------------------------------------------------------------------
# Model turn + provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class ModelTurn:
	text: str | None
	tool_calls: list[ToolCall] | None
	usage: Usage | None
	stop_reason: str | None


OnText = Callable[[str], None]
OnToolCall = Callable[[str, Literal['function_call', 'function_args']], None]


class Provider(Protocol):
	"""Anything that can turn messages + tools into a ModelTurn."""

	def generate(
		self,
		messages: Messages,
		model: str,
		tools: list[dict] | None,
		*,
		stream: bool,
		on_text: OnText | None,
		on_tool_call: OnToolCall | None,
		timeout: float,
		tool_choice: str | None,
	) -> ModelTurn: ...


class OpenAIProvider:
	def __init__(self, client: OpenAI | None = None, timeout: float = DEFAULT_TIMEOUT_S):
		# max_retries=0: the retry loop in with_retries() owns retries, not the SDK.
		self.client = client or OpenAI(timeout=timeout, max_retries=0)

	def generate(
		self,
		messages: Messages,
		model: str,
		tools: list[dict] | None,
		*,
		stream: bool = False,
		on_text: OnText | None = None,
		on_tool_call: OnToolCall | None = None,
		timeout: float = DEFAULT_TIMEOUT_S,
		tool_choice: str | None = None,
	) -> ModelTurn:
		kwargs: dict = {
			"model": model,
			"input": messages,
			"timeout": timeout,
			"reasoning": {"effort": DEFAULT_REASONING_EFFORT},
		}
		if tools:
			kwargs["tools"] = tools
			if tool_choice:
				kwargs["tool_choice"] = tool_choice

		if stream:
			with self.client.responses.stream(**kwargs) as s:
				for event in s:
					if event.type == "response.output_text.delta" and on_text:
						on_text(event.delta)
					if event.type == "response.output_item.added" and on_tool_call and event.item.type=='function_call':
						on_tool_call(event.item.name, "function_call")
					if event.type == "response.function_call_arguments.delta" and on_tool_call:
						on_tool_call(event.delta, "function_args")

				response = s.get_final_response()
		else:
			response = self.client.responses.create(**kwargs)

		return self._to_turn(response, model)

	@staticmethod
	def _to_turn(response, model: str) -> ModelTurn:
		tool_calls = [
			ToolCall(id=item.call_id, name=item.name, arguments=item.arguments)
			for item in response.output
			if item.type == "function_call"
		]

		usage = None
		if response.usage is not None:
			in_details = getattr(response.usage, "input_tokens_details", None)
			out_details = getattr(response.usage, "output_tokens_details", None)
			cached = getattr(in_details, "cached_tokens", 0) or 0
			reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
			usage = Usage(
				input_tokens=response.usage.input_tokens,
				cached_input_tokens=cached,
				output_tokens=response.usage.output_tokens,
				reasoning_tokens=reasoning,
				cost_usd=compute_cost(model, response.usage.input_tokens, cached,
				                      response.usage.output_tokens, "openai"),
			)

		return ModelTurn(
			text=response.output_text or None,
			tool_calls=tool_calls or None,
			usage=usage,
			stop_reason=response.status,
		)


# ---------------------------------------------------------------------------
# Chat Completions backends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Backend:
	"""A platform that speaks OpenAI Chat Completions, and the ways they differ.

	Four fields, because after the base URL and the key there are only two real
	differences between a hosted platform and our own vLLM: how you ask for less
	thinking, and whether `top_k` is a first-class parameter.
	"""
	name: str
	base_url: str
	base_url_env: str
	api_key_env: str
	# "effort"        -> OpenAI's reasoning_effort, what the hosted platforms take
	# "chat_template" -> vLLM's chat_template_kwargs.enable_thinking flag
	reasoning_style: Literal["effort", "chat_template"] = "effort"
	# top_k is not an OpenAI parameter. vLLM takes it in extra_body; Fireworks, Together
	# and Baseten accept it top-level.
	top_k_in_extra_body: bool = False


BACKENDS: dict[str, Backend] = {
	"fireworks": Backend(
		"fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_BASE_URL", "FIREWORKS_API_KEY"),
	"together": Backend(
		"together", "https://api.together.xyz/v1", "TOGETHER_BASE_URL", "TOGETHER_API_KEY"),
	"baseten": Backend(
		"baseten", "https://inference.baseten.co/v1", "BASETEN_BASE_URL", "BASETEN_API_KEY"),
	"qwen": Backend(
		"qwen", QWEN_BASE_URL, "QWEN_BASE_URL", "QWEN_API_KEY",
		reasoning_style="chat_template", top_k_in_extra_body=True),
}


class ChatProvider:
	"""Any OpenAI-compatible Chat Completions endpoint, hosted or self-hosted.

	The rest of the loop speaks the Responses API shapes (Message / FunctionCallItem /
	FunctionCallOutputItem, tools as {"type","name","parameters"}), so this provider
	translates both directions and returns the same ModelTurn as OpenAIProvider.

	Env, per backend: <NAME>_BASE_URL, <NAME>_API_KEY. Shared knobs:
	  REASONING_EFFORT   override the catalog's floor ("low"/"high"/"max"/an int budget)
	  MAX_OUTPUT_TOKENS  override the catalog's per-model cap
	  QWEN_THINKING=1    the vLLM backend only, where effort is a boolean and not a dial
	"""

	def __init__(
		self,
		backend: Backend | str = "qwen",
		client: OpenAI | None = None,
		timeout: float = DEFAULT_TIMEOUT_S,
		base_url: str | None = None,
		api_key: str | None = None,
		thinking: bool | None = None,
		reasoning_effort: str | None = None,
		temperature: float | None = None,
		seed: int | None = None,
		max_output_tokens: int | None = None,
	):
		self.backend = BACKENDS[backend] if isinstance(backend, str) else backend
		b = self.backend
		self.client = client or OpenAI(
			base_url=base_url or os.getenv(b.base_url_env, b.base_url),
			# "EMPTY" is what vLLM wants and what a hosted platform will reject with a
			# 401 - which is the right failure: a missing key should not look like a
			# model error 200 requests into an eval.
			api_key=api_key or os.getenv(b.api_key_env, "EMPTY"),
			timeout=timeout,
			max_retries=0,
		)
		# QWEN_THINKING is the vLLM box's own switch and is read only there. On a hosted
		# platform the knob is REASONING_EFFORT, which says how much rather than whether -
		# and leaving this env-driven let a stale QWEN_THINKING=1 in .env quietly buy
		# max-effort reasoning on every Fireworks call. ./test.sh providers caught it.
		self.thinking = thinking if thinking is not None else (
			b.reasoning_style == "chat_template" and os.getenv("QWEN_THINKING", "0") == "1")
		self.reasoning_effort = reasoning_effort or os.getenv("REASONING_EFFORT") or None
		self.temperature = temperature if temperature is not None else float(
			os.getenv("QWEN_TEMPERATURE", DEFAULT_TEMPERATURE))
		self.seed = seed if seed is not None else int(os.getenv("QWEN_SEED", DEFAULT_SEED))
		self.max_output_tokens = max_output_tokens if max_output_tokens is not None else (
			int(env) if (env := os.getenv("MAX_OUTPUT_TOKENS") or os.getenv("QWEN_MAX_OUTPUT_TOKENS")) else None)

	# -- Responses-style input -> Chat Completions messages --------------------

	@staticmethod
	def _to_chat_messages(messages: Messages) -> list[dict]:
		if isinstance(messages, str):
			return [{"role": "user", "content": messages}]
		out: list[dict] = []
		for item in messages:
			kind = item.get("type")
			if kind == "function_call":
				call = {
					"id": item["call_id"],
					"type": "function",
					"function": {"name": item["name"], "arguments": item["arguments"]},
				}
				# Merge consecutive tool calls into one assistant message (chat format).
				if out and out[-1]["role"] == "assistant" and out[-1].get("tool_calls") and out[-1].get("content") is None:
					out[-1]["tool_calls"].append(call)
				else:
					out.append({"role": "assistant", "content": None, "tool_calls": [call]})
			elif kind == "function_call_output":
				out.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
			else:
				role = "system" if item["role"] == "developer" else item["role"]
				out.append({"role": role, "content": item["content"]})
		return out

	@staticmethod
	def _to_chat_tools(tools: list[dict]) -> list[dict]:
		converted = []
		for t in tools:
			if "function" in t:  # already chat format
				converted.append(t)
				continue
			converted.append({
				"type": "function",
				"function": {
					"name": t["name"],
					"description": t.get("description", ""),
					"parameters": t.get("parameters", {"type": "object", "properties": {}}),
				},
			})
		return converted

	# -- how much to think -----------------------------------------------------

	def _reasoning(self, kwargs: dict, extra_body: dict, spec: Model | None) -> None:
		"""Write the backend's own spelling of "think this much" into the request.

		The floor is the model's, not ours: asking GLM 5.3 or Kimi K3 for "none" is a 400,
		so the catalog's first entry is the cheapest thing each will actually accept.
		"""
		if self.backend.reasoning_style == "chat_template":
			extra_body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
			return
		ladder = spec.reasoning if spec else _TOGGLEABLE
		kwargs["reasoning_effort"] = self.reasoning_effort or (ladder[-1] if self.thinking else ladder[0])

	def generate(
		self,
		messages: Messages,
		model: str,
		tools: list[dict] | None,
		*,
		stream: bool = False,
		on_text: OnText | None = None,
		on_tool_call: OnToolCall | None = None,
		timeout: float = DEFAULT_TIMEOUT_S,
		tool_choice: str | None = None,
	) -> ModelTurn:
		spec = model_spec(model, self.backend.name)
		extra_body: dict = {}
		kwargs: dict = {
			"model": model,
			"messages": self._to_chat_messages(messages),
			"timeout": timeout,
			"temperature": self.temperature,
			"top_p": DEFAULT_TOP_P,
			"seed": self.seed,
			"max_tokens": self.max_output_tokens or (spec.max_output if spec else DEFAULT_MAX_OUTPUT_TOKENS),
		}
		if self.backend.top_k_in_extra_body:
			extra_body["top_k"] = DEFAULT_TOP_K
		else:
			kwargs["top_k"] = DEFAULT_TOP_K
		self._reasoning(kwargs, extra_body, spec)
		if extra_body:
			kwargs["extra_body"] = extra_body
		if tools:
			kwargs["tools"] = self._to_chat_tools(tools)
			kwargs["tool_choice"] = tool_choice or "auto"

		if not stream:
			resp = self.client.chat.completions.create(**kwargs)
			choice = resp.choices[0]
			text = choice.message.content or None
			calls = [
				ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
				for tc in (choice.message.tool_calls or [])
			]
			return self._to_turn(text, calls, resp.usage, choice.finish_reason, model)

		# Streaming: accumulate text + tool-call fragments keyed by index.
		text_parts: list[str] = []
		pending: dict[int, dict] = {}
		usage = None
		finish = None
		for chunk in self.client.chat.completions.create(
			stream=True, stream_options={"include_usage": True}, **kwargs
		):
			if chunk.usage:
				usage = chunk.usage
			if not chunk.choices:
				continue
			choice = chunk.choices[0]
			finish = choice.finish_reason or finish
			delta = choice.delta
			if delta.content:
				text_parts.append(delta.content)
				if on_text:
					on_text(delta.content)
			for tc in delta.tool_calls or []:
				slot = pending.setdefault(tc.index, {"id": None, "name": None, "args": []})
				if tc.id:
					slot["id"] = tc.id
				fn = tc.function
				if fn and fn.name:
					slot["name"] = fn.name
					if on_tool_call:
						on_tool_call(fn.name, "function_call")
				if fn and fn.arguments:
					slot["args"].append(fn.arguments)
					if on_tool_call:
						on_tool_call(fn.arguments, "function_args")

		calls = [
			ToolCall(
				id=slot["id"] or f"call_{i}",
				name=slot["name"] or "",
				arguments="".join(slot["args"]) or "{}",
			)
			for i, slot in sorted(pending.items())
		]
		return self._to_turn("".join(text_parts) or None, calls, usage, finish, model)

	def _to_turn(self, text, calls, raw_usage, finish_reason, model: str) -> ModelTurn:
		usage = None
		if raw_usage is not None:
			in_details = getattr(raw_usage, "prompt_tokens_details", None)
			out_details = getattr(raw_usage, "completion_tokens_details", None)
			cached = getattr(in_details, "cached_tokens", 0) or 0
			reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
			usage = Usage(
				input_tokens=raw_usage.prompt_tokens,
				cached_input_tokens=cached,
				output_tokens=raw_usage.completion_tokens,
				reasoning_tokens=reasoning,
				cost_usd=compute_cost(model, raw_usage.prompt_tokens, cached,
				                      raw_usage.completion_tokens, self.backend.name),
			)
		# Map chat finish_reason onto the Responses-style status the loop expects.
		stop = {"stop": "completed", "tool_calls": "completed", "length": "incomplete"}.get(finish_reason, finish_reason)
		return ModelTurn(text=text, tool_calls=calls or None, usage=usage, stop_reason=stop)


class QwenProvider(ChatProvider):
	"""The self-hosted vLLM box, pinned. Kept as a name because the evals refer to it."""

	def __init__(self, **kw):
		super().__init__(backend="qwen", **kw)


def make_provider(name: str | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> Provider:
	"""Build the provider named by `name` (or the PROVIDER env var)."""
	name = (name or os.getenv("PROVIDER", DEFAULT_PROVIDER)).lower()
	if name in BACKENDS:
		backend = BACKENDS[name]
		# vLLM is happy without a key; a hosted platform is not, and finding that out on
		# request 1 beats finding it out on request 200 of an eval shard.
		if name != "qwen" and not os.getenv(backend.api_key_env):
			raise RuntimeError(f"{backend.api_key_env} is not set in the environment or .env")
		return ChatProvider(backend=backend, timeout=timeout)
	# NOTE: OpenAIProvider deliberately sends neither temperature nor seed. The Responses
	# API reasoning models (gpt-5*) reject `temperature` outright, so there is no greedy
	# setting to ask for - an OpenAI run cannot be made as repeatable as the others.
	if name == "openai":
		if not os.getenv("OPENAI_API_KEY"):
			raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
		return OpenAIProvider(timeout=timeout)
	known = ", ".join(["openai", *BACKENDS])
	raise ValueError(f"unknown PROVIDER {name!r} (expected one of: {known})")


# The model each platform gets when the caller names none. Not the best model on the
# platform - the one whose price makes a 164-task eval a rounding error.
DEFAULT_MODELS: dict[str, str] = {
	"openai": DEFAULT_MODEL,
	"qwen": QWEN_MODEL,
	"fireworks": "glm-5.3-flash",
	"together": "glm-5.3-flash",
	"baseten": "glm-5.3-flash",
}


def default_model(provider_name: str | None = None) -> str:
	"""Model to use when the caller didn't pick one: depends on the active provider.

	MODEL takes a catalog alias (`kimi-k3`) or a raw id; the per-provider env vars are
	kept because the evals and run.sh already pass them.
	"""
	name = (provider_name or os.getenv("PROVIDER", DEFAULT_PROVIDER)).lower()
	legacy = {"qwen": "QWEN_MODEL", "openai": "OPENAI_MODEL"}.get(name)
	chosen = os.getenv("MODEL") or (os.getenv(legacy) if legacy else None) or DEFAULT_MODELS.get(name, DEFAULT_MODEL)
	return resolve_model(chosen, name)


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

# openai>=3 vendors its transport as httpx2, and a transport-level timeout on the
# streaming path reaches us raw rather than wrapped in APITimeoutError - so retry the
# transport exceptions too, or a slow server kills the whole run.
try:
	import httpx2 as _httpx
except ModuleNotFoundError:  # older SDKs ship plain httpx
	import httpx as _httpx

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, _httpx.TransportError)


def is_retryable(exc: Exception) -> bool:
	if isinstance(exc, RETRYABLE_ERRORS):
		return True
	# 503 Service Overloaded is the shape Fireworks load-sheds with, and it arrives while
	# you are *inside* your rate limit - so it is transient by definition, not a bug to
	# surface. 429 arrives as RateLimitError above.
	return isinstance(exc, APIStatusError) and exc.status_code >= 500


def with_retries(
	fn: Callable[[], ModelTurn],
	max_retries: int = DEFAULT_MAX_RETRIES,
	base_delay: float = 1.0,
	max_delay: float = 30.0,
) -> ModelTurn:
	"""Call fn, retrying transient failures with exponential backoff + jitter."""
	for attempt in range(max_retries + 1):
		try:
			return fn()
		except Exception as exc:
			if not is_retryable(exc) or attempt == max_retries:
				raise
			delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)
			time.sleep(delay)
	raise AssertionError("unreachable")
