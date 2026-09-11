# AI_OWNED
"""Providers, the wire shapes they speak, and the retry policy around them.

A Provider turns messages + tools into a ModelTurn. Everything above this module
sees only ModelTurn, so adding a backend never reaches the loop.

Two wire shapes, four backends:

  Responses API      openai
  Chat Completions   fireworks | together | qwen (self-hosted vLLM)

The three Chat Completions backends share one class. They differ in a base URL, a key,
the model ids they answer to, and how they express "do not think" - which is the whole
of `Backend` below. Everything that is not an OpenAI parameter goes in `extra_body`
regardless of platform, because the SDK checks names before the server ever sees them.

Fireworks and Together serve the same six models, so `FallbackProvider` pairs them: when
Fireworks will not serve a request, the same weights are one alias lookup away.
"""

import json
import os
import random
import sys
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

DEFAULT_MODEL = "gpt-5.4-nano"  # the openai backend's pick: fastest TTFT (~0.67s), reasoning off
DEFAULT_REASONING_EFFORT = "none"  # no reasoning tokens before the first output token
# 600s, not 60. A task's slowest single request grows with how many agents share the GPU:
# at 24 concurrent containers a request queues behind the others, and a timeout there is
# not a saved second - it is a retry, which costs the GPU the whole generation twice and
# re-rolls the sample, putting variance back into a run we are trying to make repeatable.
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_RETRIES = 3
# An interactive front end wants the opposite trade: a stall there is a person listening to
# silence, and 600s of it is indistinguishable from a hang. AGENT_TIMEOUT_S and
# AGENT_MAX_RETRIES override the two, and they override the *caller* as well as this
# default -- agent_loop() binds both into its signature at import and takes no argument for
# either, so the environment is the only way in. voice/bridge.py uses it.


def _env(name: str, default, cast=float):
	try:
		return cast(os.environ[name])
	except (KeyError, ValueError):  # unset, or a typo in .env: the default still runs
		return default

# Provider selection: PROVIDER=fireworks (default) | together | openai | qwen
# Fireworks is the default because it is the only one of the two hosted platforms that
# serves all six models, and the cheapest on the two flagships once prompt caching is
# counted - which on an agent loop is most of the bill. docs/PROVIDERS.md has the
# comparison. `openai` and `qwen` ignore MODEL and read OPENAI_MODEL / QWEN_MODEL.
DEFAULT_PROVIDER = "fireworks"
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
# The same six models are sold by two platforms at two different prices, so a price
# lookup is keyed by (platform, id) and not by id. The two ids in play happen not to
# collide today; Baseten, evaluated and dropped, served GLM 5.3 under the identical
# string `zai-org/GLM-5.3` at a different cached rate, and an id-keyed table billed it at
# Together's. That is not a crash - it is a plausible-looking invoice that is 2x off, so
# the key stays a pair and the next platform cannot reintroduce the bug.
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
	max_images: int | None = None  # 0: unsupported; None: maximum not published/verified
	vision: bool = False
	image_batch: int = 1  # conservative operating cap, distinct from the provider maximum
	streaming_only: bool = False
	image_formats: tuple[str, ...] | None = None  # native accepted formats; None: unverified
	rejected_image_formats: tuple[str, ...] = ()  # '*': all; unlisted formats are unverified


# Effort ladders, named once so the tables below stay readable.
_TOGGLEABLE = ("none", "low", "medium", "high", "max")  # reasoning can be turned off
_ALWAYS_ON = ("low", "high", "max")  # it cannot; "max" is the model's own default
_M = 1_048_576

# Native encodings, before image_view conversion; GIF means a single frame.
# Shared tuples keep identical verified capabilities consistent across exact endpoints.
_WEB_IMAGES = ("jpeg", "png", "webp", "gif")
_RASTER_IMAGES = _WEB_IMAGES + ("bmp", "tiff", "ppm")
_EXTENDED_IMAGES = _RASTER_IMAGES + ("heif", "heic", "avif")
_RAW_IMAGES = ("dng", "cr2", "cr3", "nef", "arw", "raf")

# Image limits/formats checked 2026-09-10; sources and unknowns are in docs/IMAGE.md.
# Local Qwen's launch script explicitly sets --limit-mm-per-prompt image=4.
CATALOG: dict[str, dict[str, Model]] = {
	"fireworks": {
		# Both platforms are pinned to a dated build. `deepseek-v4-pro` also resolves on
		# Fireworks and floats to whatever is current, which would silently change what a
		# stored eval number means.
		"deepseek-v4-pro":  Model("accounts/fireworks/models/deepseek-v4-pro-0813",   1.32, 0.044, 3.96, _TOGGLEABLE, _M, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"deepseek-v4-flash": Model("accounts/fireworks/models/deepseek-v4-flash-0731", 0.22, 0.007, 0.66, _TOGGLEABLE, _M, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"glm-5.3":          Model("accounts/fireworks/models/glm-5p3",          1.40, 0.26,  4.40, _ALWAYS_ON,  _M, 32_768, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"glm-5.3-flash":    Model("accounts/fireworks/models/glm-5p3-flash",    0.15, 0.03,  0.50, _ALWAYS_ON,  _M, 32_768, max_images=30, vision=True, image_batch=8,
		                         image_formats=_EXTENDED_IMAGES, rejected_image_formats=("pdf",)),
		"kimi-k3":          Model("accounts/fireworks/models/kimi-k3",          3.00, 0.30, 15.00, _ALWAYS_ON,  _M, 32_768, max_images=30, vision=True, image_batch=8,
		                         image_formats=_EXTENDED_IMAGES, rejected_image_formats=("pdf",)),
		"qwen-3.7-plus":    Model("accounts/fireworks/models/qwen3p7-plus",     0.40, 0.08,  1.60, _TOGGLEABLE, 262_144, max_images=30, vision=True, image_batch=8,
		                         image_formats=None),
		"qwen-3.8-max":     Model("accounts/fireworks/models/qwen3p8-max",      2.00, 0.25,  6.00, _TOGGLEABLE, _M, max_images=30, vision=True, image_batch=8,
		                         image_formats=_EXTENDED_IMAGES, rejected_image_formats=("pdf",)),
	},
	"together": {
		"deepseek-v4-pro":  Model("deepseek-ai/DeepSeek-V4-Pro-0813",   1.32, 0.13, 3.96, _TOGGLEABLE, _M, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"deepseek-v4-flash": Model("deepseek-ai/DeepSeek-V4-Flash-0731", 0.14, 0.03, 0.28, _TOGGLEABLE, _M, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"glm-5.3":          Model("zai-org/GLM-5.3",                    1.40, 0.26, 4.40, _ALWAYS_ON,  _M, 32_768, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
		"glm-5.3-flash":    Model("zai-org/GLM-5.3-Flash",              0.15, 0.03, 0.50, _ALWAYS_ON,  _M, 32_768, vision=True, image_batch=8,
		                         image_formats=_RASTER_IMAGES, rejected_image_formats=("heif", "heic", "avif", "pdf")),
		"kimi-k3":          Model("moonshotai/Kimi-K3",                 3.00, 0.30, 15.00, _ALWAYS_ON, _M, 32_768, vision=True, image_batch=8,
		                         image_formats=_RASTER_IMAGES, rejected_image_formats=("heif", "heic", "avif", "pdf")),
		# No cached-input rate is published for either Qwen; billed here at the full input
		# rate, which over-states the cost rather than under-stating it.
		"qwen-3.7-plus":    Model("Qwen/Qwen3.7-Plus",                  0.32, 0.32, 1.28, _TOGGLEABLE, 1_000_000, vision=True, image_batch=8, streaming_only=True,
		                         image_formats=_EXTENDED_IMAGES, rejected_image_formats=("pdf",)),
		"qwen-3.8-max":     Model("Qwen/Qwen3.8-2.4T-A95B",             2.00, 0.25, 6.00, ("low", "medium", "xhigh"), _M, max_images=0,
		                         image_formats=(), rejected_image_formats=("*",)),
	},
	# Self-hosted vLLM on the local box: the GPU is already paid for, so per-token is 0.
	"qwen": {
		"qwen3.5-9b": Model(QWEN_MODEL, 0.0, 0.0, 0.0, _TOGGLEABLE, 32_768, max_images=4, vision=True, image_batch=4,
		                         image_formats=_RASTER_IMAGES + ("avif",), rejected_image_formats=("heif", "heic", "pdf")),
	},
	"openai": {
		"gpt-5.4-nano": Model("gpt-5.4-nano", 0.20, 0.02, 1.25, _TOGGLEABLE, 400_000, max_images=1500, vision=True, image_batch=8,
		                         image_formats=_WEB_IMAGES, rejected_image_formats=("bmp", "tiff", "ppm", "heif", "heic", "avif", "pdf") + _RAW_IMAGES),
		"gpt-5-nano":   Model("gpt-5-nano",   0.05, 0.005, 0.40, _TOGGLEABLE, 400_000, max_images=1500, vision=True, image_batch=8,
		                         image_formats=_WEB_IMAGES, rejected_image_formats=("bmp", "tiff", "ppm", "heif", "heic", "avif", "pdf") + _RAW_IMAGES),
		"gpt-5-mini":   Model("gpt-5-mini",   0.25, 0.025, 2.00, _TOGGLEABLE, 400_000, max_images=1500, vision=True, image_batch=8,
		                         image_formats=_WEB_IMAGES, rejected_image_formats=("bmp", "tiff", "ppm", "heif", "heic", "avif", "pdf") + _RAW_IMAGES),
		"gpt-5":        Model("gpt-5",        1.25, 0.125, 10.00, _TOGGLEABLE, 400_000, max_images=1500, vision=True, image_batch=8,
		                         image_formats=_WEB_IMAGES, rejected_image_formats=("bmp", "tiff", "ppm", "heif", "heic", "avif", "pdf") + _RAW_IMAGES),
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
		# .env pins MODEL, so the likely cause is overriding PROVIDER without it. Say so:
		# the alternative is a stack trace that names neither variable.
		known = ", ".join(sorted(CATALOG.get(provider, {}))) or "no models in the catalog"
		raise ValueError(
			f"{provider!r} does not serve {name!r}. Set MODEL= to clear it, or to one of: {known}")
	return name


def model_spec(model_id: str, provider: str) -> Model | None:
	"""The catalog entry for an id already resolved for `provider`, or None if unlisted."""
	for entry in CATALOG.get(provider, {}).values():
		if entry.id == model_id:
			return entry
	return None


def image_parts(messages) -> list[dict]:
	if isinstance(messages, str):
		return []
	return [part for item in messages for part in
	        (item.get("content") if isinstance(item.get("content"), list) else [])
	        if part.get("type") == "input_image"]


def image_limit(model: str, provider: str) -> int:
	spec = model_spec(model, provider)
	if spec is None or not spec.vision:
		raise ValueError(f"image input is unsupported or unverified for {provider}/{model}")
	return spec.max_images if spec.max_images is not None else spec.image_batch


def validate_images(messages, model: str, provider: str) -> None:
	parts = image_parts(messages)
	if not parts:
		return
	limit = image_limit(model, provider)
	if len(parts) > limit:
		raise ValueError(f"{provider}/{model} accepts at most {limit} images per request; got {len(parts)}")
	if provider == "fireworks" and sum(len(part.get("image_url", "")) for part in parts) >= 10_000_000:
		raise ValueError("Fireworks image payload must be below 10 MB")
	if provider == "openai" and len(json.dumps(messages).encode()) > 512_000_000:
		raise ValueError("OpenAI request payload exceeds 512 MB")


def alias_for(model_id: str, provider: str) -> str | None:
	"""The portable name behind a platform's id, or None if the catalog does not know it.

	This is what makes a fallback possible at all: `accounts/fireworks/models/glm-5p3`
	and `zai-org/GLM-5.3` are the same weights under two names, so failing over means
	going back through the alias, never re-sending the id.
	"""
	for alias, entry in CATALOG.get(provider, {}).items():
		if entry.id == model_id:
			return alias
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
	# Calls this Usage covers that the primary platform refused and the fallback served.
	# It rides here rather than in a counter of its own because everything downstream -
	# the tracker, the per-task deltas in evals/harness.py, the merge across 24 shards -
	# already sums Usage, and a fallback nobody can see is a bill nobody can explain.
	fallback_calls: int = 0

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
			fallback_calls=self.fallback_calls + other.fallback_calls,
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
			+ (f" fallback={u.fallback_calls}" if u.fallback_calls else "")
		)


TRACKER = CostTracker()


# ---------------------------------------------------------------------------
# Input message shapes (Responses API `input` items)
# ---------------------------------------------------------------------------

class Message(TypedDict):
	"""A chat turn from the user, system, or assistant."""
	role: Literal["user", "assistant", "system", "developer"]
	content: str | list[dict]


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


def resendable_arguments(arguments: str) -> str:
	"""A tool call's arguments as the model emitted them, unless they are not JSON.

	A call cut off by the output limit is refused by the registry, which is right, but
	echoing its half-written string back into the next request is not: vLLM parses every
	assistant tool call while rendering the chat template and refuses the whole request
	with a 400, and the call sits in history, so every later turn fails the same way. Send
	a JSON object carrying the fragment instead; the model still sees what it cut off."""
	try:
		json.loads(arguments)
		return arguments
	except (TypeError, ValueError):
		return json.dumps({"truncated": arguments})


def resendable(messages: "Messages") -> "Messages":
	"""`messages` with every function_call's arguments made safe to send back."""
	if isinstance(messages, str):
		return messages
	return [{**m, "arguments": resendable_arguments(m["arguments"])}
	        if m.get("type") == "function_call" else m for m in messages]

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
		max_output_tokens: int | None = None,
		thinking: bool | None = None,
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
		max_output_tokens: int | None = None,
		thinking: bool | None = None,
	) -> ModelTurn:
		validate_images(messages, model, "openai")
		kwargs: dict = {
			"model": model,
			"input": resendable(messages),
			"timeout": _env("AGENT_TIMEOUT_S", timeout),
			"reasoning": {"effort": DEFAULT_REASONING_EFFORT},
		}
		if thinking is not None:
			kwargs["reasoning"] = {"effort": ("high" if thinking else
			    "none" if model.startswith("gpt-5.4") else "minimal")}
		if max_output_tokens is not None:
			validate_output_tokens(max_output_tokens)
			kwargs["max_output_tokens"] = max_output_tokens
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


BACKENDS: dict[str, Backend] = {
	"fireworks": Backend(
		"fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_BASE_URL", "FIREWORKS_API_KEY"),
	"together": Backend(
		"together", "https://api.together.xyz/v1", "TOGETHER_BASE_URL", "TOGETHER_API_KEY"),
	"qwen": Backend(
		"qwen", QWEN_BASE_URL, "QWEN_BASE_URL", "QWEN_API_KEY", reasoning_style="chat_template"),
}


# Fireworks reports prompt-cache hits ONLY in this response header. Its body's
# `prompt_tokens_details.cached_tokens` is present and always 0, which is the worst shape
# a discrepancy can take: it looks like an answer. Reading the body alone billed every
# Fireworks token at the uncached rate - measured at 24k tokens of shared prefix, DeepSeek
# V4 Pro reports 24012 of 24013 cached, so the over-charge on a long agent run is not a
# rounding error, it is most of the invoice.
#
# Caching is on by default and needs no key; it simply does not engage on short prefixes,
# and where it kicks in is per-model. Measured against the live API: glm-5.3-flash caches
# in exact 2048-token blocks, so anything under 2048 caches nothing at all, while
# deepseek-v4-pro hits ~100% from 1800 tokens up. This repo's prompts are ~1400-1800
# tokens, so a 0% hit rate on the default model is the design, not a broken header.
CACHED_TOKENS_HEADER = "fireworks-cached-prompt-tokens"


def cached_header(headers) -> int | None:
	"""Cached prompt tokens per the response headers, or None if this platform says
	nothing there and the body should be believed instead."""
	value = headers.get(CACHED_TOKENS_HEADER)
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _parses(arguments: str) -> bool:
	"""Whether a tool call's argument string is complete JSON."""
	try:
		json.loads(arguments)
	except (ValueError, TypeError):
		return False
	return True


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
					"function": {"name": item["name"], "arguments": resendable_arguments(item["arguments"])},
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
				content = item["content"]
				if isinstance(content, list):
					content = [
						{"type": "text", "text": part["text"]} if part["type"] == "input_text" else
						{"type": "image_url", "image_url": {"url": part["image_url"], "detail": part.get("detail", "low")}}
						for part in content
					]
				out.append({"role": role, "content": content})
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

	def _reasoning(self, kwargs: dict, extra_body: dict, spec: Model | None, thinking: bool | None = None) -> None:
		"""Write the backend's own spelling of "think this much" into the request.

		The floor is the model's, not ours: asking GLM 5.3 or Kimi K3 for "none" is a 400,
		so the catalog's first entry is the cheapest thing each will actually accept.
		"""
		effective_thinking = self.thinking if thinking is None else thinking
		if self.backend.reasoning_style == "chat_template":
			extra_body["chat_template_kwargs"] = {"enable_thinking": effective_thinking}
			return
		ladder = spec.reasoning if spec else _TOGGLEABLE
		kwargs["reasoning_effort"] = (self.reasoning_effort if thinking is None else None) or (ladder[-1] if effective_thinking else ladder[0])

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
		max_output_tokens: int | None = None,
		thinking: bool | None = None,
	) -> ModelTurn:
		if max_output_tokens is not None:
			validate_output_tokens(max_output_tokens)
		validate_images(messages, model, self.backend.name)
		spec = model_spec(model, self.backend.name)
		if spec and spec.streaming_only and not stream:
			stream = True
			on_text = on_tool_call = None  # Collect internally for non-streaming callers.
		extra_body: dict = {}
		kwargs: dict = {
			"model": model,
			"messages": self._to_chat_messages(messages),
			"timeout": _env("AGENT_TIMEOUT_S", timeout),
			"temperature": self.temperature,
			"top_p": DEFAULT_TOP_P,
			"seed": self.seed,
			"max_tokens": max_output_tokens if max_output_tokens is not None else (self.max_output_tokens or (spec.max_output if spec else DEFAULT_MAX_OUTPUT_TOKENS)),
		}
		# top_k is not an OpenAI parameter, so extra_body is the only way past the SDK -
		# which validates keyword names against its own signature and raises before the
		# request is ever sent, whatever the server would have accepted. Fireworks
		# documents top_k as top-level and it still has to travel down here.
		extra_body["top_k"] = DEFAULT_TOP_K
		self._reasoning(kwargs, extra_body, spec, thinking)
		if extra_body:
			kwargs["extra_body"] = extra_body
		if tools:
			kwargs["tools"] = self._to_chat_tools(tools)
			kwargs["tool_choice"] = tool_choice or "auto"

		if not stream:
			raw = self.client.chat.completions.with_raw_response.create(**kwargs)
			resp = raw.parse()
			choice = resp.choices[0]
			text = choice.message.content or None
			calls = [
				ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
				for tc in (choice.message.tool_calls or [])
			]
			return self._to_turn(text, calls, resp.usage, choice.finish_reason, model,
			                     cached_header(raw.headers), cap=kwargs["max_tokens"])

		# Streaming: accumulate text + tool-call fragments keyed by index.
		text_parts: list[str] = []
		pending: dict[int, dict] = {}
		usage = None
		finish = None
		raw = self.client.chat.completions.with_raw_response.create(
			stream=True, stream_options={"include_usage": True}, **kwargs)
		for chunk in raw.parse():
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
		return self._to_turn("".join(text_parts) or None, calls, usage, finish, model,
		                     cached_header(raw.headers), cap=kwargs["max_tokens"])

	def _to_turn(self, text, calls, raw_usage, finish_reason, model: str,
	             cached_from_header: int | None = None, cap: int | None = None) -> ModelTurn:
		usage = None
		if raw_usage is not None:
			in_details = getattr(raw_usage, "prompt_tokens_details", None)
			out_details = getattr(raw_usage, "completion_tokens_details", None)
			cached = getattr(in_details, "cached_tokens", 0) or 0
			if cached_from_header is not None:
				cached = cached_from_header
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
		# vLLM's stream says finish_reason="tool_calls" for a call that max_tokens cut off
		# mid-arguments; only its non-streaming reply says "length". The loop's cut-off
		# nudge keys on "incomplete", so trusting the stream left a voice turn rewriting
		# the same half-file until max_steps. The token count is the honest signal: at
		# the cap with no call, or a call whose arguments do not parse, nothing finished.
		if stop == "completed" and cap is not None and usage is not None and usage.output_tokens >= cap:
			if not calls or any(not _parses(c.arguments) for c in calls):
				stop = "incomplete"
		return ModelTurn(text=text, tool_calls=calls or None, usage=usage, stop_reason=stop)


class QwenProvider(ChatProvider):
	"""The self-hosted vLLM box, pinned. Kept as a name because the evals refer to it."""

	def __init__(self, **kw):
		super().__init__(backend="qwen", **kw)


# ---------------------------------------------------------------------------
# Falling over to the second platform
# ---------------------------------------------------------------------------

def can_fail_over(exc: Exception) -> bool:
	"""True when the *platform* would not serve the request; false when the request is
	the problem.

	The distinction is the whole value of the check. A 400 is a malformed request and
	will be just as malformed on the second platform, so failing over on it pays twice
	for the same rejection and hides the bug. A 503, a 429, a timeout, a dead key or a
	model that is not there are all "ask someone else", and are what this exists for.
	"""
	if is_retryable(exc):  # 5xx, timeouts, transport, and RateLimitError
		return True
	# By status rather than by exception class, so this does not quietly depend on the SDK
	# still mapping 429 to RateLimitError: 401/403 a bad or missing key, 404 a model this
	# platform does not serve, 408/429 a request it would not take right now.
	return isinstance(exc, APIStatusError) and exc.status_code in (401, 403, 404, 408, 429)


class FallbackProvider:
	"""The same model on a second platform when the first will not serve it.

	Only the alias travels. `accounts/fireworks/models/glm-5p3` means nothing to Together,
	so the id is resolved back through the catalog and forward again - and if the second
	platform does not serve that model at all, the original error is raised rather than a
	quietly different model being run.

	The fall-over is eager: the first 503 goes to Together rather than sleeping through a
	backoff. `with_retries` still wraps this from the loop, so both platforms failing is
	what gets retried, and the pair is tried up to `DEFAULT_MAX_RETRIES + 1` times.

	Every fall-over is counted into Usage.fallback_calls and narrated on stderr. Silence
	here would be the expensive kind: a mistyped FIREWORKS_API_KEY sends an entire eval to
	the more expensive platform and the only evidence would be the invoice.
	"""

	def __init__(self, primary: ChatProvider, secondary: ChatProvider):
		self.primary, self.secondary = primary, secondary

	def _twin(self, model: str) -> str | None:
		"""The secondary's id for the same weights, or None if it cannot serve them."""
		alias = alias_for(model, self.primary.backend.name)
		entry = CATALOG[self.secondary.backend.name].get(alias) if alias else None
		return entry.id if entry else None

	def generate(self, messages: Messages, model: str, tools: list[dict] | None, **kw) -> ModelTurn:
		if image_parts(messages):
			twin = self._twin(model)
			validate_images(messages, twin, self.secondary.backend.name)
		try:
			return self.primary.generate(messages, model, tools, **kw)
		except Exception as exc:
			twin = self._twin(model)
			if twin is None or not can_fail_over(exc):
				raise
			print(f"[fallback] {self.primary.backend.name} -> {self.secondary.backend.name} "
			      f"({type(exc).__name__}: {exc}); retrying as {twin}", file=sys.stderr, flush=True)
			turn = self.secondary.generate(messages, twin, tools, **kw)
			# Attribute the call before it reaches the tracker. A turn with no usage still
			# gets one, or a fallback that returned nothing would not be counted.
			turn.usage = (turn.usage or Usage())
			turn.usage.fallback_calls += 1
			return turn


# Who covers for whom when the primary will not serve. Fireworks and Together carry the
# same six models, so the pair is free; nothing covers for the local box, because a
# fallback that leaves the machine is a different experiment, not the same one.
DEFAULT_FALLBACK: dict[str, str] = {"fireworks": "together"}


def _chat_provider(name: str, timeout: float) -> ChatProvider:
	backend = BACKENDS[name]
	# vLLM is happy without a key; a hosted platform is not, and finding that out on
	# request 1 beats finding it out on request 200 of an eval shard.
	if name != "qwen" and not os.getenv(backend.api_key_env):
		raise RuntimeError(f"{backend.api_key_env} is not set in the environment or .env")
	return ChatProvider(backend=backend, timeout=timeout)


def make_provider(name: str | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> Provider:
	"""Build the provider named by `name` (or the PROVIDER env var), and its fallback.

	FALLBACK names the second platform; "none" or "" turns it off. Unset, Fireworks pairs
	with Together. Asking for a fallback whose key is missing is an error - but *not*
	having asked, and silently getting none, is only a warning, because the run is still
	the run the operator described.
	"""
	timeout = _env("AGENT_TIMEOUT_S", timeout)
	name = (name or os.getenv("PROVIDER", DEFAULT_PROVIDER)).lower()
	if name in BACKENDS:
		primary = _chat_provider(name, timeout)
		requested = os.getenv("FALLBACK")
		second = (requested if requested is not None else DEFAULT_FALLBACK.get(name, "")).lower()
		if second in ("", "none"):
			return primary
		if second not in BACKENDS or second == name:
			raise ValueError(f"unknown FALLBACK {second!r} (expected one of: {', '.join(BACKENDS)})")
		if requested is None and not os.getenv(BACKENDS[second].api_key_env):
			print(f"[fallback] none: {name} has no cover because "
			      f"{BACKENDS[second].api_key_env} is not set", file=sys.stderr, flush=True)
			return primary
		return FallbackProvider(primary, _chat_provider(second, timeout))
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
# platform - the one whose price makes a 164-task eval a rounding error. On the hosted
# two that is glm-5.3-flash at $0.15/$0.03/$0.50, which put up 8/8 on humaneval for
# $0.006; kimi-k3 is the most capable of the six and twenty times the price.
DEFAULT_MODELS: dict[str, str] = {
	"openai": DEFAULT_MODEL,
	"qwen": QWEN_MODEL,
	"fireworks": "glm-5.3-flash",
	"together": "glm-5.3-flash",
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
	max_retries = _env("AGENT_MAX_RETRIES", max_retries, int)
	for attempt in range(max_retries + 1):
		try:
			return fn()
		except Exception as exc:
			if not is_retryable(exc) or attempt == max_retries:
				raise
			delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)
			time.sleep(delay)
	raise AssertionError("unreachable")


def validate_output_tokens(value: int) -> None:
	if type(value) is not int or not 1 <= value <= 65_536:
		raise ValueError("max_output_tokens must be an integer from 1 to 65536")
