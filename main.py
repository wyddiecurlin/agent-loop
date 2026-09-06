import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Protocol, TypedDict

from dotenv import load_dotenv
from openai import (
	APIConnectionError,
	APIStatusError,
	APITimeoutError,
	OpenAI,
	RateLimitError,
)
import json
import shutil
import subprocess
from pathlib import Path

from tools import DONE_TOOL, REGISTRY, ToolCall, ToolResult


DEFAULT_MODEL = "gpt-5.4-nano"  # fastest OpenAI TTFT (~0.67s) with reasoning off
DEFAULT_REASONING_EFFORT = "none"  # no reasoning tokens before the first output token
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3

# USD per 1M tokens. Verify against the provider's pricing page before relying on these.
PRICING: dict[str, dict[str, float]] = {
	"gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
	"gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
	"gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
	"gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
	# self-hosted (vLLM on the local box) -> no per-token cost
	"qwen3.5-9b": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
}

# Provider selection: PROVIDER=openai (default) | qwen
# qwen = self-hosted Qwen3.5-9B behind vLLM's OpenAI-compatible Chat Completions API.
DEFAULT_PROVIDER = "openai"
QWEN_BASE_URL = "http://localhost:9000/v1"
QWEN_MODEL = "qwen3.5-9b"


# ---------------------------------------------------------------------------
# Token / cost accounting
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


def compute_cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
	"""Return USD cost for one call, or 0.0 if the model has no pricing entry."""
	price = PRICING.get(model)
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

	def record(self, turn: "ModelTurn") -> None:
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
				cost_usd=compute_cost(model, response.usage.input_tokens, cached, response.usage.output_tokens),
			)

		return ModelTurn(
			text=response.output_text or None,
			tool_calls=tool_calls or None,
			usage=usage,
			stop_reason=response.status,
		)


class QwenProvider:
	"""Self-hosted Qwen (vLLM) via the OpenAI Chat Completions API.

	The rest of the loop speaks the Responses API shapes (Message / FunctionCallItem /
	FunctionCallOutputItem, tools as {"type","name","parameters"}), so this provider
	translates both directions and returns the same ModelTurn as OpenAIProvider.

	Env: QWEN_BASE_URL (default http://localhost:9000/v1), QWEN_API_KEY (optional),
	QWEN_THINKING=1 to let the model reason before answering (slower, more tokens).
	"""

	def __init__(
		self,
		client: OpenAI | None = None,
		timeout: float = DEFAULT_TIMEOUT_S,
		base_url: str | None = None,
		api_key: str | None = None,
		thinking: bool | None = None,
	):
		self.client = client or OpenAI(
			base_url=base_url or os.getenv("QWEN_BASE_URL", QWEN_BASE_URL),
			api_key=api_key or os.getenv("QWEN_API_KEY", "EMPTY"),
			timeout=timeout,
			max_retries=0,
		)
		self.thinking = thinking if thinking is not None else os.getenv("QWEN_THINKING", "0") == "1"

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
			"messages": self._to_chat_messages(messages),
			"timeout": timeout,
			"extra_body": {"chat_template_kwargs": {"enable_thinking": self.thinking}},
		}
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

	@staticmethod
	def _to_turn(text, calls, raw_usage, finish_reason, model: str) -> ModelTurn:
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
				cost_usd=compute_cost(model, raw_usage.prompt_tokens, cached, raw_usage.completion_tokens),
			)
		# Map chat finish_reason onto the Responses-style status the loop expects.
		stop = {"stop": "completed", "tool_calls": "completed", "length": "incomplete"}.get(finish_reason, finish_reason)
		return ModelTurn(text=text, tool_calls=calls or None, usage=usage, stop_reason=stop)


def make_provider(name: str | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> Provider:
	"""Build the provider named by `name` (or the PROVIDER env var)."""
	name = (name or os.getenv("PROVIDER", DEFAULT_PROVIDER)).lower()
	if name == "qwen":
		return QwenProvider(timeout=timeout)
	if name == "openai":
		if not os.getenv("OPENAI_API_KEY"):
			raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
		return OpenAIProvider(timeout=timeout)
	raise ValueError(f"unknown PROVIDER {name!r} (expected 'openai' or 'qwen')")


def default_model(provider_name: str | None = None) -> str:
	"""Model to use when the caller didn't pick one: depends on the active provider."""
	name = (provider_name or os.getenv("PROVIDER", DEFAULT_PROVIDER)).lower()
	if name == "qwen":
		return os.getenv("QWEN_MODEL", QWEN_MODEL)
	return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

'''
given list of messages and available tools, generate a response from the model
'''
def generate(
	messages: Messages | None = None,
	model: str | None = None,
	tools: list[dict] | None = None,
	*,
	provider: Provider | None = None,
	stream: bool = False,
	on_text: OnText | None = None,
	on_tool_call: OnToolCall | None = None,
	timeout: float = DEFAULT_TIMEOUT_S,
	max_retries: int = DEFAULT_MAX_RETRIES,
	tracker: CostTracker | None = None,
	tool_choice: str | None = None,
) -> ModelTurn:
	"""Generate a ModelTurn for the given messages, with retries, timeout, and cost tracking."""
	load_dotenv()

	if tools is None:
		tools = REGISTRY.schema()
	if provider is None:
		provider = make_provider(timeout=timeout)
	if model is None:
		model = default_model()

	turn = with_retries(
		lambda: provider.generate(
			messages, model, tools, stream=stream, on_text=on_text, on_tool_call=on_tool_call,
			timeout=timeout, tool_choice=tool_choice,
		),
		max_retries=max_retries,
	)
	(tracker or TRACKER).record(turn)
	return turn

def stream(t: str) -> str:
	print(t, end='', flush=True)

def stream_tool_calls(t: str, kind: Literal['function_call', 'function_args']):
	if kind == 'function_call':
		print()
		print(f"tool calling: {t}")
	else:
		print(t, end='', flush=True)


SYSTEM_PROMPT = '''
	You are an agent that completes tasks by invoking tools according to user's requests.
	RULES:
	- Every turn must call at least one tool; there is no way to reply with plain text.
	- If you announce that you are going to call a tool, call it in the same turn.
	- The task ends only when you call `done`, so call it as soon as you have what you need.
	- Call `done` for either outcome:
	  - it worked: `answer` is the complete reply for the user.
	  - it cannot be done: call `done` anyway and use `answer` to say what you tried and
	    why it failed. Never keep calling tools hoping the problem fixes itself.
	- `answer` must stand on its own: give the actual values, file names, and command output
	  you found rather than referring back to earlier steps.
'''


def agent_loop(prompt, max_steps: int = 20) -> list:
	"""Run tools until the model calls `done`.

	tool_choice="required" makes every turn carry at least one tool call, so `done` is the
	only exit and the loop never has to guess whether a plain-text reply meant "finished".
	"""
	messages: list[InputItem] = [
		Message(role='system', content=SYSTEM_PROMPT),
		Message(role='user', content=prompt),
	]

	for _ in range(max_steps):
		print(f"[LOG] loop start 1 message length {len(messages)}")
		turn = generate(
			messages=messages,
			tools=REGISTRY.schema(),
			tool_choice="required",
			stream=True,
			on_text=stream,
			on_tool_call=stream_tool_calls,
		)
		print(json.dumps(asdict(turn), indent=2))
		print()

		# Any preamble text, then the calls themselves, so the model can see what it asked for.
		if turn.text:
			messages.append(Message(role='assistant', content=turn.text))
		for call in turn.tool_calls or []:
			messages.append(FunctionCallItem(
				type='function_call',
				call_id=str(call.id),
				name=call.name,
				arguments=call.arguments,
			))

		for call in turn.tool_calls or []:
			result: ToolResult = REGISTRY.execute(call)
			print(f"[LOG]   {result.output[:300]}")
			messages.append(FunctionCallOutputItem(
				type='function_call_output',
				call_id=str(call.id),
				output=result.to_model_output(),
			))
			# `done` hands back the final answer as its output. A malformed call fails like
			# any other tool, so the model reads the error and gets another turn.
			if call.name == DONE_TOOL and result.ok:
				messages.append(Message(role='assistant', content=result.output))
				return messages

	print("[LOG]   hit max_steps")
	return messages


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

SANDBOX = Path("sandbox")

SANDBOX_FILES = {
	"notes.txt": "The answer is 42\nsecond line\nthird line: banana\n",
	"app.py": (
		"def greet(name: str) -> str:\n"
		"    return f\"Hello, {name}!\"\n"
		"\n"
		"\n"
		"if __name__ == \"__main__\":\n"
		"    print(greet(\"World\"))\n"
	),
	"data/config.json": '{"version": "1.2.3", "debug": false}\n',
}


def setup_sandbox() -> None:
	"""Fresh sandbox/ tree with known contents so the fs/shell tests are deterministic."""
	shutil.rmtree(SANDBOX, ignore_errors=True)
	for rel, content in SANDBOX_FILES.items():
		path = SANDBOX / rel
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(content)


def teardown_sandbox() -> None:
	shutil.rmtree(SANDBOX, ignore_errors=True)


def final_text(messages: list[InputItem] | None) -> str:
	"""Last assistant message in the transcript, or '' if the loop hit its cap."""
	if not messages:
		return ""
	for item in reversed(messages):
		if item.get("role") == "assistant":
			return item["content"]
	return ""


def contains_number(text: str, n: int) -> str:
	"""True if `n` appears in `text`, ignoring thousands separators like 54,702 or LaTeX 54{,}702."""
	normalized = re.sub(r"(?<=\d)(,|\{,\}|\s)(?=\d{3})", "", text)
	return re.search(rf"(?<!\d){n}(?!\d)", normalized) is not None


def check(label: str, passed: bool, detail: str = "") -> bool:
	print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
	return passed


def run_case(n: int, prompt: str) -> str:
	print(f"\n{'=' * 70}\nTest case {n}: {prompt}\n{'=' * 70}")
	messages = agent_loop(prompt=prompt)
	return final_text(messages)


def main() -> None:
	results: list[bool] = []

	# --- arithmetic / date tools -------------------------------------------------

	answer = run_case(1, "Get today's date and compute the multiplication of month, day, and year.")
	from datetime import datetime
	today = datetime.now()
	expected = today.month * today.day * today.year
	results.append(check("case 1: product of month*day*year", contains_number(answer, expected), f"expected {expected}"))

	answer = run_case(
		2,
		"Multiply month, day, and year of today's date, and do the same for the founding date of "
		"China's communist party 1949.10.1, and substract the two results.",
	)
	expected = today.month * today.day * today.year - 10 * 1 * 1949
	results.append(check("case 2: difference of the two products", contains_number(answer, expected), f"expected {expected}"))

	# --- filesystem / shell tools --------------------------------------------------

	setup_sandbox()
	try:
		# 3. fs_list
		answer = run_case(3, "List every file under the sandbox directory, recursively, and tell me how many files there are.")
		results.append(check(
			"case 3: fs_list finds all 3 files",
			"3" in answer and "notes.txt" in answer and "app.py" in answer and "config.json" in answer,
		))

		# 4. fs_read
		answer = run_case(4, "Read sandbox/data/config.json and tell me the version number it contains.")
		results.append(check("case 4: fs_read reports version", "1.2.3" in answer))

		# 5. fs_search
		answer = run_case(5, "Search the sandbox directory for the word 'banana'. Tell me the file name and the line number it appears on.")
		results.append(check("case 5: fs_search locates banana", "notes.txt" in answer and "3" in answer))

		# 6. fs_patch
		answer = run_case(6, "In sandbox/app.py, change the greeting word 'Hello' to 'Howdy'. Do not change anything else.")
		app_src = (SANDBOX / "app.py").read_text()
		results.append(check(
			"case 6: fs_patch edits app.py in place",
			"Howdy" in app_src and "Hello" not in app_src and "greet(" in app_src,
		))

		# 7. shell_run
		expected_out = subprocess.run(["python3", str(SANDBOX / "app.py")], capture_output=True, text=True).stdout.strip()
		answer = run_case(7, "Run the shell command `python3 sandbox/app.py` and tell me exactly what it printed.")
		results.append(check("case 7: shell_run captures stdout", expected_out in answer, f"expected {expected_out!r}"))

		# 8. combined: list + read/shell + create file + read back
		answer = run_case(
			8,
			"Create a new file sandbox/summary.md containing a markdown bullet list of every file in the sandbox "
			"directory (recursively) with its line count, for example '- notes.txt: 3 lines'. "
			"Then read the file back and confirm its contents.",
		)
		summary = SANDBOX / "summary.md"
		body = summary.read_text() if summary.exists() else ""
		results.append(check(
			"case 8: summary.md created with all files",
			summary.exists() and all(name in body for name in ("notes.txt", "app.py", "config.json")),
			f"exists={summary.exists()}",
		))
	finally:
		teardown_sandbox()

	print(f"\n[RESULT] {sum(results)}/{len(results)} test cases passed")
	print(f"[USAGE] {TRACKER.summary()}")


if __name__ == "__main__":
	main()
