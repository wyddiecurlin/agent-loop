import os
import random
import time
from dataclasses import dataclass, field
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


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3

# USD per 1M tokens. Verify against the provider's pricing page before relying on these.
PRICING: dict[str, dict[str, float]] = {
	"gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
	"gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
	"gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}


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
class ToolCall:
	id: str  # call id; echoed back to the model alongside the tool's result
	name: str  # tool name, key into TOOL_FUNCTIONS
	arguments: str  # JSON-encoded arguments; json.loads() before calling the tool


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
	) -> ModelTurn:
		kwargs: dict = {"model": model, "input": messages, "timeout": timeout}
		if tools:
			kwargs["tools"] = tools

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


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


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

# ---------------------------------------------------------------------------
# Tool schemas (Responses API function-tool format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
	{
		"type": "function",
		"name": "multiply",
		"description": "Use this for all multiplication. Never compute products yourself.",
		"parameters": {
			"type": "object",
			"properties": {
				"a": {"type": "number", "description": "First factor."},
				"b": {"type": "number", "description": "Second factor."},
				"c": {"type": "number", "description": "Optional third factor."},
			},
			"required": ["a", "b"],
		},
	},
	{
		"type": "function",
		"name": "get_today_date",
		"description": "Return today's date as month, day, and year.",
		"parameters": {"type": "object", "properties": {}},
	},
	{
		"type": "function",
		"name": "substract",
		"description": "Use this for all substraction. Never compute yourself.",
		"parameters": {
			"type": "object",
			"properties": {
				"a": {"type": "integer", "description": "Minuend."},
				"b": {"type": "integer", "description": "Subtrahend."},
			},
			"required": ["a", "b"],
		},
	},
]


'''
given list of messages and available tools, generate a response from the model
'''
def generate(
	messages: Messages | None = None,
	model: str = DEFAULT_MODEL,
	tools: list[dict] | None = TOOLS,
	*,
	provider: Provider | None = None,
	stream: bool = False,
	on_text: OnText | None = None,
	on_tool_call: OnToolCall | None = None,
	timeout: float = DEFAULT_TIMEOUT_S,
	max_retries: int = DEFAULT_MAX_RETRIES,
	tracker: CostTracker | None = None,
) -> ModelTurn:
	"""Generate a ModelTurn for the given messages, with retries, timeout, and cost tracking."""
	load_dotenv()

	if provider is None:
		if not os.getenv("OPENAI_API_KEY"):
			raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
		provider = OpenAIProvider(timeout=timeout)

	turn = with_retries(
		lambda: provider.generate(
			messages, model, tools, stream=stream, on_text=on_text, on_tool_call=on_tool_call, timeout=timeout
		),
		max_retries=max_retries,
	)
	(tracker or TRACKER).record(turn)
	return turn

def multiply(**params) -> int:
	res = 1
	for k, v in params.items():
		res *= v

	return res

def get_today_date() -> dict:
	from datetime import datetime

	today = datetime.now()
	return {"month": today.month, "day": today.day, "year": today.year}
	
def substract(a: int, b: int) -> int:
	return a - b


# tool name -> python callable, for dispatching tool calls in the agent loop
TOOL_FUNCTIONS: dict[str, Callable] = {
	"multiply": multiply,
	"get_today_date": get_today_date,
	"substract": substract,
}

def stream(t: str) -> str:
	print(t, end='', flush=True)

def stream_tool_calls(t: str, kind: Literal['function_call', 'function_args']):
	if kind == 'function_call':
		print(f"tool calling: {t}")
	else:
		print(t, end='', flush=True)


def agent_loop(prompt) -> None:
	messages = [
		Message(role='system', content='You are a helpful assistant'),
		Message(role='user', content=prompt)
	]
	while len(messages)<20:
		print(f"[LOG] loop start 1 message length {len(messages)}")
		turn = generate(
			messages=messages,
			stream=True,
			on_text=stream,
			on_tool_call=stream_tool_calls,
			tools=TOOLS	
		)
		print()
		# state 2: agent receives current state, emits tool call
		if turn.tool_calls:
			print(f"[LOG] turn result {turn.text}, {turn.tool_calls}")
			if turn.text:
				messages.append(Message(role='assistant', content=turn.text))

			for tool_call in turn.tool_calls:
				result = TOOL_FUNCTIONS[tool_call.name](**json.loads(tool_call.arguments))
				messages.append(FunctionCallItem(type='function_call', name=tool_call.name, arguments=tool_call.arguments, call_id=str(tool_call.id)))
				print(f"[LOG] calling {tool_call.name} with {tool_call.arguments}, result {result}")
				messages.append(FunctionCallOutputItem(type='function_call_output', output=str(result), call_id=str(tool_call.id)))

			continue
		
		else:
			if turn.text:
				messages.append(Message(role='assistant', content=turn.text))
			return messages
			 
			



def main() -> None:
	# test case 1: get today's date and compute the multiplication of month and day and year

	print("Test case 1: Get today's date and compute the multiplication of month, day, and year.")
	prompt = "Get today's date and compute the multiplication of month, day, and year."
	res = agent_loop(prompt=prompt)
	# print(f"expected output: 2026 * 9 * 2 = 36468, actual output: {json.dumps(res, indent=2)}")


	# test case 2: get today's date and compute the multiplication of month and day and year

	print("Test case 2: Multiply month, day, and year of today's date, and do the same for the founding date of China's communist party 1949.10.1, and substract the two results.")
	prompt = "Multiply month, day, and year of today's date, and do the same for the founding date of China's communist party 1949.10.1, and substract the two results."
	res = agent_loop(prompt=prompt)
	# print(f"expected output: 16978, actual output: {json.dumps(res, indent=2)}")

	print(f"\n[USAGE] {TRACKER.summary()}")


if __name__ == "__main__":
	main()
