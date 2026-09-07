"""The agent loop.

Every tool the model can reach is bound to the Runtime handed in, so the loop cannot
act outside the sandbox it was given - there is no other registry to reach for.

All narration goes to stderr. stdout belongs to the result JSON (see __main__.py).
"""

import json
import sys
from dataclasses import asdict
from typing import Iterable, Literal

from .providers import (
	DEFAULT_MAX_RETRIES,
	DEFAULT_TIMEOUT_S,
	FunctionCallItem,
	FunctionCallOutputItem,
	InputItem,
	Message,
	Messages,
	ModelTurn,
	OnText,
	OnToolCall,
	Provider,
	CostTracker,
	TRACKER,
	default_model,
	make_provider,
	with_retries,
)

from .runtime import DockerRuntime
from .tools import DONE_TOOL, ToolResult, build_registry



def log(*args) -> None:
	"""Everything the loop narrates goes to stderr, so stdout stays a clean result."""
	print(*args, file=sys.stderr, flush=True)


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
	if tools is None:
		raise ValueError("tools is required: build it with build_registry(runtime).schema()")
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


def stream(t: str) -> None:
	print(t, end="", flush=True, file=sys.stderr)


def stream_tool_calls(t: str, kind: Literal["function_call", "function_args"]) -> None:
	if kind == "function_call":
		log()
		log(f"tool calling: {t}")
	else:
		print(t, end="", flush=True, file=sys.stderr)


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


def agent_loop(
	prompt,
	runtime: DockerRuntime,
	max_steps: int = 120,
	tools: Iterable[str] | None = None,
	system_prompt: str = SYSTEM_PROMPT,
) -> list:
	"""Run tools until the model calls `done`.

	tool_choice="required" makes every turn carry at least one tool call, so `done` is the
	only exit and the loop never has to guess whether a plain-text reply meant "finished".

	Every tool the model can reach is bound to `runtime`, so the loop cannot act outside
	the sandbox it was handed - there is no other registry to reach for. `tools` narrows
	that set further; see build_registry.
	"""
	registry = build_registry(runtime, allow=tools)
	messages: list[InputItem] = [
		Message(role='system', content=system_prompt),
		Message(role='user', content=prompt),
	]

	for _ in range(max_steps):
		log(f"[LOG] step {len(messages)} items in context")
		turn = generate(
			messages=messages,
			tools=registry.schema(),
			tool_choice="required",
			stream=True,
			on_text=stream,
			on_tool_call=stream_tool_calls,
		)
		log(json.dumps(asdict(turn), indent=2))

		if turn.text:
			messages.append(Message(role='assistant', content=turn.text))
		for call in turn.tool_calls or []:
			messages.append(FunctionCallItem(
				type='function_call',
				call_id=str(call.id),
				name=call.name,
				arguments=call.arguments,
			))

		answer = None
		for call in turn.tool_calls or []:
			result: ToolResult = registry.execute(call)
			log(f"[LOG]   {result.output[:300]}")
			messages.append(FunctionCallOutputItem(
				type='function_call_output',
				call_id=str(call.id),
				output=result.to_model_output(),
			))
			# `done` hands back the final answer as its output. A malformed call fails like
			# any other tool, so the model reads the error and gets another turn. Every
			# call in the turn still runs, so no function_call is left without an output.
			if call.name == DONE_TOOL and result.ok:
				answer = result.output
		if answer is not None:
			messages.append(Message(role='assistant', content=answer))
			return messages

	log(f"[LOG] hit max_steps {max_steps}")
	return messages


def final_text(messages: list[InputItem] | None) -> str:
	"""The answer `done` returned, or '' if the loop ran out of steps without one.

	Intermediate assistant narration does not count. The loop appends an assistant
	message directly after a successful `done` output and nowhere else after a tool
	output, so that pair - and only that pair - is the signature of a finished task.
	"""
	if not messages or len(messages) < 2:
		return ""
	last, prev = messages[-1], messages[-2]
	names = {m["call_id"]: m["name"] for m in messages if m.get("type") == "function_call"}
	if last.get("role") == "assistant" and prev.get("type") == "function_call_output" \
	   and names.get(prev["call_id"]) == DONE_TOOL:
		return last["content"]
	return ""
