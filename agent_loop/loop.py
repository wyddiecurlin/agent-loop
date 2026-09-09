"""The agent loop.

Every tool the model can reach is bound to the Runtime handed in, so the loop cannot
act outside the sandbox it was given - there is no other registry to reach for.

All narration goes to stderr. stdout belongs to the result JSON (see __main__.py).
"""

import json
import sys
from collections import Counter
from datetime import datetime
from dataclasses import asdict, dataclass
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

from .context import ContextBudget
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
	max_output_tokens: int | None = None,
	thinking: bool | None = None,
) -> ModelTurn:
	"""Generate a ModelTurn for the given messages, with retries, timeout, and cost tracking."""
	if tools is None:
		raise ValueError("tools is required: build it with build_registry(runtime).schema()")
	if provider is None:
		provider = make_provider(timeout=timeout)
	if model is None:
		model = default_model()

	overrides = {}
	if max_output_tokens is not None:
		overrides["max_output_tokens"] = max_output_tokens
	if thinking is not None:
		overrides["thinking"] = thinking
	turn = with_retries(
		lambda: provider.generate(
			messages, model, tools, stream=stream, on_text=on_text, on_tool_call=on_tool_call,
			timeout=timeout, tool_choice=tool_choice, **overrides,
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


@dataclass
class AgentRun:
	"""What one agent_loop call did, as a struct rather than a bare message list.

	The list alone could not answer "why did this stop". A run that exhausted its budget
	and one that called `done` with an empty answer both ended with no answer, and a run
	whose provider raised did not come back at all - so a caller measuring failures could
	not tell a scaffold problem from a wrong answer. `stop_reason` is that distinction.

	`steps` counts model turns. len(messages) grows about three items per turn, so
	reporting it as "steps" reads as four times more work than actually happened.
	"""
	messages: list[InputItem]
	stop_reason: Literal["done", "max_steps", "error", "stalled"]
	steps: int
	error: str = ""  # set only when stop_reason == "error"
	history_cleared: bool = False

	@property
	def ok(self) -> bool:
		return self.stop_reason == "done"

	@property
	def answer(self) -> str:
		return final_text(self.messages)

	def tool_histogram(self) -> dict[str, int]:
		"""How many times each tool was called, most used first."""
		names = Counter(m["name"] for m in self.messages if m.get("type") == "function_call")
		return dict(names.most_common())

	def repeated_calls(self) -> int:
		"""Calls that repeat an earlier (name, arguments) pair exactly.

		A loop that is stuck usually is not varying its input. This separates "the task
		needed 30 turns" from "the model asked the same thing 30 times", which need
		opposite fixes: a bigger budget, or an escape from the rut.
		"""
		seen: set[tuple[str, str]] = set()
		repeats = 0
		for m in self.messages:
			if m.get("type") != "function_call":
				continue
			key = (m["name"], m["arguments"])
			if key in seen:
				repeats += 1
			else:
				seen.add(key)
		return repeats



MAX_STALLED_TURNS = 2
STALL_NUDGE = (
	"Your last response contained no tool call. Every turn must call exactly one tool. "
	"Do not explain your reasoning first - call the tool now, and if the task is already "
	"finished or cannot be done, call `done`."
)
TRUNCATION_NUDGE = (
	"Your last output was cut off by the output limit ({limit}) before the tool call was "
	"complete, so it did nothing. Do not write it again. Anything long - code, a document, "
	"a list - goes into a file with the file tools, and programs are run with the shell; "
	"`done` is only for a short report of what you did and where it is."
)


PERSONA = '''
	CHARACTER:
	You are a small, capable robot named Mimo with a big personality: playful, curious, quick, and
	completely straight with people.
	- Honest and pragmatic. Say what is true, what you found, and what you could not do.
	  No hedging, no padding, no bullshit.
	- Zero flattery. Never praise the user's question, idea or taste ("great question",
	  "what a cool idea"), never gush. If something is good, say why in one plain clause;
	  if it is not, say so, kindly.
	- Dry humor, lightly and rarely, never at the user's expense and never in place of an
	  answer. Skip the jokes when the person is stressed or in a hurry.
	- Talk like an old friend who happens to be good at this: relaxed, direct, familiar,
	  respectful. Contractions, plain words, no corporate tone, no lectures.
	- No exclamation-mark enthusiasm, no emoji, no written laughter (haha, lol, 哈哈) and
	  no stage directions.
	- Short by default: the answer first, the reasons after, and only the ones that matter.
'''

SYSTEM_PROMPT = '''
	You are an agent that completes tasks by invoking tools according to user's requests.
''' + PERSONA + '''
	RULES:
	- Use web search and fetch for user's reqeusts relating to recent news, movies, etc
	- Every turn must call at least one tool; there is no way to reply with plain text.
	- If you announce that you are going to call a tool, call it in the same turn.
	- The task ends only when you call `done`, so call it as soon as you have what you need.
	- Call `done` for either outcome:
	  - it worked: `answer` is the complete reply for the user.
	  - it cannot be done: call `done` anyway and use `answer` to say what you tried and
	    why it failed. Never keep calling tools hoping the problem fixes itself.
	- `answer` must stand on its own: if the task requires modifying outside state or files,
		state what you have done. Otherwise, state the answer concisely. 
'''


def stamp(system_prompt: str, now: datetime | None = None) -> str:
	"""`system_prompt` with the current date and time on the end.

	There is no clock tool: the model reads the time here, and the system message is
	rebuilt with a fresh stamp at the start of every agent_loop call, so a conversation
	that lasts an hour does not keep believing it is still the first minute. The zone is
	named because the container's clock is UTC unless TZ is set (run.sh passes .env
	through; voice/bridge.py takes the host's zone from the client).
	"""
	now = now or datetime.now().astimezone()
	when = f"{now:%A}, {now.day} {now:%B} {now.year}, {now.hour:02d}:{now.minute:02d} ({now.tzname() or 'UTC'})"
	return (f"{system_prompt.rstrip()}\n\tRight now it is {when}. Use this for anything that "
	        f"depends on the date or the time.\n")


def agent_loop(
	prompt,
	runtime: DockerRuntime,
	max_steps: int = 120,
	tools: Iterable[str] | None = None,
	system_prompt: str = SYSTEM_PROMPT,
	history: list[InputItem] | None = None,
	verbose: bool = True,
	max_output_tokens: int | None = None,
	thinking: bool | None = None,
	context_budget: ContextBudget | None = None,
) -> AgentRun:
	"""Run tools until the model calls `done`.

	To work for smaller self-hosted models, we set tool_choice="required" to make every turn 
	carry at least one tool call, so `done` is the only exit. 

	`history` is a previous run's `messages`: pass it to continue that conversation with a
	new user prompt instead of starting from the system prompt.
	`max_output_tokens` and `thinking` override only this run's model requests.
	`context_budget` may clear previous turns while preserving this run's tool work.

	`verbose=False` keeps only the tool names on stderr: no step counter, no streamed
	arguments, no turn dump, no tool output. For talking to it, not for debugging it.
	"""
	trace = log if verbose else (lambda *_: None)
	on_tool_call = stream_tool_calls if verbose else (lambda t, kind: kind == "function_call" and log(f"  [{t}]"))
	registry = build_registry(runtime, allow=tools)
	system = Message(role='system', content=stamp(system_prompt))
	messages: list[InputItem] = list(history) if history else [system]
	if history and messages[0].get("role") == "system":
		messages[0] = system  # the clock moved on since the conversation started
	current_start = len(messages)
	messages.append(Message(role='user', content=prompt))
	history_cleared = False

	def finish_run(reason, steps, error=""):
		return AgentRun(messages, reason, steps, error, history_cleared)


	stalled = 0
	for step in range(1, max_steps + 1):
		trace(f"[LOG] step {step}/{max_steps} ({len(messages)} items in context)")
		try:
			if context_budget is not None:
				if context_budget.prepare(messages, registry.schema(), current_start, max_output_tokens):
					current_start = 1
					history_cleared = True
			turn = generate(
				messages=messages,
				tools=registry.schema(),
				tool_choice="required",
				stream=True,
				on_text=stream,
				on_tool_call=on_tool_call,
				**({"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}),
				**({"thinking": thinking} if thinking is not None else {}),
			)
			if context_budget is not None:
				context_budget.observe(messages, registry.schema(), turn.usage)
		except Exception as exc:  # noqa: BLE001
			log(f"[ERROR] generate failed at step {step}: {type(exc).__name__}: {exc}")
			return finish_run("error", step - 1, f"{type(exc).__name__}: {exc}")
		trace(json.dumps(asdict(turn), indent=2))

		# potentially stalled agent
		if not turn.text and not turn.tool_calls:
			stalled += 1
			log(f"[LOG] stalled turn {stalled}/{MAX_STALLED_TURNS} (stop_reason={turn.stop_reason}, "
			    f"{turn.usage.output_tokens if turn.usage else 0} output tokens)")
			if stalled >= MAX_STALLED_TURNS:
				return finish_run("stalled", step,
				                f"{stalled} consecutive turns produced no tool call")
			messages.append(Message(role='user', content=STALL_NUDGE))
			continue
		stalled = 0

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
			trace(f"[LOG] {result.output[:300]}")
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
			return finish_run("done", step)
		if turn.stop_reason == "incomplete":
			# The output limit cut the model off mid-call. The registry has refused the
			# half-written arguments; left there, the model writes the same thing again
			# and is cut off again. Say what happened and where long output belongs.
			limit = f"{max_output_tokens} tokens" if max_output_tokens else "the output token limit"
			log(f"[LOG] output cut off at {limit}")
			messages.append(Message(role='user', content=TRUNCATION_NUDGE.format(limit=limit)))

	log(f"[LOG] hit max_steps {max_steps}")
	return finish_run("max_steps", max_steps)


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
