"""Run one task, then grade it somewhere the agent could never reach.

The score is an exit code, never something the agent said. `final_text()` is recorded
for debugging and is not consulted.

How grading stays honest, in the order the defences apply:

 1. The grader is not in the workspace while the agent runs. It exists only as a string
	in this process. There is no file to patch, so `fs_patch` and `shell_run` have
	nothing to aim at.
 2. When the agent stops, exactly one file is lifted out - solution.py - and then
	`runtime.reset()` deletes everything else it created. A sitecustomize.py, a
	conftest.py, a .pth file, a shadowed `os` module: all gone before grading starts.
 3. The grader runs in a directory that contains only that file and itself.
 4. Passing needs exit 0 AND a per task random token on stderr. A solution that calls
	`sys.exit(0)` at import exits 0 having asserted nothing; it cannot print a token it
	has never seen.
 5. The agent's tools are narrowed (see EVAL_TOOLS) to the ones the task needs.

Defence 1 is the load-bearing one. The rest are there so that no single mistake in it
is enough to move a number.
"""

import time
from dataclasses import dataclass, field

from agent_loop.loop import agent_loop, log
from agent_loop.providers import TRACKER, Usage
from agent_loop.runtime import DockerRuntime

from .datasets import SOLUTION, Task

# Where the grader is assembled, after the workspace has been wiped. Never written while
# the agent is running.
GRADE_DIR = "grade"
GRADE_FILE = "grade.py"
GRADE_TIMEOUT_S = 20.0

# What the agent may call during an eval.
#
# `fs_patch` is out: the task is to write one file from scratch, so it is redundant with
# `fs_write`, and a tool that can rewrite arbitrary text on disk has no job here.
#
# `shell_run` is in, deliberately. Running your own code and reading the traceback is the
# agentic part of this benchmark - remove it and what is left is one-shot code generation
# measured through an agent loop, which is a different and less interesting number. It is
# safe to keep precisely because grading happens after the workspace is destroyed. Use
# --no-shell to score the locked-down variant.
EVAL_TOOLS = ["fs_read", "fs_write", "fs_list", "shell_run"]
EVAL_TOOLS_NO_SHELL = ["fs_read", "fs_write", "fs_list"]

# Enough to write a file, run it, read the error, and fix it a few times. High enough not
# to bind on a real attempt, low enough that a loop that has lost the plot stops.
MAX_STEPS = 60


@dataclass
class Result:
	task_id: str
	passed: bool
	reason: str  # "" when passed; otherwise why it did not
	steps: int  # model turns, not message-list length
	duration_s: float
	usage: Usage = field(default_factory=Usage)
	answer: str = ""  # recorded for debugging; never consulted for scoring
	# Why the loop stopped, independent of whether the answer was right. A task can fail
	# the grader having finished cleanly (wrong code) or pass it having run out of budget
	# (the file was already correct), so this cannot be inferred from `passed`.
	stop_reason: str = "done"
	# What the model actually did with its turns. A budget that binds and a model that is
	# stuck look identical in the score and completely different here.
	tools_used: dict[str, int] = field(default_factory=dict)
	repeated_calls: int = 0


def grade(runtime: DockerRuntime, task: Task, token: str) -> tuple[bool, str]:
	"""Score the workspace as it stands. Destroys it in the process.

	Returns (passed, reason). See the module docstring for why each step is here.
	"""
	try:
		solution = runtime.read_text(SOLUTION)
	except (FileNotFoundError, PermissionError, IsADirectoryError) as exc:
		return False, f"no {SOLUTION}: {type(exc).__name__}"

	if not solution.strip():
		return False, f"{SOLUTION} is empty"

	# Everything the agent did, except the one file above, stops existing here.
	runtime.reset()
	runtime.put({
		f"{GRADE_DIR}/{SOLUTION}": solution,
		f"{GRADE_DIR}/{GRADE_FILE}": task.grader,
	})

	r = runtime.run(f"python3 {GRADE_FILE}", cwd=GRADE_DIR, timeout_s=GRADE_TIMEOUT_S)

	if r.timed_out:
		return False, f"grader timed out after {GRADE_TIMEOUT_S}s"
	if token not in r.stderr:
		if r.exit_code == 0:
			# Exit 0 without the token: the asserts did not run to completion.
			return False, "grader exited 0 without reaching the end (asserts did not run)"
		tail = (r.stderr.strip().splitlines() or ["no stderr"])[-1]
		return False, f"tests failed (exit {r.exit_code}): {tail[:200]}"
	return True, ""


def run_task(
	runtime: DockerRuntime,
	task: Task,
	token: str,
	*,
	tools: list[str],
	max_steps: int = MAX_STEPS,
	canonical: bool = False,
) -> Result:
	"""One task, end to end: clean workspace -> seed -> agent -> grade.

	With canonical=True the reference solution is written instead of running a model.
	That scores the harness rather than the agent, and it should come out near 100%: any
	shortfall is a bug in the grader, not a hard problem.
	"""
	started = time.perf_counter()
	before = TRACKER.total

	runtime.reset()
	if task.seed:
		runtime.put(task.seed)

	steps, answer, stop_reason, hist, repeats = 0, "", "canonical", {}, 0
	if canonical:
		runtime.write(SOLUTION, task.canonical)
	else:
		run = agent_loop(task.instruction, runtime, max_steps=max_steps, tools=tools)
		steps, answer, stop_reason = run.steps, run.answer, run.stop_reason
		hist, repeats = run.tool_histogram(), run.repeated_calls()

	passed, reason = grade(runtime, task, token)
	# The grader only ever sees the workspace, so on its own it reports "no solution.py"
	# for a provider that died at step 1 and for a model that simply never wrote one.
	# Keep both facts.
	if not passed and stop_reason == "error":
		reason = f"agent error ({run.error}); {reason}"
	elif not passed and stop_reason == "max_steps":
		reason = f"hit max_steps={max_steps}; {reason}"
	usage = Usage(
		input_tokens=TRACKER.total.input_tokens - before.input_tokens,
		cached_input_tokens=TRACKER.total.cached_input_tokens - before.cached_input_tokens,
		output_tokens=TRACKER.total.output_tokens - before.output_tokens,
		reasoning_tokens=TRACKER.total.reasoning_tokens - before.reasoning_tokens,
		cost_usd=TRACKER.total.cost_usd - before.cost_usd,
	)
	return Result(task.task_id, passed, reason, steps, time.perf_counter() - started,
	              usage, answer, stop_reason, hist, repeats)
