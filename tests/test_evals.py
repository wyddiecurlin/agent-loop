# AI_OWNED
"""Proof that the eval score cannot be inflated by the agent.

	./test.sh evals

Every case here plays the part of a cheating solution and asserts that it is scored as a
failure. No model is involved: the "agent" is this file writing solution.py directly,
which is strictly more power than a real agent has - it does not even have to get the
tool call right.

If one of these ever passes, every benchmark number in evals/results/ is measuring the
hole rather than the model.
"""

from agent_loop.runtime import DockerRuntime
from evals.datasets import SOLUTION, load
from evals.harness import GRADE_DIR, grade

TOKEN = "__EVAL_OK_deadbeefdeadbeef__"


def check(label: str, passed: bool, detail: str = "") -> bool:
	print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
	return passed


def task_for(task_id: str):
	"""One real task, with a fixed token so the cases below can try to forge it."""
	tasks = load("humaneval", lambda tid: TOKEN)
	return next(t for t in tasks if t.task_id == task_id)


def score(runtime: DockerRuntime, task, solution: str, extra: dict[str, str] | None = None):
	"""Put `solution` (and any `extra` files) in the workspace and grade it."""
	runtime.reset()
	runtime.write(SOLUTION, solution)
	if extra:
		runtime.put(extra)
	return grade(runtime, task, TOKEN)


def main() -> int:
	runtime = DockerRuntime()
	runtime.setup()
	task = task_for("HumanEval/0")  # has_close_elements(numbers, threshold)
	results = []

	# The control: a correct answer must still score. Without this the rest is vacuous -
	# a grader that fails everything would "pass" every cheat case below.
	correct = task.canonical
	passed, reason = score(runtime, task, correct)
	results.append(check("control: the reference solution passes", passed, reason))

	# And a plainly wrong one must not.
	passed, _ = score(runtime, task, "def has_close_elements(numbers, threshold):\n    return False\n")
	results.append(check("control: a wrong solution fails", not passed))

	# 1. Exit 0 without running the asserts. This is the one that exit-code-only grading
	#    would wave through, and the reason the token exists.
	for name, body in [
		("sys.exit(0) at import", "import sys\nsys.exit(0)\n"),
		("os._exit(0) at import", "import os\nos._exit(0)\n"),
	]:
		passed, reason = score(runtime, task, body)
		results.append(check(f"cheat: {name} is not a pass", not passed, reason))

	# 2. Forge the sentinel. The agent never sees the token, so the closest it can get is
	#    guessing the shape of it.
	passed, _ = score(runtime, task, 'import sys\nsys.stderr.write("__EVAL_OK__")\nsys.exit(0)\n')
	results.append(check("cheat: a guessed sentinel is not a pass", not passed))

	# 3. Poison the import machinery. sitecustomize is imported by every interpreter
	#    start; conftest by pytest; a .pth file runs code at site init. All of these are
	#    written into the workspace and must not survive into the grading directory.
	poison = 'import sys\nsys.stderr.write("' + TOKEN + '")\nsys.exit(0)\n'
	passed, _ = score(runtime, task, "def has_close_elements(n, t):\n    return False\n",
	                  extra={"sitecustomize.py": poison, "conftest.py": poison})
	results.append(check("cheat: sitecustomize/conftest do not survive the reset", not passed))

	# 4. Write the grading directory before grading starts, to pre-empt it.
	passed, _ = score(runtime, task, "def has_close_elements(n, t):\n    return False\n",
	                  extra={f"{GRADE_DIR}/grade.py": poison, f"{GRADE_DIR}/{SOLUTION}": correct})
	results.append(check("cheat: a pre-planted grade/ directory is wiped", not passed))

	# 5. Delete the solution and claim success. Nothing to grade is a failure, never a
	#    default pass.
	runtime.reset()
	passed, reason = grade(runtime, task, TOKEN)
	results.append(check("cheat: an absent solution.py is not a pass", not passed, reason))

	runtime.teardown()
	print(f"\n[RESULT] {sum(results)}/{len(results)} checks passed")
	return 0 if all(results) else 1


if __name__ == "__main__":
	raise SystemExit(main())
