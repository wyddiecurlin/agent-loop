"""HumanEval and MBPP, normalised to one shape.

Neither benchmark ships an agent harness. Both are a jsonl file holding a natural
language task and a few assert statements, so there is no framework to integrate with -
we read the file and build the task ourselves. What `human-eval` adds beyond the data is
`execution.py`, whose exec call ships commented out precisely because running model
written code unsandboxed is dangerous. We have a sandbox; we use ours instead.

The one rule that shapes everything here: **the grader is never a file in the agent's
workspace.** It is a string that lives in this process until the agent has stopped
running. There is nothing on disk for the agent to edit, so `shell_run` and `fs_patch`
have no purchase on the score.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# Where the agent works. One file at the workspace root: nothing for a small model to
# get lost in, and it means the instruction can name the path in three words.
SOLUTION = "solution.py"


@dataclass(frozen=True)
class Task:
	task_id: str
	instruction: str  # what the agent is told
	seed: dict[str, str]  # files placed in the workspace before the agent starts
	grader: str  # python source, run AFTER the agent stops, never visible to it
	canonical: str  # the reference solution, for --canonical harness validation


def _grader(setup: str, tests: str, token: str) -> str:
	"""Build the grading script: solution, then setup, then asserts, in one namespace.

	The ordering is not cosmetic. Both benchmarks concatenate solution and tests into a
	single scope, and MBPP's `test_setup_code` leans on it - task 367's setup builds
	`Node(1)` using a class the *solution* defines. Importing the solution as a module
	instead puts those in two namespaces and the task fails on a NameError that has
	nothing to do with the answer. So: exec the file into our own globals, then run the
	setup, then assert.

	Exit code alone is forgeable - a solution whose import calls `sys.exit(0)` exits 0
	having asserted nothing - so passing also requires `token` on stderr. It is generated
	per task and the agent never sees it, so nothing but the last line of this file
	actually being reached can produce it. `_t` is bound after the exec, so a solution
	that defines its own `_t` cannot shadow it.
	"""
	return (
		'_src = open("solution.py").read()\n'
		'exec(compile(_src, "solution.py", "exec"), globals())\n\n'
		f"{setup}\n\n{tests}\n\n"
		f"import sys as _t; _t.stderr.write({token!r})\n"
	)


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

HUMANEVAL_INSTRUCTION = """\
The file `solution.py` holds a Python function whose body is missing - it has only the \
imports, the signature, and the docstring.

Implement the function so that it does what the docstring says. Then write the complete \
file back to `solution.py` with the fs_write tool: the original imports, the unchanged \
signature and docstring, and your working body.

Rules:
- Keep the function name and parameters exactly as given.
- Write Python source only. No markdown fences, no commentary in the file.
- You may run `python3 solution.py` to check that it at least parses.

Call `done` once `solution.py` holds your finished implementation.

Here is the current content of solution.py:

```python
{prompt}
```
"""


def humaneval(token_for) -> list[Task]:
	tasks = []
	for row in (json.loads(l) for l in (DATA / "HumanEval.jsonl").read_text().splitlines() if l.strip()):
		tid = row["task_id"]  # e.g. "HumanEval/0"
		entry = row["entry_point"]
		tasks.append(Task(
			task_id=tid,
			instruction=HUMANEVAL_INSTRUCTION.format(prompt=row["prompt"].rstrip()),
			seed={SOLUTION: row["prompt"]},
			grader=_grader("", f"{row['test']}\ncheck({entry})", token_for(tid)),
			canonical=row["prompt"] + row["canonical_solution"],
		))
	return tasks


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

# Task ids 11-510 are the test split (README: 1-10 are the few-shot prompts, 511-600
# validation, 601-974 training). Scoring anything else is not comparable to a published
# number, so the default range is not configurable by accident.
MBPP_TEST_SPLIT = range(11, 511)

MBPP_INSTRUCTION = """\
Write a Python solution for this task:

{text}

Your code must pass these tests:

```python
{tests}
```

Write your solution to the file `solution.py` with the fs_write tool.

Rules:
- The function must have exactly the name and parameter order used in the tests above.
- Write Python source only. No markdown fences, no commentary in the file.
- You may run `python3 solution.py` to check that it at least parses.

Call `done` once `solution.py` holds your finished implementation.
"""


def mbpp(token_for) -> list[Task]:
	tasks = []
	for row in (json.loads(l) for l in (DATA / "mbpp.jsonl").read_text().splitlines() if l.strip()):
		if row["task_id"] not in MBPP_TEST_SPLIT:
			continue
		tid = f"mbpp/{row['task_id']}"
		tests = "\n".join(row["test_list"])
		tasks.append(Task(
			task_id=tid,
			# Showing the asserts is the standard MBPP prompt (see the dataset README);
			# without them the function name is unguessable and every task fails on naming.
			instruction=MBPP_INSTRUCTION.format(text=row["text"].strip(), tests=tests),
			seed={},  # nothing to seed: the agent creates the file, which is the point
			grader=_grader(row.get("test_setup_code") or "", tests, token_for(tid)),
			canonical=row["code"],
		))
	return tasks


DATASETS = {"humaneval": humaneval, "mbpp": mbpp}


def load(name: str, token_for) -> list[Task]:
	if name not in DATASETS:
		raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
	return DATASETS[name](token_for)
