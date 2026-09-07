# Evals — HumanEval and MBPP

```bash
./evals.sh --dataset humaneval --canonical --limit 164   # validate the grader first
./evals.sh --dataset humaneval --limit 20
./evals.sh --dataset mbpp --limit 20
./test.sh evals                                          # prove the score cannot be forged
```

stdout is one JSON summary, stderr is the trace. Redirect stdout to keep a run.

## Why these two, and what they measure

Neither is an agent benchmark. Both are a jsonl file of `(task description, assert
statements)` with no harness attached, which is exactly why they are cheap to wire up:
there is no framework that wants to own the loop, so ours does. What `human-eval` adds
beyond the data is `execution.py`, whose `exec` call ships **commented out** because
running model-written code unsandboxed is dangerous. We have `DockerRuntime`; we use it
and ignore their file. Neither repo is a dependency — `evals/data/` is two downloaded
jsonl files.

Both benchmarks are saturated and deep in pretraining data. That is the point rather than
a flaw: with a high base rate, a failure is attributable to the *scaffold* rather than to
model weakness. This is the benchmark you run to debug the loop, not to rank the model.

| | tasks | agent writes | grading |
|---|---|---|---|
| HumanEval | 164 | body of a seeded `solution.py` | `check(entry_point)` |
| MBPP | 500 (ids 11–510, the test split) | `solution.py` from scratch | 3 asserts + `test_setup_code` |

## The score cannot be inflated

`shell_run` and `fs_patch` can reach anything in `/work`. So nothing that decides the
score is ever in `/work` while the agent is running. In order:

1. **The grader is not on disk during the run.** It is a string in the harness process.
   There is no file to patch and no test to edit.
2. **Exactly one file survives the agent.** When the loop stops, `solution.py` is lifted
   out and `runtime.reset()` (`git reset --hard && git clean -fdx`) deletes everything
   else — a planted `sitecustomize.py`, a `conftest.py`, a `.pth`, a shadowed stdlib
   module, a pre-built `grade/` directory.
3. **Grading runs in a directory holding only that file and the grader.**
4. **Passing needs exit 0 *and* a per-task random token on stderr.** Exit codes are
   forgeable — a solution whose import calls `sys.exit(0)` exits 0 having asserted
   nothing. The token is generated per task, never shown to the agent, and only the last
   line of the grader can emit it.
5. **The tool set is narrowed** to `fs_read`, `fs_write`, `fs_list`, `shell_run`, `done`.
   `fs_patch` is dropped: the task writes one file from scratch, so it is redundant.
6. **`/app` is mode 700.** The datasets carry the reference solutions and model-written
   commands run as `sandbox`, so `cat /app/evals/data/HumanEval.jsonl` is denied.
7. **The score is an exit code, never something the agent said.** `final_text()` is
   recorded for debugging and never consulted.

Defence 1 is load-bearing; the rest exist so no single mistake in it moves a number.
`./test.sh evals` plays each cheat and asserts it is scored as a failure — including a
control that a correct solution still passes, without which a grader that failed
everything would "pass" every cheat case.

`shell_run` is deliberately kept. Running your own code and reading the traceback is the
agentic part of this benchmark; remove it and what is left is one-shot code generation
measured through an agent loop, which is a different and less interesting number. Use
`--no-shell` to score that variant.

## Always validate the grader first

`--canonical` writes the reference solution instead of calling a model. It must come out
at or near 100%; anything less is a bug in the harness, and every agent number it
produced measures that instead of the agent.

This is not hypothetical. The first MBPP grader scored the reference solutions 499/500.
The one failure was task 367, whose `test_setup_code` builds `Node(1)` using a class the
*solution* defines — MBPP concatenates solution and tests into a single namespace, and
importing the solution as a module puts them in two. The grader now execs the file into
its own globals, then runs the setup, then asserts. Both datasets are 100% canonical.

## Sharding

One container per run, `runtime.reset()` between tasks: the reset unit is a git checkout
(~50ms) rather than a container boot (~600ms). The cost is that tasks within a run are
serial. For a full sweep, shard across containers with the flags that already exist:

```bash
for i in 0 1 2 3; do
  ./evals.sh --dataset humaneval --offset $((i*41)) --limit 41 \
    > evals/results/he_shard$i.json 2> evals/results/he_shard$i.err &
done; wait
```

Shards share nothing, so merging is concatenating `results` and summing `passed`.

## Scored run — 2026-09-06

`qwen3.5-9b`, thinking off, `tool_choice=required`, `max_steps=30`,
tools `fs_read fs_write fs_list shell_run done`. Full sets, 4 shards each.

| | pass@1 | scaffold-lost | model-adjusted | wall clock |
|---|---|---|---|---|
| HumanEval (164) | **133/164 = 81.1%** | 2 | 82.3% | 10 min |
| MBPP (500, ids 11–510) | **420/500 = 84.0%** | 10 | 86.0% | 69 min |

Canonical validation for the same run: HumanEval 164/164, MBPP 500/500.

"scaffold-lost" is tasks that failed without the model ever getting a fair attempt:

- **8 tasks (2 HumanEval, 6 MBPP) died on a provider 400.** The model emits a `done` call and
  then starts a *second* tool call inside the first one's arguments —
  `{"answer": "...</think>\n\n<tool_call>\n<function=done>{` — the gateway's tool parser
  fails on the unterminated JSON and returns 400. `is_retryable()` correctly does not
  retry a 400, so the exception leaves `agent_loop` and the whole task is lost.
  A malformed turn should cost a step, not the task: catching this in the loop and
  feeding it back as a tool error is worth ~1.2 points on HumanEval and ~1.2 on MBPP.
- **4 MBPP tasks produced no `solution.py`** — the loop called `done` claiming success
  with nothing written. `done` is trusted; nothing checks that the work exists.

Everything else is the model being wrong: 88 AssertionError, 7 TypeError, 2 NameError,
1 ModuleNotFoundError, and 1 grader timeout (a solution that does not terminate in 20s).
