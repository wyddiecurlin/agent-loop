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

## Measuring a change, not a coin flip

The score moves on its own. Re-running 42 tasks at identical settings once moved them
41 points: they had been selected *because* they failed in one run, and re-running
regressed them to the mean. Two things follow, and both are now defaults.

**Seeded sampling, not greedy decoding.** `QwenProvider` sends `seed=0` with
`temperature=0.6, top_p=0.95, top_k=20` (`QWEN_SEED` / `QWEN_TEMPERATURE` to override).
Repeatability comes from the seed; `temperature=0` is a different thing and is actively
harmful here - see "What moved HumanEval from 81.1% to 93.3%" for the 31k-token loop it
caused. Even seeded, this is not bitwise reproducible: under continuous batching the
kernel path depends on what else is in the batch, so 24 concurrent agents still flip the
occasional token. It removes sampling as the *dominant* source of noise, not all of it.

`OpenAIProvider` deliberately sends neither. The Responses API reasoning models reject
`temperature` outright, so a hosted run cannot be pinned the way a self-hosted one can,
and its run-to-run variance is correspondingly higher. That asymmetry is a reason to
lean on the paired comparison below rather than on aggregate pass rates.

**Paired comparison.** Two independent 500-task runs at p=0.84 give their *difference* a
95% interval of about +/-4.5 points, which is wider than most changes worth making. So
compare per task, not in aggregate:

```bash
python3 -m evals.compare before.json after.json
```

Tasks that pass in both runs, or fail in both, cancel - they carry no information about
which run is better. Only the discordant tasks count, and the question becomes whether
the split between fixed and broke is more lopsided than a coin would give (exact
McNemar). It also prints *which* tasks a change broke, which is usually worth more than
the net number. Both runs must cover the same task ids.

## Sharding

One container per shard, `runtime.reset()` between tasks inside it: the reset unit is a
git checkout (~50ms) rather than a container boot (~600ms). Tasks within a shard are
serial, so the shard count is also the number of concurrent requests the GPU sees.

```bash
./evals.sh --dataset humaneval --limit 0 --shards 24 > evals/results/humaneval.json
```

Sharding is striped (`tasks[i::N]`), not blocked. With blocks, wall clock is set by
whichever shard happened to draw the slow tasks - measured at 8 shards, that made a run
*slower* than at 4. Striping spreads them.

Measured on one self-hosted 9B 4-bit GPU, 16 tasks per level:

| containers | wall | output tok/s | prefill tok/s |
|---|---|---|---|
| 1 | 537s | 89 | 249 |
| 4 | 211s | 178 | 675 |
| 8 | 224s | 218 | 590 |
| 16 | 132s | 258 | 804 |

Throughput climbs all the way to 16, so the GPU was never saturated; read the tok/s
column, since wall clock at high concurrency is set by the slowest single task. Note
`DEFAULT_TIMEOUT_S` is 600s: as concurrency rises, requests queue, and a timeout is not
a saved second - it is a retry that costs the GPU the whole generation twice.

The deeper cost is prefill, not parallelism. Runs come in at 17-19:1 input:output with
86-91% prefix-cache hits, and every turn re-prefills the growing transcript, so GPU time
is quadratic in turn count. Cutting wasted turns beats raising the shard count.

Shards share nothing, so `python3 -m evals.merge shard*.json` concatenates `results` and
re-derives the aggregates; `--shards` does this for you.

## Scored runs

Current configuration, unless a row says otherwise: `tool_choice=required`, `max_steps=60`,
tools `fs_read fs_write fs_list shell_run done`, 24 shards, `DEFAULT_TIMEOUT_S=600`.
Qwen samples at `temperature=0.6, top_p=0.95, top_k=20, seed=0` and caps output at 8192
tokens. Every run below was preceded by a canonical validation of the dataset it scored:
HumanEval 164/164, MBPP 500/500.

### 2026-09-07 — HumanEval, three configurations, same 164 ids

| | pass@1 | wall | turns/task | output tok | reasoning tok | repeated calls | clean finishes |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5-9b, thinking **on** | **153/164 = 93.3%** | 498s | 4.1 | 303k | 208k | 13 | 162/164 |
| qwen3.5-9b, thinking **off** | 140/164 = 85.4% | 545s | 6.8 | 171k | 0 | **372** | 156/164 |
| gpt-5.4-nano (`effort: none`) | **153/164 = 93.3%** | **74s** | 4.8 | 83k | 0 | 17 | **164/164** |

Paired (`python3 -m evals.compare`):

- **qwen thinking on vs off: 5 fixed, 18 broke, p = 0.0106.** Turning thinking off is a
  real regression of 7.9 points.
- **qwen thinking on vs gpt-5.4-nano: 8 fixed, 8 broke, p = 1.0000.** A dead tie - the
  self-hosted 9B is at parity with the hosted model here.
- **qwen thinking off vs gpt-5.4-nano: 19 fixed, 6 broke, p = 0.0146.** gpt beats
  *no-think* qwen, which is the same comparison reaching the opposite verdict once one
  knob moves. Report which configuration was measured, not just which model.

**Thinking tokens are cheaper than turns.** The obvious read of row 1 - 208k reasoning
tokens must be waste - is wrong twice over. Removing them cost 8 points, and it did not
even return the time: 498s became 545s. Output halved, but repeated tool calls went 13 ->
372 and turns went 4.1 -> 6.8, because the model does the same reasoning through the tool
loop instead. Reasoning tokens are generated once inside one request; an extra turn pays
prefill on the whole transcript again, and at 17-19:1 input:output that is the more
expensive half. Keep `QWEN_THINKING=1`.

**HumanEval is finished as a signal at this level.** Two models 25x apart in size score
identically, and the benchmark is saturated and in pretraining data. 93.3% says they have
both converged on its ceiling, not that either writes good code. Measure context changes
against MBPP or something unsaturated.

### 2026-09-06 — the first full run, and what it cost

`qwen3.5-9b`, thinking off, `max_steps=30`, provider defaults for sampling, 4 shards.

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
- **4 MBPP tasks produced no `solution.py`**, and reading the traces they split two ways.
  Two (84, 348) called `done` claiming success having written nothing - 84 put the code
  in the `answer` text instead of the file. Two (122, 430) never called `done` at all;
  they exhausted `max_steps` and the grader found an empty workspace. Only the first pair
  is a `done`-is-trusted problem; the second pair is budget, and the wider version of it
  is below.

- **42 of 500 MBPP tasks (8.4%) exhausted `max_steps=30`**, and 36 of them failed. That
  was the largest single loss in this run and it is invisible in the score: `grade()`
  reads the workspace regardless of how the loop ended, so the 6 that exhausted their
  budget with a correct file already written still passed.

Everything else is the model being wrong: 88 AssertionError, 7 TypeError, 2 NameError,
1 ModuleNotFoundError, and 1 grader timeout (a solution that does not terminate in 20s).

### What moved HumanEval from 81.1% to 93.3%

Paired over the same 164 ids: **25 fixed, 5 broke, p = 0.0003**. Four changes landed
together, so the credit is not separable, but each was forced by an observed failure.

**`max_steps` 30 -> 60.** Weakly supported. Re-running the 42 MBPP tasks that had
exhausted 30 gave 23/42 at 30 and 28/42 at 60 - +12 points with a +/-15 interval, so not
significant at n=42. Included because it demonstrably converts `max_steps` stops into
`done` stops, not because the score moved.

**`DEFAULT_TIMEOUT_S` 60 -> 600.** At 24 concurrent containers a request queues behind
the others. A timeout there is not a saved minute; it is a retry that pays for the whole
generation twice and re-rolls the sample, putting variance back into a run meant to be
repeatable.

**Sampling, seeded - *not* greedy.** `temperature=0` was tried first and is a trap. It
walked the model into an unbounded reasoning loop: 31,360 output tokens, every one of
them reasoning, no text and no tool call, task after task. Sampling escapes such a loop by
taking a different token; greedy cannot, because the argmax that opened it is the same
argmax every time. Repeatability comes from the **seed**, so the fix was to keep `seed=0`
and sample at the model card's coding values. `max_tokens=8192` bounds the damage if it
recurs.

**The loop no longer retries an identical request.** The runaway above exposed a real bug
rather than just a bad setting: a turn returning neither text nor a tool call appended
*nothing*, so the next request was byte-identical to the one that had just failed. Under
sampling that escapes on its own; under greedy it never can, and the loop would have run
31k-token generations for all 60 steps against an unchanging context. A stalled turn now
gets a nudge appended - which is what makes attempt two differ from attempt one - and two
consecutive stalls end the run with `stop_reason="stalled"`. It fired once in the run
above, on HumanEval/145, and caught a real one.

Still open, both visible in the table as non-`done` finishes:

- **The provider 400 survives.** One HumanEval task still dies when the model nests a
  second tool call inside the first one's arguments and the gateway's parser rejects the
  whole request. It is nondeterministic - re-running the same task passes - so treating
  that specific 400 as retryable would recover it. `is_retryable()` correctly declines all
  400s today; the discrimination has to be narrow, or a genuinely malformed request burns
  three retries and still fails.
- **`done` is trusted.** Two MBPP tasks in the 2026-09-06 run called `done` claiming
  success having written no file. The *score* is unaffected - `grade()` catches it - but
  the loop should not accept a completion whose work does not exist. The mechanism is
  already there: `done` returns `ok=False` like any tool, and the model gets another turn.
  It needs a precondition, bound at `build_registry` time the way the runtime is.
