# Roadmap — Self-improving Agent OS

The goal is an agent that gets better at its job over time: a small, auditable core it
runs inside, an eval that can prove a change helped, and a learning layer that feeds the
first two. Three tracks, in dependency order — core, then eval, then learning.

Status: `[x]` done, `[ ]` not started.

## Core (500 LOC budget)

The whole core stays small enough to read in one sitting. Everything the agent can reach
is bound to the Runtime it was handed, so "the agent runs in the sandbox" is structural
rather than a convention (`./test.sh lint`).

- [x] **model** — providers and the wire shapes they speak (`docs/PROVIDERS.md`).
- [x] **agent loop** — one request in flight, tools dispatched through the registry.
- [x] **tools** — the prebaked registry, reaching the world only via Runtime.
- [x] **sandbox** — one container topology; conformance and escape both tested.
- [x] **log and tracing** — stdout is result JSON, stderr is the trace.
- [ ] **web browsing** — a native way for the agent to fetch and read the web
      (plan: `docs/WEB.md`).
- [ ] **image understanding** — vision in the loop, including face recognition.
- [ ] **mountable folders** — attach arbitrary host directories when starting a session.
- [ ] **resume from container state** — a session that survives the container it began in.
- [ ] **user context** — who the agent is working for, carried into every session.
- [ ] **memory** — what persists across sessions, and how it is retrieved.
- [ ] **rewind and fork** — the agent can snapshot, roll back, and branch its own computer.
- [ ] **cron** — scheduled work the agent wakes up to do.
- [ ] **inbox / message queue** — durable work items the agent processes on its own.
- [ ] **human monitoring** — a live view of what it is doing while it does it.

## Eval

Nothing on the learning track is believable without this. Grading happens somewhere the
agent cannot reach, and a `--canonical` run validates the grader before any agent number
is trusted (`docs/EVALS.md`).

- [x] **run benchmarks** — HumanEval and MBPP, sharded, with anti-cheat tests.
- [ ] **learning improvements over time** — measure the same agent across versions of
      itself, paired rather than aggregate (`evals/compare.py` does McNemar for exactly
      this reason: two independent 500-task runs cannot see a change under ~4.5 points).

## Learning

The payoff: the agent changes itself, and the eval says whether that was an improvement.

- [ ] **build its own toolbox** — the agent writes, keeps, and reuses its own tools.
- [ ] **efficient memory store** — a retrieval layer good enough that the agent measurably
      performs better as it accumulates experience.
