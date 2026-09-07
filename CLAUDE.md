# agent-loop

A self-improving agent OS: a small agent loop and a prebaked tool registry, always run
inside a container. Roadmap: `docs/ROADMAP.md`.

## Ownership — read before editing

Files whose first line is `# AI_OWNED` may be changed freely: `agent_loop/providers.py`,
everything in `evals/`, everything in `tests/`.

**Every other file is written and managed by humans** — `loop.py`, `tools.py`,
`runtime.py`, `__main__.py`, the shell scripts, the Dockerfile, `docs/`. Do not edit
one without (1) explicit permission for that specific file and (2) a concrete plan the
user has agreed to. Proposing a diff is fine; applying one unasked is not.

## Layout

- `agent_loop/` — `loop.py` (the loop), `tools.py` (registry), `runtime.py` (the only
  module allowed to touch the filesystem or spawn a process), `providers.py` (backends).
- `evals/` — HumanEval/MBPP harness, graded out of the agent's reach.
- `tests/` — sandbox conformance + escape, provider catalog, eval anti-cheat, e2e.
- `docs/` — RUNTIME, PROVIDERS, EVALS, WEB, ROADMAP.

## Running things

Nothing runs on the host; `./run.sh` is the only thing that creates a container.

    ./run.sh "task"            one task, JSON on stdout; no arg = interactive chat
    PROVIDER=fireworks MODEL=kimi-k3 ./run.sh "..."
    ./test.sh [lint|sandbox|providers|evals|agent]
    ./evals.sh --dataset humaneval --limit 20 [--shards N]

## Conventions

- stdout is result JSON; all narration goes to stderr.
- `./test.sh lint` enforces that only `runtime.py` reaches the world — keep it passing.
- Validate the grader with `--canonical` before believing any eval number.
