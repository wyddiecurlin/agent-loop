# Ivon — isolated runtime, plan & abstractions

Ivon is the agent core: the loop, plus a prebaked tool registry. It ships in two
shapes from one binary.

| mode | agent process runs | `AGENT_SANDBOX` | isolated from | entrypoint |
|---|---|---|---|---|
| **CLI** | the developer's machine | `docker` | that machine | `ivon run <task>` |
| **Fleet** | inside the container | `local` | the host, by the container | `python -m ivon` as PID 1 |

Everything below follows from that table. Get the topology right and the rest is
plumbing; get it wrong and no amount of sandboxing helps.

## Goal

Every command the agent runs executes inside a container whose environment
variables, network access, and filesystem we control. **The container's
filesystem is the only filesystem.** No host directory is shared. Files enter
and leave through an explicit transfer API.

**Threat model: a confused agent, not an adversary.** We defend against `rm -rf`
in the wrong tree, a hallucinated `curl`, a typo'd `pip install`, and secret
exfiltration through inherited env vars. We are *not* defending against a kernel
exploit. That choice is what lets us stop at containers.

**Design constraint that drives everything: prod runs 1000 remote containers.**
No mechanism may assume the host and the container share a disk. Anything that
works only because both sides see the same inode is a local-only affordance that
dies on the move to prod — bind mounts above all. Every file crossing the
boundary crosses it explicitly, and every crossing is a round trip we can count.

## Non-goals (for now)

- Bind mounts, shared volumes, any host path visible inside.
- Defending against a container escape.
- Agent-to-agent communication. Each run is independent.

---

## Topology: where the agent loop runs

Two arrangements are possible, and the choice is the architecture.

**A — agent outside, container is an execution target:**

```
orchestrator process ──N Runtime handles──> N containers
  (loop, registry, LLM client, keys)          (filesystem + shell only)
```

**B — agent inside the box:**

```
orchestrator ──spawn──> N containers, each running `python -m ivon` as PID 1
                          (loop, registry, tools, workspace, all inside)
```

**Fleet mode is B.** At 1000 agents:

- **Chatter.** In A every tool call is an RPC across the container boundary:
  1000 agents × ~30 calls × ~80ms of exec overhead, for nothing. In B they are
  local function calls.
- **Memory and crash domain.** The agent accumulates a transcript and parses
  large tool outputs. In A that lives in a shared orchestrator, so one runaway
  agent degrades the fleet and `--memory` does not cover it. In B, `--memory 2g`
  bounds the agent *and* its work, and a crash takes down one container.
- **Blast radius.** In A the orchestrator holds every live handle and every
  credential; one indexing bug crosses tenants.

The one real argument for A is **secrets**: with the agent inside, the LLM API
key sits in a container executing model-chosen commands — exactly the
exfiltration the env allowlist exists to prevent.

**Resolution: B plus a gateway.** The agent inside holds a short-lived scoped
token, never `OPENAI_API_KEY`, and talks to an LLM gateway we run. Network policy
allows that one host and nothing else. If the token leaks it buys minutes of
inference against our gateway, not our account. The `QWEN_BASE_URL` /
model-agnostic gateway work already has this shape; pointing the OpenAI
provider's `base_url` at the same gateway generalizes it.

### The consequence people miss

In fleet mode `make_runtime()` returns **`LocalRuntime`, and that is correct** —
not a downgrade. The container already provides the boundary, so the path checks
inside go back to being what they honestly are: argument hygiene, with the real
isolation one level out. `DockerRuntime` is the *CLI* story, where the agent sits
on a developer machine and needs a sandbox around its tools.

This is also why Claude Code ships `--dangerously-skip-permissions`. When an
agent is alone in a disposable container, per-tool approval prompts buy nothing —
the container **is** the permission boundary.

---

## Two abstractions, not one

One laptop running one agent hides the fact that these are different concerns.
At 1000 they separate, and forcing them into one interface fits badly.

| concern | interface | who calls it | fleet impl |
|---|---|---|---|
| how a *tool* touches the world | `Runtime` | the agent, in-process | `LocalRuntime`, inside the box |
| how boxes are *created and reaped* | `Launcher` | the orchestrator | Docker / Modal / Fly API |

A `Launcher` wants `create(task)`, `await_result()`, `collect_artifacts()`,
`destroy()`, plus concurrency limits, retries, and warm pools. Do not grow it out
of `Runtime`.

### `Runtime`, as built in `runtime.py`

```python
class Runtime(Protocol):
    def setup(self)   -> None            # create the environment
    def teardown(self) -> None           # destroy it

    # execution
    def run(self, cmd, *, cwd=None, timeout_s=30, env=None) -> RunResult

    # tool I/O — per-call, small, chatty
    def relpath(self, path)              -> str          # normalize + containment, no I/O
    def stat(self, path)                 -> FileStat | None
    def list_dir(self, path, *, recursive, skip, max_entries) -> list[FileStat]
    def read_bytes(self, path, *, max_bytes=None)        -> bytes
    def read_text(self, path)            -> str
    def write(self, path, content)       -> None
    def remove(self, path)               -> None

    # provisioning — bulk, at setup/teardown, batched
    def put(self, files: Mapping[str, str | bytes])      -> None
    def get(self, paths: Iterable[str])                  -> dict[str, bytes]

    # state
    def snapshot(self)                   -> str
    def reset(self, snapshot=None)       -> None
```

`stat` / `list_dir` / `remove` / `relpath` are the four the first sketch missed:
`fs_list`, `fs_search`, and `fs_patch` need to ask about the filesystem without
reading it, and doing that through `run("ls")` would mean parsing shell output.
`list_dir` carries an explicit ordering contract (directories first then files,
alphabetical; recursive yields files only, depth-first) so tool output does not
shift when the runtime is swapped.

`RunResult` is a struct, not a string: `(exit_code, stdout, stderr, timed_out,
duration_s)`. The tool layer formats it for the model; the runtime does not.

**Two transfer surfaces, deliberately separate**, because remote makes the cost
difference stark:

| | when | volume | Docker impl |
|---|---|---|---|
| `read`/`write` | per tool call, in the loop | one small file | `exec cat` / `exec tee` |
| `put`/`get` | setup / teardown | whole trees | tar stream via `docker cp` |

Never loop `write()` over a thousand files. On a remote runtime that is a
thousand round trips; `put()` takes a dict and sends one tar.

| impl | `run` | transfer | isolation |
|---|---|---|---|
| `LocalRuntime` | `subprocess` | host file I/O | none *by itself* — see topology |
| `DockerRuntime` | `docker exec` | `docker cp` / stdin | container (CLI mode) |
| *`RemoteRuntime`* | vendor exec API | vendor upload API | container, elsewhere |

If adding a row ever requires touching `agent_loop`, the interface was wrong.

---

## Container model

### Fleet mode — the agent is PID 1

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ivon/ /app/ivon/
WORKDIR /work
ENTRYPOINT ["python", "-m", "ivon"]
```

```
docker run --rm --init \
  --memory 2g --cpus 2 --pids-limit 256 \
  --cap-drop ALL --security-opt no-new-privileges \
  --network ivon-egress \             # NOT none — the agent calls the gateway
  -e IVON_GATEWAY_URL=... -e IVON_GATEWAY_TOKEN=<short-lived, scoped> \
  -e AGENT_SANDBOX=local \
  ivon:<pinned-tag> "fix the failing test in src/parser.py"
```

No `-v`. Nothing from the host is reachable. No `docker exec` either: the
container's whole life *is* one run, and it exits when the agent does.

Baking the agent's own source into the image is the one place where baking beats
transferring — it is fixed per deploy, so paying a transfer every run is waste.
"Prebaked tools" means exactly this. Per-deployment tool *selection* is then
config (an allowlist of tool names), not different code.

### CLI mode — the agent is outside

Long-lived container, `sleep infinity` as PID 1 to hold the namespaces open, and
every tool call is `docker exec -w <cwd> <cid> sh -c "<command>"`. Here
`--network none` **is** right: the agent is on the developer's machine and makes
the LLM calls itself, so the container needs no egress at all.

### Notes that apply to both

- **`--init`.** Gives `tini` as a reaping PID 1. In CLI mode `exec`'d processes
  are not children of PID 1, so without it a long session accumulates zombies.
- **Env starts empty.** A container's environment is essentially just `PATH`,
  which makes the allowlist the platform default rather than something we
  remember. `LocalRuntime` already matches this (`DEFAULT_ENV_ALLOWLIST`), so the
  two implementations stay honest with each other.
- **`-t` is forbidden.** A TTY merges stdout and stderr and injects control
  characters. We capture them separately and feed them to a model.
- **The image is the real ongoing cost.** The agent can only use tools that exist
  inside. Pin the tag; `git` is mandatory (see Snapshots).

### What persists between `exec` calls (CLI mode)

| persists | does not |
|---|---|
| filesystem, installed packages | `cd` |
| background processes, dev servers | `export` / shell env |
| network state | shell functions, history |

Each `exec` is a brand new `sh`. `shell_run` already takes `cwd`, which becomes
`docker exec -w <cwd>`. Do **not** try to hold an interactive shell open over
stdin.

---

## Getting the task in, and the result out

### In

1. **argv / env** for the prompt and config. Enough for most tasks.
2. **`put()`** — a tar stream — for a workspace. One round trip, binary-safe,
   arbitrarily many files. Every remote provider has a direct equivalent.
3. **Bake into the image** for anything fixed at build time.
4. **Let the container fetch it** — `git clone` during a network-open setup
   phase, then lock down. Best when the source of truth is already a remote.

### Out — the contract we do not have yet

Today `done()`'s answer is a return value read by `run_suite`. In a container
nobody holds that reference. The contract should be the Unix one:

- **stdout** — one JSON object: final answer, `git diff`, usage/cost, stop reason.
- **stderr** — logs and streaming trace.
- **exit code** — did the task succeed.

That composes with any orchestrator without inventing a protocol, and it is
exactly what `claude -p --output-format json` does.

Nothing is extracted implicitly. If a run produced something and nobody captured
the diff, it is gone when the container exits. That is intended.

`TRACKER` is a module global — fine per-container, but budgets now aggregate
across 1000 processes. Emit usage as part of the stdout envelope rather than
printing a summary.

---

## Snapshots run *inside*

Host-side git over a shared directory is impossible without a mount, so git runs
in the container:

- `snapshot()` → `git -C /work stash create` (or `write-tree` + `commit-tree`).
- `reset(sha)` → `git reset --hard <sha> && git clean -fdx`.
- `setup()` does `git init` + a baseline commit after `put()`, so there is always
  a base to diff and reset against.

Better than the mounted version: identical code on Docker and any remote
provider, no host git state to corrupt, and the isolation boundary and the
rollback boundary become the same object — the container's `/work`. Anything the
run changed is either in there (revertable) or was never possible.

**Split the two things this bundles.** *Baseline + diff* is needed almost
immediately — it is how the work product leaves a remote container — and needs no
`reset()` at all. *Checkpoint/restore* only pays off once the loop can retry or
branch, which it cannot today. Ship `snapshot()` because it is four lines once
git is there; expect nothing to call `reset()` yet. Keep the method: retrofitting
checkpointing later is a control-flow change in `agent_loop`, not plumbing.

`LocalRuntime` raises `NotImplementedError` for both, deliberately — its root is
a real repository, and `reset()` there would discard uncommitted work.

---

## Enforcement — done

The loop runs where we say **structurally**, not by convention.

- `tools.py` exposes `build_registry(runtime)`; there is no module-level
  registry, so no caller can obtain a filesystem tool without first deciding
  which runtime it acts on.
- Tool functions receive `runtime` and have no other way to touch the world:
  `tools.py` does not import `subprocess` and does not open a file.
- `agent_loop(prompt, runtime)` builds its registry from what it is handed.
- `main()` owns the lifecycle; `run_suite(runtime)` uses what it is given.

**`./lint_isolation.sh`** is the proof, and it covers `tools.py` and `main.py`.
If it ever prints a hit, "runs in the sandbox" is back to being a convention.

**Only effectful tools are bound.** `substract`, `multiply`, `get_today_date`,
and `done` run as ordinary Python — they read no file, spawn no process, open no
socket, so there is nothing to contain. In fleet mode they run inside the
container anyway, because the whole process does. The invariant is not "every
tool runs in the sandbox" but:

> Every tool that touches the filesystem or spawns a process does so only through
> the Runtime it was handed.

**And a hard fail, never a silent downgrade:** probe `docker info` at startup; if
`AGENT_SANDBOX=docker` and the daemon is down, raise. `make_runtime` already
refuses unknown backends rather than falling back.

---

## Package layout

`main.py` is 742 lines holding cost accounting, message types, two providers,
retries, `generate`, `agent_loop`, *and* the test harness. `python -m ivon`
cannot import that without dragging the tests along.

```
ivon/
  __main__.py    argv -> runtime -> agent_loop -> JSON on stdout, exit code
  loop.py        agent_loop, SYSTEM_PROMPT
  providers.py   OpenAIProvider, QwenProvider, make_provider, with_retries
  usage.py       Usage, CostTracker, compute_cost
  tools.py       (unchanged)
  runtime.py     (unchanged)
tests/
  test_suite.py  today's run_suite / run_case / check
```

`runtime.py` and `tools.py` survive untouched, which is the abstraction paying
off.

---

## Escape suite

Ordinary test cases, each asserting a *denial*. Run against `LocalRuntime` first
and expect failures — that list **is** the build order.

| case | expected | mode |
|---|---|---|
| `cat /work/../.env` | no such file | both |
| `env \| grep -i key` | empty (allowlist) | both |
| `ls /Users` | does not exist | both |
| `cd / && ls` | the image, not the Mac | both |
| `while :; do :; done` | killed at timeout, nothing left running | both |
| `:(){ :\|:& };:` | contained by `--pids-limit` | both |
| `python -c "b=' '*10**10"` | killed by `--memory` | both |
| `curl -m5 https://example.com` | fails | CLI (`--network none`) |
| `curl -m5 https://example.com` | fails; gateway host succeeds | fleet (egress policy) |

---

## Known traps

- **`--network none` is wrong for fleet mode.** The agent inside needs exactly
  one route out, to the gateway. Binary on/off is no longer sufficient, and the
  egress-allowlist case we wanted to defer is no longer deferrable.
- **`docker exec` has no timeout, and killing the client does not kill the
  process inside.** Kill the client, then reach back in (`pkill`), or accept the
  leak with `docker rm -f` at teardown as the backstop. Decide explicitly.
  (CLI mode only; fleet mode has no `exec`.)
- **Cold start now matters.** For one long session it is noise — one ~600ms boot
  against 30 LLM round trips of 1–30s each. For 1000 fresh containers it is not.
  Warm pools, or splitting `docker create` from `docker start`.
- **Debuggability regresses.** No mount means no peeking at files mid-run.
  Mitigate with a `dump` helper (`get()` into a temp host dir on failure), not a
  debug-only mount that will rot.
- **Apple silicon → `linux/arm64`.** Pin `--platform`; prod is probably x86.
- **`get_today_date` reads the host clock**, not the container's. The one unbound
  tool that would have to move if runs ever need to be reproducible.
- **Never mount `/var/run/docker.sock`.** Full host escape.
- **Stale containers** from crashed runs — prune by name prefix at startup.

---

## Build order

1. ~~**Injection.** Kill `ROOT` and the `REGISTRY` singleton;
   `build_registry(runtime)`.~~ **Done.** `./lint_isolation.sh` enforces it.
2. ~~**`LocalRuntime`** implementing the full interface including `put`/`get`.~~
   **Done.** Also picked up the process-group timeout fix and an env allowlist,
   so its behavior already matches what Docker will do.
3. ~~**Harness migration** — `setup_sandbox` → `put()`, the three host-reading
   assertions → `runtime.read_text`/`runtime.run`.~~ **Done.** `main()` owns the
   lifecycle; `run_suite(runtime)` uses what it is handed.
4. **`DockerRuntime`** — lifecycle, `exec`, tar transfer, env allowlist,
   `--network none`, resource limits, in-container git. This is **CLI mode**.
5. **Escape suite green** against `DockerRuntime`.
6. **Package split** into `ivon/` + `tests/`, with `__main__.py` and the stdout
   JSON result contract. Nothing about `runtime.py` or `tools.py` should change.
7. **Fleet image** — Dockerfile above, `python -m ivon` as PID 1,
   `AGENT_SANDBOX=local` inside. Verify one container end to end by hand before
   writing any orchestration.
8. **Gateway** — scoped short-lived tokens, egress policy allowing only the
   gateway host. Until this exists, fleet mode is not safe to run on untrusted
   tasks.
9. **`Launcher`** — create / await / collect / destroy, concurrency limits,
   warm pool. Only after a single container is boring.

If step 7 requires changing `agent_loop`, go back to step 1.
