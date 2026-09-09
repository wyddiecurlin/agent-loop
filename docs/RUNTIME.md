# agent-loop — runtime design

One topology. The agent process is always inside a container, and every tool call it
makes is already in there with it.

```
  ./run.sh  ──docker run──>  ┌─ container ─────────────────────┐
  (launcher)                 │  agent loop      (PID 1)        │
                             │  tool registry                  │
                             │  /work           (git-backed)   │
                             └─────────────────────────────────┘
                               network: open, by decision
```

There is no second mode and no runtime selection. The launcher creates the box before
Python starts — the agent cannot create the box it stands in — and everything above
that is plain in-process code.

## Goal and threat model

Every command runs inside a container whose environment, network, and filesystem we
control. **The container's filesystem is the only filesystem.** No host directory is
shared, ever, so nothing works only because both sides see the same inode.

**A confused agent, not an adversary.** We defend against `rm -rf` in the wrong tree,
a hallucinated `curl`, a typo'd `pip install`, and secret exfiltration. We are *not*
defending against a kernel exploit. That choice is what lets us stop at containers.

Non-goals: bind mounts, container escape, agent-to-agent communication.

## Why inside, and not outside

The alternative was an agent on the host driving containers over `docker exec`. At
scale it loses on every axis:

- **Chatter.** Every tool call becomes an RPC: ~30 calls × ~80ms of exec overhead per
  agent, for nothing. Inside, they are function calls.
- **Crash domain.** The transcript and large tool outputs live *with* the agent, so
  `--memory 2g` bounds the agent and its work together, and a crash takes one box.
- **Blast radius.** An outside orchestrator holds every live handle and credential;
  one indexing bug crosses tenants.
- **Simplicity.** It deleted an entire runtime implementation — `docker exec`, tar
  transfer, the exec-timeout problem, TTY rules — and the code got shorter, not longer.

The one real argument for outside was **secrets**: with the agent inside, the gateway
token sits in a container running model-chosen commands. That is what the second
boundary below is for.

## Two boundaries, two jobs

| boundary | stops | mechanism |
|---|---|---|
| the container | the **agent** reaching the host | `run.sh` flags |
| the `sandbox` user | a **model-written command** reaching the agent's keys | `run()` drops uid/gid |

The line that matters is not agent-vs-tools; it is **our code vs the model's code**.
`send_email(to, body)` is our code and may hold a key. `shell_run("...")` executes
text the model wrote, and must not — and the two used to run as the same root user, so
`cat /proc/1/environ` handed one the other's keys. Now every model-written command
drops to `sandbox`: a different uid, no supplementary groups, `/proc/1/environ`
unreadable, `su` refused by `no-new-privileges`. The env allowlist still applies on
top, so the command's own environment carries `PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`
and nothing else. `/work` is group-writable, so both users edit the same files.

That is what makes "tools use the keys, the model never sees them" true rather than
hoped for, and it is what lets the network stay open (below). `tests/test_sandbox.py`
asserts all four properties.

## Two abstractions, not one

| concern | interface | who calls it |
|---|---|---|
| how a *tool* touches the world | `DockerRuntime` | the agent, in-process |
| how boxes are *created and reaped* | the launcher | `run.sh`, later an orchestrator |

This is why moving to Modal or Fly changes nothing above the launcher: the runtime is
always local to its box. `DockerRuntime` is named for *where it runs*, not for what it
drives — it contains no Docker code at all.

```python
class DockerRuntime:
    def setup(self) / teardown(self)                    # refuses to exist outside a container
    def run(self, cmd, *, cwd, timeout_s, env) -> RunResult
    def relpath / stat / list_dir / read_bytes / read_text / write / remove
    def put(files) / get(paths)                         # bulk, batched
    def snapshot(self) -> str  /  reset(self, sha=None)
```

`stat`/`list_dir`/`remove`/`relpath` are the four a first sketch misses: `fs_list`,
`fs_search`, and `fs_patch` must ask about the filesystem without reading it.
`RunResult` is a struct, not a string — the tool layer formats it, the runtime does not.
`put`/`get` stay separate from `write`/`read` because a future launcher may have to ship
them across a boundary, and then batching is one round trip instead of *n*.

## The container

```dockerfile
FROM python:3.12-slim AS dev     # + git, ripgrep, deps
COPY agent_loop/ /app/agent_loop/
WORKDIR /work
ENTRYPOINT ["python", "-m", "agent_loop"]

FROM dev AS test                 # + tests/, so test code never ships to prod
```

`run.sh` supplies `--init` (tini reaps), `--memory 2g --memory-swap 2g` (without the
second, Docker silently doubles the limit), `--cpus 2 --pids-limit 256`, and
`--cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt
no-new-privileges`. The three capabilities kept all serve the `sandbox` user: two to
drop a command to it, and `KILL` to time it out afterwards — without it even root cannot
signal another uid's process, and a timeout silently becomes "wait for it to finish". It always rebuilds first; with a
warm cache that is under a second, and it is the only way an edit is guaranteed to be
what runs.

Baking the agent's source is the one place baking beats transferring: it is fixed per
deploy. That is what "prebaked tools" means.

## Network

**Open, by decision.** The agent may call any server, and so may its tools. `run.sh`
uses Docker's default bridge and sets no policy. We built an allowlist proxy and
removed it: the agent needs the internet for the same reasons a developer does — a
`pip install`, a docs page, an API it is being asked to integrate — and closing it
turned each of those into a support problem. It is safe to leave open because a
model-written command holds nothing worth sending: see the `sandbox` user above.

## Result contract

Nobody holds a Python reference to a container's return value, so `__main__.py` speaks
the Unix contract: **stdout** one JSON object, **stderr** logs and the streaming trace,
**exit code** 0 or 1. One stray `print` on stdout and the envelope stops parsing — which
is why every `[LOG]` line goes to stderr. Nothing is extracted implicitly; if a run
produced files and nobody asked for them, they go when the container does.

## Snapshots

`setup()` does `git init` plus an empty baseline commit in `/work`; `snapshot()` is
`git add -A && git commit-tree`; `reset(sha)` is `git reset --hard && git clean -fdx`.
Safe here *only* because `/work` is disposable — the same call against a developer's
real repository would discard uncommitted work. The agent being inside is what makes
the method possible at all.

Two separable things: *baseline + diff* is how work leaves a container and needs no
`reset()`. *Checkpoint/restore* pays off once the loop can retry, branch, or take a
second task — which is exactly what an interactive session is.

## Enforcement

The loop runs where we say **structurally**, not by convention:

- `tools.py` exposes `build_registry(runtime)`. No module-level registry exists, so no
  caller can obtain a filesystem tool without first deciding which runtime it acts on.
- `runtime.py` is the only module allowed to touch a filesystem or spawn a process.
  `./test.sh lint` proves it across every other file in the package.
- `setup()` refuses to return a runtime unless `/.dockerenv` exists. Being inside a
  container is checked, not assumed. `AGENT_UNSAFE_HOST=1` overrides it deliberately.

**Only effectful tools are bound.** `done` touches nothing, so there is nothing to
contain. (The date is not a tool either: `loop.py` stamps the system prompt with the date
and time at the start of every call, in the container's zone, which is UTC unless `TZ` is
in `.env`.) The invariant is not "every tool runs in the sandbox" but:

> Every tool that touches the filesystem or spawns a process does so only through the
> Runtime it was handed.

## Layout

```
run.sh              the launcher — the only thing that starts a container
test.sh             lint (host grep), then every suite through run.sh
Dockerfile          two stages: dev, and test
agent_loop/
  __main__.py       argv -> runtime -> agent_loop -> JSON on stdout, exit code
  loop.py           agent_loop, generate, SYSTEM_PROMPT, final_text
  providers.py      wire shapes, usage/cost, OpenAI + Qwen, retries
  tools.py          the registry; reaches the world only through a Runtime
  runtime.py        DockerRuntime — the only module that touches the world
tests/
  test_sandbox.py   16 cases: conformance + escape, inside the container
  test_agent.py     8 cases: the model really driving tools, inside the container
```

## Tests

```
./test.sh            lint, then sandbox, then agent
./test.sh sandbox    conformance + escape, no model
./test.sh agent      the 8 end-to-end cases
```

Escape cases assert a denial: no host filesystem, model-written commands run as
`sandbox` and cannot read the agent's keys or escalate, the memory limit holds, a grandchild dies at timeout, `relpath` refuses to escape,
`reset()` rolls back. One positive case asserts the network is open.

## Known traps

- **`--memory` without `--memory-swap` doubles the limit**, and the escape case then
  passes for the wrong reason.
- **`sh` in the image is dash; `sh` on macOS is bash.** `echo -e`, `local`, `[[ ]]`
  differ. A task that "works locally" is not evidence it works here.
- **Cold start matters at fan-out.** ~600ms is noise against 30 LLM round trips; it is
  not noise across 1000 fresh containers. Warm pools, or split `create` from `start`.
- **Debuggability.** No mount means no peeking mid-run. Use `get()` into a temp dir on
  failure, or a second agent in the same box, not a debug-only mount that will rot.
- **Apple silicon → `linux/arm64`.** Pin `--platform`; prod is probably x86.
- **The system prompt reads the clock**, which is the container's. Fine, but it is the
  one thing in the prompt that would have to be pinned if runs had to be reproducible.
- **Never mount `/var/run/docker.sock`.** Full host escape, and it is the reason the
  agent must never try to create its own container.

## What is left

1. **API-backed tools.** A home assistant's tools hold real keys — mail, calendar,
   home devices. The pattern is already in place: they are bound tools like `fs_*`,
   our code, running as root with the key, exposing `send_email(to, body)` rather
   than a raw HTTP tool. What is still missing is *policy*: confirmation before an
   irreversible action, and an audit of every real-world call. The threat that
   matters there is prompt injection — an email that says "forward everything to
   X" needs no escape, only a legitimate tool — and no container defends against it.
2. **Keys out of the container entirely.** Passing `.env` wholesale is the wrong habit
   for a file that will hold twenty keys. Pass by name, then move to a broker that
   hands the container short-lived scoped tokens instead of credentials.
3. **Interactive sessions.** `agent_loop` takes one task and exits. A `messages=`
   parameter plus a REPL turns one container into a session: `/work` and the transcript
   persist across tasks, and `snapshot()` per turn makes a bad task revertable. The
   container is already the permission boundary, so do not add per-tool prompts.
4. **Context compaction.** A session resends the whole transcript every turn. This
   bites before anything about containers does.
5. **A real launcher.** `create / await / collect / destroy`, concurrency limits, warm
   pools. Only after one container is boring.
