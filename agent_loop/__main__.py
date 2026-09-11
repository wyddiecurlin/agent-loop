"""CLI entry point inside the container; the host launcher owns session persistence."""

import json
import os
import sys
from dataclasses import asdict

from dotenv import load_dotenv

from .loop import AgentRun, agent_loop, log
from .runtime import DockerRuntime
from .providers import TRACKER

USAGE = 'usage: python -m agent_loop "<task>" (launch it with agent-loop or ./run.sh)'


def run_turn(runtime: DockerRuntime, prompt: str | None, history: list | None, *, verbose: bool) -> AgentRun:
    runtime.in_turn = True
    try:
        return agent_loop(prompt=prompt, runtime=runtime, history=history, verbose=verbose)
    finally:
        runtime.in_turn = False


def chat(runtime: DockerRuntime) -> int:
    print("agent-loop interactive. Ctrl-D or 'exit' to quit.", file=sys.stderr)
    history = runtime.conversation.get("messages") or None
    continuing = runtime.conversation.get("continue", False)
    runtime.conversation["interactive"] = True
    while True:
        if continuing:
            prompt = None
            continuing = False
        else:
            try:
                prompt = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            if prompt in ("exit", "quit"):
                return 0
            if not prompt:
                continue
        run = run_turn(runtime, prompt, history, verbose=False)
        history = run.messages
        if run.stop_reason == "mount":
            return 75
        print(f"\nagent> {run.answer or run.error or f'stopped: {run.stop_reason}'}", flush=True)
        if runtime.stop_requested:
            return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not os.getenv("AGENT_CONTROL_SOCKET"):
        load_dotenv()
    runtime = DockerRuntime()
    run = AgentRun([], "error", 0, "did not start")
    try:
        runtime.setup()
        runtime.install_signal_handlers()
        if not argv or (runtime.conversation.get("continue") and runtime.conversation.get("interactive")):
            if not sys.stdin.isatty() and not os.getenv("AGENT_CONTROL_SOCKET"):
                print(USAGE, file=sys.stderr)
                return 2
            return chat(runtime)
        prompt = None if runtime.conversation.get("continue") else " ".join(argv)
        run = run_turn(runtime, prompt, runtime.conversation.get("messages") or None, verbose=True)
        if run.stop_reason == "mount":
            return 75
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        run = AgentRun([], "error", 0, f"{type(exc).__name__}: {exc}")
    finally:
        runtime.teardown()
    json.dump({"ok": run.ok, "answer": run.answer or run.error, "stop_reason": run.stop_reason,
               "steps": run.steps, "usage": asdict(TRACKER.total), "calls": TRACKER.calls}, sys.stdout, indent=2)
    print()
    return 0 if run.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
