"""Entry point: argv -> runtime -> agent_loop -> one JSON object on stdout.

	python -m agent_loop "list the files in ."
	python -m agent_loop                        # no task: interactive, one turn per line

The contract is the Unix one, because it composes with any orchestrator without
inventing a protocol, and because nobody outside the container holds a Python
reference to the result:

  stdout      one JSON object - answer, usage, steps, stop reason
  stderr      logs and the streaming trace
  exit code   0 if the task finished, 1 if it did not

Nothing is extracted implicitly. If a run produced files and nobody asked for them,
they go when the container does. That is intended.
"""

import json
import sys
from dataclasses import asdict

from dotenv import load_dotenv

from .loop import AgentRun, agent_loop, log
from .runtime import DockerRuntime
from .providers import TRACKER

USAGE = 'usage: python -m agent_loop "<task>"   (launch it with ./run.sh; ./test.sh runs the suites)'


def chat(runtime: DockerRuntime) -> int:
	"""Interactive mode: each line is a task, and the conversation carries over between them."""
	print("agent-loop interactive. Ctrl-D or 'exit' to quit.", file=sys.stderr)
	history = None
	while True:
		try:
			prompt = input("\nyou> ").strip()
		except EOFError:
			return 0
		if prompt in ("exit", "quit"):
			return 0
		if not prompt:
			continue
		run = agent_loop(prompt=prompt, runtime=runtime, history=history, verbose=False)
		history = run.messages
		print(f"\nagent> {run.answer or run.error or f'stopped: {run.stop_reason}'}", flush=True)


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
	load_dotenv()
	runtime = DockerRuntime()
	runtime.setup()
	if not argv:
		if not sys.stdin.isatty():
			print(USAGE, file=sys.stderr)
			return 2
		return chat(runtime)
	run = AgentRun([], "error", 0, "did not start")
	try:
		run = agent_loop(prompt=" ".join(argv), runtime=runtime)
	except Exception as exc:  # noqa: BLE001 - a crash is still a result to report
		log(f"[ERROR] {type(exc).__name__}: {exc}")
		run = AgentRun([], "error", 0, f"{type(exc).__name__}: {exc}")
	finally:
		runtime.teardown()

	json.dump({
		"ok": run.ok,
		"answer": run.answer or run.error,
		"stop_reason": run.stop_reason,
		"steps": run.steps,
		"usage": asdict(TRACKER.total),
		"calls": TRACKER.calls,
	}, sys.stdout, indent=2)
	print()
	return 0 if run.ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
