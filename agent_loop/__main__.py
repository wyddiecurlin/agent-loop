"""Entry point: argv -> runtime -> agent_loop -> one JSON object on stdout.

	python -m agent_loop "list the files in ."

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

from .loop import agent_loop, final_text, log
from .runtime import DockerRuntime
from .providers import TRACKER

USAGE = 'usage: python -m agent_loop "<task>"   (launch it with ./run.sh; ./test.sh runs the suites)'


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
	if not argv:
		print(USAGE, file=sys.stderr)
		return 2

	load_dotenv()
	runtime = DockerRuntime()
	runtime.setup()
	messages: list = []
	try:
		messages = agent_loop(prompt=" ".join(argv), runtime=runtime)
		answer = final_text(messages)
		ok = bool(answer)
	except Exception as exc:  # noqa: BLE001 - a crash is still a result to report
		log(f"[ERROR] {type(exc).__name__}: {exc}")
		answer, ok = f"{type(exc).__name__}: {exc}", False
	finally:
		runtime.teardown()

	json.dump({
		"ok": ok,
		"answer": answer,
		"steps": len(messages),
		"usage": asdict(TRACKER.total),
		"calls": TRACKER.calls,
	}, sys.stdout, indent=2)
	print()
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
