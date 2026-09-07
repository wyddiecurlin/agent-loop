"""End-to-end suite: the model actually drives the tools, inside the container.

	./test.sh agent
"""

import re
from datetime import datetime

from agent_loop.loop import agent_loop
from agent_loop.runtime import DockerRuntime
from agent_loop.providers import TRACKER


SANDBOX = "sandbox"

SANDBOX_FILES = {
	"notes.txt": "The answer is 42\nsecond line\nthird line: banana\n",
	"app.py": (
		"def greet(name: str) -> str:\n"
		"    return f\"Hello, {name}!\"\n"
		"\n"
		"\n"
		"if __name__ == \"__main__\":\n"
		"    print(greet(\"World\"))\n"
	),
	"data/config.json": '{"version": "1.2.3", "debug": false}\n',
}


def setup_sandbox(runtime: DockerRuntime) -> None:
	"""Fresh sandbox/ tree with known contents so the fs/shell tests are deterministic.

	One put() rather than a write() per file: on a remote runtime the difference is one
	round trip versus len(SANDBOX_FILES) of them.
	"""
	runtime.remove(SANDBOX)
	runtime.put({f"{SANDBOX}/{rel}": content for rel, content in SANDBOX_FILES.items()})


def teardown_sandbox(runtime: DockerRuntime) -> None:
	runtime.remove(SANDBOX)


def contains_number(text: str, n: int) -> str:
	"""True if `n` appears in `text`, ignoring thousands separators like 54,702 or LaTeX 54{,}702."""
	normalized = re.sub(r"(?<=\d)(,|\{,\}|\s)(?=\d{3})", "", text)
	return re.search(rf"(?<!\d){n}(?!\d)", normalized) is not None


def check(label: str, passed: bool, detail: str = "") -> bool:
	print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
	return passed


def run_case(n: int, runtime: DockerRuntime, prompt: str) -> str:
	print(f"\n{'=' * 70}\nTest case {n}: {prompt}\n{'=' * 70}")
	return agent_loop(prompt=prompt, runtime=runtime).answer


def run_suite(runtime: DockerRuntime) -> None:
	results: list[bool] = []

	# --- arithmetic / date tools -------------------------------------------------

	answer = run_case(1, runtime, "Get today's date and compute the multiplication of month, day, and year.")
	from datetime import datetime
	today = datetime.now()
	expected = today.month * today.day * today.year
	results.append(check("case 1: product of month*day*year", contains_number(answer, expected), f"expected {expected}"))

	answer = run_case(
		2, runtime,
		"Multiply month, day, and year of today's date, and do the same for the founding date of "
		"China's communist party 1949.10.1, and substract the two results.",
	)
	expected = today.month * today.day * today.year - 10 * 1 * 1949
	results.append(check("case 2: difference of the two products", contains_number(answer, expected), f"expected {expected}"))

	# --- filesystem / shell tools --------------------------------------------------

	setup_sandbox(runtime)
	try:
		# 3. fs_list
		answer = run_case(3, runtime, "List every file under the sandbox directory, recursively, and tell me how many files there are.")
		results.append(check(
			"case 3: fs_list finds all 3 files",
			"3" in answer and "notes.txt" in answer and "app.py" in answer and "config.json" in answer,
		))

		# 4. fs_read
		answer = run_case(4, runtime, "Read sandbox/data/config.json and tell me the version number it contains.")
		results.append(check("case 4: fs_read reports version", "1.2.3" in answer))

		# 5. fs_search
		answer = run_case(5, runtime, "Search the sandbox directory for the word 'banana'. Tell me the file name and the line number it appears on.")
		results.append(check("case 5: fs_search locates banana", "notes.txt" in answer and "3" in answer))

		# 6. fs_patch
		answer = run_case(6, runtime, "In sandbox/app.py, change the greeting word 'Hello' to 'Howdy'. Do not change anything else.")
		app_src = runtime.read_text(f"{SANDBOX}/app.py")
		results.append(check(
			"case 6: fs_patch edits app.py in place",
			"Howdy" in app_src and "Hello" not in app_src and "greet(" in app_src,
		))

		# 7. shell_run
		expected_out = runtime.run(f"python3 {SANDBOX}/app.py").stdout.strip()
		answer = run_case(7, runtime, "Run the shell command `python3 sandbox/app.py` and tell me exactly what it printed.")
		results.append(check("case 7: shell_run captures stdout", expected_out in answer, f"expected {expected_out!r}"))

		# 8. combined: list + read/shell + create file + read back
		answer = run_case(
			8, runtime,
			"Create a new file sandbox/summary.md containing a markdown bullet list of every file in the sandbox "
			"directory (recursively) with its line count, for example '- notes.txt: 3 lines'. "
			"Then read the file back and confirm its contents.",
		)
		summary_exists = runtime.stat(f"{SANDBOX}/summary.md") is not None
		body = runtime.read_text(f"{SANDBOX}/summary.md") if summary_exists else ""
		results.append(check(
			"case 8: summary.md created with all files",
			summary_exists and all(name in body for name in ("notes.txt", "app.py", "config.json")),
			f"exists={summary_exists}",
		))
	finally:
		teardown_sandbox(runtime)

	print(f"\n[RESULT] {sum(results)}/{len(results)} test cases passed")
	print(f"[USAGE] {TRACKER.summary()}")


def main() -> None:
	"""Own the runtime for the whole suite: one sandbox, torn down whatever happens.

	This process is already inside the container, so the runtime is a plain one - the
	boundary is one level out, put there by ./run.sh before Python started.
	"""
	runtime = DockerRuntime()
	runtime.setup()
	try:
		run_suite(runtime)
	finally:
		runtime.teardown()


if __name__ == "__main__":
	main()
