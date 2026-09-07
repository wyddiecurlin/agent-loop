"""Entry point for the eval harness. Same Unix contract as `agent_loop`:
stdout is one JSON object, stderr is the trace, exit code says whether the run finished.

	./evals.sh --dataset humaneval --limit 20
	./evals.sh --dataset mbpp --limit 20
	./evals.sh --dataset humaneval --canonical      # validate the grader, expect ~100%

One container for the whole suite, `runtime.reset()` between tasks: the reset unit is a
git checkout rather than a container boot, which is ~50ms instead of ~600ms. The cost of
that choice is that tasks cannot run in parallel, which does not matter yet - inference
dominates the wall clock by an order of magnitude.
"""

import argparse
import json
import secrets
import sys
import time
from dataclasses import asdict

from dotenv import load_dotenv

from agent_loop.loop import log
from agent_loop.providers import TRACKER, default_model
from agent_loop.runtime import DockerRuntime

from .datasets import DATASETS, load
from .harness import EVAL_TOOLS, EVAL_TOOLS_NO_SHELL, MAX_STEPS, run_task


def parse_args(argv: list[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(prog="evals", description="Run HumanEval / MBPP through the agent loop.")
	p.add_argument("--dataset", choices=sorted(DATASETS), default="humaneval")
	p.add_argument("--limit", type=int, default=20, help="how many tasks to run (0 = all)")
	p.add_argument("--offset", type=int, default=0, help="skip this many tasks first")
	p.add_argument("--max-steps", type=int, default=MAX_STEPS)
	p.add_argument("--no-shell", action="store_true",
	               help="withhold shell_run: scores one-shot writing with no chance to run the code")
	p.add_argument("--canonical", action="store_true",
	               help="write the reference solution instead of calling a model; validates the grader")
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(sys.argv[1:] if argv is None else argv)
	load_dotenv()

	# One token per task, generated here and never shown to the agent. See harness.grade.
	tokens: dict[str, str] = {}

	def token_for(task_id: str) -> str:
		return tokens.setdefault(task_id, f"__EVAL_OK_{secrets.token_hex(8)}__")

	tasks = load(args.dataset, token_for)
	tasks = tasks[args.offset:]
	if args.limit:
		tasks = tasks[: args.limit]

	tools = EVAL_TOOLS_NO_SHELL if args.no_shell else EVAL_TOOLS
	model = "canonical" if args.canonical else default_model()
	log(f"==> {args.dataset}: {len(tasks)} tasks, model={model}, tools={tools + ['done']}")

	runtime = DockerRuntime()
	runtime.setup()
	results, started = [], time.perf_counter()
	try:
		for i, task in enumerate(tasks, 1):
			r = run_task(runtime, task, tokens[task.task_id], tools=tools,
			             max_steps=args.max_steps, canonical=args.canonical)
			results.append(r)
			passed = sum(x.passed for x in results)
			log(f"[{'PASS' if r.passed else 'FAIL'}] {r.task_id}  ({i}/{len(tasks)}, "
			    f"running {passed}/{i} = {passed / i:.1%})" + (f"  -- {r.reason}" if r.reason else ""))
	finally:
		runtime.teardown()

	passed = sum(r.passed for r in results)
	total = len(results)
	summary = {
		"dataset": args.dataset,
		"model": model,
		"tools": tools + ["done"],
		"n": total,
		"passed": passed,
		# n=1 per task, so pass@1 is just the mean. The unbiased estimator in the
		# human-eval repo only matters when sampling several completions per task.
		"pass@1": round(passed / total, 4) if total else 0.0,
		"duration_s": round(time.perf_counter() - started, 1),
		"usage": asdict(TRACKER.total),
		"results": [asdict(r) for r in results],
	}
	json.dump(summary, sys.stdout, indent=2, default=str)
	print()
	log(f"\n==> {args.dataset} pass@1 = {passed}/{total} = {summary['pass@1']:.1%}   "
	    f"({summary['duration_s']}s, ${TRACKER.total.cost_usd:.4f})")
	return 0 if total else 1


if __name__ == "__main__":
	raise SystemExit(main())
