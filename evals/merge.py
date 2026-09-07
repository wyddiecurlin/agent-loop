"""Merge sharded eval runs into one summary.

	python3 -m evals.merge evals/results/humaneval_shard*.json > evals/results/humaneval.json

Shards share no state, so merging is concatenating `results` and re-deriving the
aggregates. Wall clock is the max across shards, not the sum: they ran at the same time.
"""

import collections
import json
import sys


def merge(paths: list[str]) -> dict:
	rows, usage, wall, meta = [], collections.Counter(), 0.0, {}
	for path in paths:
		with open(path) as f:
			d = json.load(f)
		rows += d["results"]
		wall = max(wall, d.get("duration_s", 0.0))
		for k, v in d["usage"].items():
			usage[k] += v
		meta = {k: d[k] for k in ("dataset", "model", "tools") if k in d}

	rows.sort(key=lambda r: r["task_id"])
	passed, n = sum(r["passed"] for r in rows), len(rows)
	tools = collections.Counter()
	for r in rows:
		tools.update(r.get("tools_used", {}))
	return {
		**meta,
		"n": n,
		"passed": passed,
		"pass@1": round(passed / n, 4) if n else 0.0,
		"wall_clock_s": round(wall, 1),
		"shards": len(paths),
		"stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows).most_common()),
		"tool_calls": dict(tools.most_common()),
		"turns_mean": round(sum(r["steps"] for r in rows) / n, 1) if n else 0,
		"repeated_calls_total": sum(r.get("repeated_calls", 0) for r in rows),
		"usage": dict(usage),
		"results": rows,
	}


def main(argv: list[str]) -> int:
	if not argv:
		print("usage: python3 -m evals.merge SHARD.json ...", file=sys.stderr)
		return 2
	out = merge(argv)
	json.dump(out, sys.stdout, indent=2)
	print()
	print(f"==> {out.get('dataset','?')} pass@1 = {out['passed']}/{out['n']} = {out['pass@1']:.1%}  "
	      f"({out['shards']} shards, {out['wall_clock_s']}s wall)\n"
	      f"    stop_reasons={out['stop_reasons']}  turns_mean={out['turns_mean']}",
	      file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
