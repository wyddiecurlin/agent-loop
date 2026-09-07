# AI_OWNED
"""Paired comparison of two eval runs. McNemar, not two pass rates.

	python3 -m evals.compare BEFORE.json AFTER.json

Two independent runs of 500 tasks at p=0.84 give the *difference* a 95% interval of
about +/-4.5 points, so an aggregate comparison cannot see a change smaller than that -
and a context tweak that is worth having may well be worth two.

Pairing fixes that by conditioning on the task. A task that passes in both runs, or
fails in both, says nothing about which run is better; those cancel. Only the discordant
tasks carry signal, so the question shrinks to: of the tasks that changed, is the split
between fixed and broken more lopsided than a coin would give?

The list of tasks a change BROKE is usually worth more than the net number, so it is
printed even when the result is not significant.
"""

import json
import sys
from math import comb


def load(path: str) -> tuple[dict, dict]:
	with open(path) as f:
		d = json.load(f)
	return d, {r["task_id"]: r for r in d["results"]}


def two_sided_binomial(k: int, n: int) -> float:
	"""Exact two-sided p for k successes in n coin flips - the exact McNemar test.

	Exact rather than the chi-square approximation because the discordant count is
	routinely under 25, which is where the approximation stops being trustworthy.
	"""
	if n == 0:
		return 1.0
	probs = [comb(n, i) for i in range(n + 1)]
	total = float(sum(probs))
	target = probs[k]
	return min(1.0, sum(p for p in probs if p <= target + 1e-9) / total)


def main(argv: list[str]) -> int:
	if len(argv) != 2:
		print("usage: python3 -m evals.compare BEFORE.json AFTER.json", file=sys.stderr)
		return 2
	(da, a), (db, b) = load(argv[0]), load(argv[1])

	shared = sorted(set(a) & set(b))
	only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
	if only_a or only_b:
		print(f"warning: {len(only_a)} task(s) only in BEFORE, {len(only_b)} only in AFTER; "
		      f"comparing the {len(shared)} in both", file=sys.stderr)
	if not shared:
		print("error: the two runs share no task ids; pairing is impossible", file=sys.stderr)
		return 1

	both = [t for t in shared if a[t]["passed"] and b[t]["passed"]]
	neither = [t for t in shared if not a[t]["passed"] and not b[t]["passed"]]
	broke = [t for t in shared if a[t]["passed"] and not b[t]["passed"]]
	fixed = [t for t in shared if not a[t]["passed"] and b[t]["passed"]]

	pa = sum(a[t]["passed"] for t in shared) / len(shared)
	pb = sum(b[t]["passed"] for t in shared) / len(shared)
	n_disc = len(broke) + len(fixed)
	p = two_sided_binomial(len(fixed), n_disc)

	print(f"paired over {len(shared)} shared tasks")
	print(f"  BEFORE {argv[0]}: {sum(a[t]['passed'] for t in shared)}/{len(shared)} = {pa:.1%}")
	print(f"  AFTER  {argv[1]}: {sum(b[t]['passed'] for t in shared)}/{len(shared)} = {pb:.1%}")
	print(f"  net {pb - pa:+.1%}\n")
	print(f"                AFTER pass   AFTER fail")
	print(f"  BEFORE pass   {len(both):>10}   {len(broke):>10}   <- broke")
	print(f"  BEFORE fail   {len(fixed):>10}   {len(neither):>10}")
	print(f"                  ^ fixed\n")
	print(f"  concordant {len(both) + len(neither)} tasks carry no signal and are excluded.")
	print(f"  discordant {n_disc}: {len(fixed)} fixed vs {len(broke)} broke")
	print(f"  exact McNemar p = {p:.4f}  ->  "
	      f"{'a real change' if p < 0.05 else 'indistinguishable from noise'}")
	if n_disc and p >= 0.05:
		need = next((k for k in range(n_disc, -1, -1) if two_sided_binomial(k, n_disc) < 0.05), None)
		if need is not None:
			print(f"  (at {n_disc} discordant tasks you would need {need} of them fixed to clear p<0.05)")

	if broke:
		print(f"\n  BROKE ({len(broke)}): {', '.join(broke[:25])}{' ...' if len(broke) > 25 else ''}")
	if fixed:
		print(f"  FIXED ({len(fixed)}): {', '.join(fixed[:25])}{' ...' if len(fixed) > 25 else ''}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
