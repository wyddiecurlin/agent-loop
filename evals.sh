#!/usr/bin/env bash
# Benchmarks, run where everything else runs: inside the container, via ./run.sh.
#
#   ./evals.sh --dataset humaneval --limit 20
#   ./evals.sh --dataset mbpp --limit 20
#   ./evals.sh --dataset humaneval --canonical --limit 164   # grade the reference
#                                                            # solutions: expect ~100%
#
# Always validate the grader with --canonical before believing an agent score. A
# canonical run that is not near 100% means the harness is broken, and every agent
# number it produced is measuring that instead of the agent.
#
# stdout is the JSON summary, so redirect it to keep a run:
#   ./evals.sh --dataset mbpp --limit 20 > evals/results/mbpp-20.json
set -euo pipefail
cd "$(dirname "$0")"

AGENT_TARGET=test AGENT_ENTRYPOINT=python exec ./run.sh -m evals "$@"
