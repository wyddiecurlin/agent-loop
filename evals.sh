#!/usr/bin/env bash
# Benchmarks, run where everything else runs: inside the container, via ./run.sh.
#
#   ./evals.sh --dataset humaneval --limit 20
#   ./evals.sh --dataset humaneval --canonical --limit 164   # grade the reference
#                                                            # solutions: expect ~100%
#   ./evals.sh --dataset mbpp --limit 0 --shards 24          # the full split, 24 containers
#
# Always validate the grader with --canonical before believing an agent score. A
# canonical run that is not near 100% means the harness is broken, and every agent
# number it produced is measuring that instead of the agent.
#
# --shards N runs N containers at once and merges them. Each agent loop is strictly
# serial - one request in flight - so N is also the number of concurrent requests the
# GPU sees, and it is the only throughput knob that matters here: container boot is
# ~600ms once per shard against a run measured in minutes. Sharding is striped
# (tasks[i::N]), so a slow task does not decide the wall clock of its whole shard.
#
# stdout is the JSON summary, so redirect it to keep a run:
#   ./evals.sh --dataset mbpp --limit 0 --shards 24 > evals/results/mbpp.json
set -euo pipefail
cd "$(dirname "$0")"

# Pull --shards N out of the arguments; everything else goes through untouched.
SHARDS=1
args=()
while [ $# -gt 0 ]; do
	case "$1" in
		--shards) SHARDS=$2; shift 2 ;;
		--shards=*) SHARDS=${1#*=}; shift ;;
		*) args+=("$1"); shift ;;
	esac
done

run_one() { AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m evals "$@"; }

if [ "$SHARDS" -le 1 ]; then
	exec_args=("${args[@]+"${args[@]}"}")
	run_one "${exec_args[@]+"${exec_args[@]}"}"
	exit $?
fi

# Build the image once up front. Otherwise N containers race on `docker build` at
# start-up, and the log of the run is buried under N copies of the build output.
AGENT_TARGET=test docker build -q --target test -t agent-loop:test . >/dev/null

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
	run_one "${args[@]+"${args[@]}"}" --shard "$i/$SHARDS" \
		> "$tmp/shard$i.json" 2> "$tmp/shard$i.err" &
	pids+=($!)
done

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
cat "$tmp"/shard*.err >&2
[ "$failed" -eq 0 ] || echo "warning: at least one shard exited non-zero" >&2

# Merged on the host: the shard files are here, and evals/merge.py is stdlib-only, so it
# needs neither the image nor a mount.
python3 -m evals.merge "$tmp"/shard*.json
