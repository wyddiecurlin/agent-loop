#!/usr/bin/env bash
# The launcher: the one place a container is created, and the only way the agent runs.
#
#   ./run.sh "fix the failing test in src/parser.py"    one task, JSON result on stdout
#   ./run.sh                                             no task: interactive chat
#
# Provider and model are a prefix, never an edit:
#   PROVIDER=fireworks MODEL=kimi-k3 ./run.sh "..."     (falls over to together)
#   PROVIDER=fireworks FALLBACK=none ./run.sh "..."     (one platform, no cover)
#
# The agent process is never outside. It cannot create the box it stands in, so this
# script does -- and everything else, including the test suites, comes through here.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${AGENT_SESSION:-0}" = 1 ]; then
	exec python3 -m launcher.main "$@"
fi

TARGET=${AGENT_TARGET:-dev}          # dev | test: the Dockerfile stage
IMAGE=agent-loop:$TARGET

# Always build. With the cache warm this is well under a second, and it is the only
# way a code change is guaranteed to be what runs -- an "if missing" check silently
# runs stale code after every edit.
docker build -q --target "$TARGET" -t "$IMAGE" . >/dev/null

opts=(--rm --init
	--memory 2g --memory-swap 2g --cpus 2 --pids-limit 256
	# Three capabilities kept, all for the `sandbox` user: SETUID/SETGID to drop a
	# model-written command to it, KILL to time it out afterwards -- without CAP_KILL even
	# root cannot signal another uid's process. no-new-privileges stops the way back up.
	--cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges)
# stdin is always attached, so a pipe reaches the agent too (voice/client.py drives
# voice/bridge.py this way); a pseudo-terminal only when there is a terminal at both ends.
opts+=(--interactive)
if [ -t 0 ] && [ -t 1 ]; then opts+=(--tty); fi
if [ -f "${AGENT_ENV_FILE:-.env}" ]; then opts+=(--env-file "${AGENT_ENV_FILE:-.env}"); fi
# Anything set in the caller's shell wins over .env, so a provider A/B is a prefix on the
# command rather than an edit to a secrets file:  PROVIDER=qwen ./evals.sh --dataset ...
for v in PROVIDER FALLBACK MODEL REASONING_EFFORT MAX_OUTPUT_TOKENS CONTEXT_WINDOW_TOKENS \
	OPENAI_MODEL QWEN_MODEL QWEN_BASE_URL QWEN_THINKING QWEN_TEMPERATURE QWEN_SEED \
	WEB_SEARCH_PROVIDER BRAVE_SEARCH_API_KEY PARALLEL_SEARCH_API_KEY PARALLEL_SEARCH_MODE; do
	if [ -n "${!v:-}" ]; then opts+=(-e "$v=${!v}"); fi
done
if [ -n "${AGENT_ENTRYPOINT:-}" ]; then opts+=(--entrypoint "$AGENT_ENTRYPOINT"); fi

exec docker run "${opts[@]}" "$IMAGE" "$@"
