#!/usr/bin/env bash
# AI_OWNED
# Talk to the agent. The audio half runs here on the host, because a container on macOS
# cannot reach the microphone; the agent itself still runs in the container, via ./run.sh.
#
#   ./voice.sh                              talk; Enter interrupts, a typed line is a turn, Ctrl-C quits
#   ./voice.sh --voice serena --lang en     another preset voice (see docs/VOICE.md), another language
#   PROVIDER=qwen QWEN_BASE_URL=https://api.lemontree.media/v1 ./voice.sh    the lowest-latency model
#
# Needs uv (brew install uv): it installs the client's five dependencies in its own cache.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "voice.sh: install uv first (brew install uv)" >&2; exit 1; }
exec uv run voice/client.py "$@"
