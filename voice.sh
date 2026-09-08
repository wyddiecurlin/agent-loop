#!/usr/bin/env bash
# AI_OWNED
# Talk to the agent. The audio half runs here on the host, because a container on macOS
# cannot reach the microphone; the agent itself still runs in the container, via ./run.sh.
#
#   ./voice.sh                              talk; Enter interrupts, a typed line is a turn, Ctrl-C quits
#   ./voice.sh --lang zh --voice serena     pin Mandarin, with the Chinese preset voice
#   ./voice.sh --lang en                    pin English; the default detects each turn
#   PROVIDER=qwen QWEN_BASE_URL=https://api.lemontree.media/v1 ./voice.sh    the lowest-latency model
#
# Every run is logged and traced in full under logs/voice/<timestamp>/ (never committed;
# logs/voice/latest is the newest run). Set VOICE_LOG_DIR or pass --log-dir to move it.
#
# Needs uv (brew install uv): it installs the client's five dependencies in its own cache.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "voice.sh: install uv first (brew install uv)" >&2; exit 1; }
exec uv run voice/client.py "$@"
