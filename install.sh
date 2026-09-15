#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")" && pwd)
bin_dir=${AGENT_BIN_DIR:-$HOME/.local/bin}
mkdir -p "$bin_dir"
if [ -e "$bin_dir/agent-loop" ] || [ -L "$bin_dir/agent-loop" ]; then
  if [ "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$bin_dir/agent-loop")" != "$repo_dir/agent-loop" ]; then
    echo "$bin_dir/agent-loop already exists" >&2
    exit 1
  fi
else
  ln -s "$repo_dir/agent-loop" "$bin_dir/agent-loop"
fi
echo "Installed $bin_dir/agent-loop. Ensure $bin_dir is on PATH."
