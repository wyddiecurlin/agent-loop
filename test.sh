#!/usr/bin/env bash
# Every suite runs where the agent runs: inside the container, launched by ./run.sh.
#
#   ./test.sh            lint, then sandbox, then agent
#   ./test.sh sandbox    runtime conformance + escape, no model
#   ./test.sh agent      the 8 end-to-end cases
#   ./test.sh lint       the boundary is structural (a host-side grep)
#
# There is only one topology now, so words like "docker" or "local" are accepted and
# ignored -- there is nothing left for them to select.
set -euo pipefail
cd "$(dirname "$0")"

suite=all
for a in "$@"; do case $a in lint | sandbox | agent) suite=$a ;; esac; done
want() { [ "$suite" = all ] || [ "$suite" = "$1" ]; }

if want lint; then
	# runtime.py is the ONLY module allowed to touch a filesystem or spawn a process.
	# If this prints a hit, "the agent runs in the sandbox" is a convention again.
	echo "==> lint: runtime.py is the only way to the world"
	pat='^[[:space:]]*(import|from)[[:space:]]+(subprocess|shutil|os\.path)|(^|[^._[:alnum:]])open\(|Path\(|os\.(walk|remove|mkdir|makedirs|system|popen)\('
	for f in agent_loop/*.py; do
		[ "$f" = agent_loop/runtime.py ] && continue
		if hits=$(grep -nE "$pat" "$f"); then
			echo "FAIL $f reaches the world directly:"; echo "$hits" | sed 's/^/  /'; exit 1
		fi
	done
	echo "ok   $(ls agent_loop/*.py | grep -vc runtime.py) modules clean"
fi

if want sandbox; then
	echo "==> sandbox: runtime conformance + escape, inside the container"
	AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_sandbox
fi

if want agent; then
	echo "==> agent: 8 end-to-end cases, the model driving tools inside the container"
	AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_agent
fi
