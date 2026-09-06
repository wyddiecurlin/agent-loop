#!/usr/bin/env bash
# Enforcement check: tools.py and main.py must reach the world only through a Runtime.
# If this ever prints a hit, the sandbox has a bypass and "runs in the container" is
# back to being a convention rather than a fact.
set -u
cd "$(dirname "$0")"

pattern='^[[:space:]]*(import|from)[[:space:]]+(subprocess|shutil|os\.path)|(^|[^._[:alnum:]])open\(|Path\(|os\.(walk|remove|mkdir|makedirs|system|popen)\('
status=0

for f in tools.py main.py; do
	if hits=$(grep -nE "$pattern" "$f"); then
		echo "FAIL $f must not touch the filesystem directly:"
		echo "$hits" | sed 's/^/  /'
		status=1
	else
		echo "ok   $f"
	fi
done

exit $status
