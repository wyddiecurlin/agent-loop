# One image, one topology: the agent always runs inside it (see RUNTIME.md).
# Two stages so test code never ships to prod.
#
#   docker build --target dev  -t agent-loop:dev  .
#   docker build --target test -t agent-loop:test .
#
# Beyond the agent's deps the runtime owes itself one thing: git, for snapshot/reset.
FROM python:3.12-slim AS dev

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent_loop/ /app/agent_loop/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# Two users. The agent (PID 1, root) holds the API keys its tools use; every command
# the *model* writes runs as `sandbox`, which cannot read /proc/1/environ and cannot
# escalate back. /work is group-writable so both can edit the same files.
RUN groupadd work && useradd -M -d /tmp -g work sandbox \
 && mkdir -p /work && chgrp work /work && chmod 2775 /work \
 && git config --system safe.directory '*'

WORKDIR /work
ENTRYPOINT ["python", "-m", "agent_loop"]

FROM dev AS test
COPY tests/ /app/tests/
