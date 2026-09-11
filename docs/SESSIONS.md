# CLI sessions, volume mounts, and resume

`agent-loop` launches a local container, saves its conversation and filesystem on managed exit, and resumes it by session ID. The [host launcher](../launcher/main.py) owns Docker and durable state. The [runtime](../agent_loop/runtime.py) accesses only the workspace, declared mounts, and a private launcher control socket.

**Install and launch.** Requires Python 3.10+, Bash, and Docker on the host. The agent runs in a Linux container.

```bash
./install.sh                       # adds ~/.local/bin/agent-loop; put that directory on PATH
agent-loop --grant-file /private/grant.json --backend-url <backend-url>
agent-loop --grant-file /private/grant.json --backend-url <backend-url> "a task"
agent-loop --list
agent-loop --resume <session-id> --grant-file /private/fresh-grant.json --backend-url <backend-url>
```

The authenticated client obtains the grant as described in [USER_ENV.md](USER_ENV.md). Resume verifies both the saved owner and session ID. Use `agent-loop --local` (and `--local --resume <id>`) for development with the repository's `.env`; `AGENT_ENV_FILE` can select another file. Local sessions are bound to the host user. The existing `./run.sh` remains the disposable launcher used by tests and evals.

**Host folders.** Pass `--mount /absolute/folder` for read-only access or `--mount-rw /absolute/folder` for writable access. During a conversation, the model can call `mount_volume(host_path, read_only=true)`. Writable access is intended for user-requested edits. The launcher resolves the source path, checks that it is a directory, and rejects paths overlapping its private state, credential sources, or protected system directories.

Mounted folders appear under `/mnt/volumes/<id>`. Filesystem tools accept those absolute paths, and shell tools can use them as a working directory. Read-only mounts are enforced by Docker and the runtime. File operations on mounted data use the host user's filesystem identity; shell processes use the host UID/GID plus the container workspace group. Mount sources must therefore be accessible to the launching user.

```text
mount_volume -> private launcher socket -> validated mount mapping
completed tool turn -> conversation checkpoint -> container exits
launcher -> commit/save -> fresh container with mount -> continue conversation
```

Docker mounts are fixed at creation, so a new tool-requested mount restarts the container after the current tool turn. The loop saves all tool outputs before restarting and continues without repeating the user prompt. Existing tools in that same turn finish before the mount becomes available. Mounts live outside `/work`; workspace Git snapshots and reset/clean do not traverse them. Writable mounts change the host files directly and are not rolled back with a session.

**Save and restore.** State lives under `$XDG_STATE_HOME/agent-loop/` or `~/.local/state/agent-loop/`, with private permissions. Each `sessions/<id>/` directory contains a lock, `metadata.json`, a conversation JSON file, and a container image archive. Metadata records the owner, session ID, image ID, archive/conversation references, nonsecret configuration, mount mappings, and timestamps. Only one launcher may use a session at a time.

On normal exit or managed shutdown, the current tool turn completes and checkpoints its conversation. The container then stops, ending remaining processes before the launcher uses [Docker commit](https://docs.docker.com/reference/cli/docker/container/commit/) and [Docker save](https://docs.docker.com/reference/cli/docker/image/save/). After both the image archive and conversation file are durable, the launcher atomically publishes metadata and removes the container. The previous snapshot files are then removed. If saving fails, it retains the container and the previous published checkpoint; `active.json` identifies the retained container, and automatic resume refuses to overwrite its unsaved work.

Resume loads the saved image into a fresh container and reattaches the original mounts. `/work`, files installed elsewhere in the container, conversation history, and nonsecret configuration survive. Missing or changed mount sources fail before launch. RAM and running processes do not survive. Abrupt host failure may leave a retained container and a pending conversation checkpoint; inspect and recover that container before removing `active.json`. Automatic crash recovery, rewind, fork, and shared hosted session infrastructure remain out of scope.

Secrets are supplied through a protected mount, not Docker environment configuration. That mount and host folders are excluded from the saved image. Python starts with a safe import path so a model-written module in `/work` cannot replace trusted application code on resume. This implementation supersedes the mount/session restrictions in [RUNTIME.md](RUNTIME.md).

**Validation.** `./test.sh sessions` runs real Docker lifecycle tests: restore of conversation and files inside/outside `/work`, an actual loop continuing across a mount, read-only/writable mounts, secret-free saved images, graceful tool completion on shutdown, missing mounts, and retention after a failed snapshot. The test orchestrator runs on the host; every agent fixture runs through the launcher inside Docker.
