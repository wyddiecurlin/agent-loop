"""Local session lifecycle and the narrow runtime control socket (stdlib only)."""

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
CONTAINER_CONTROL = "/run/agent-loop/private/live/control.sock"
CONFIG_KEYS = frozenset("PROVIDER FALLBACK MODEL REASONING_EFFORT MAX_OUTPUT_TOKENS CONTEXT_WINDOW_TOKENS OPENAI_MODEL QWEN_MODEL QWEN_BASE_URL QWEN_THINKING QWEN_TEMPERATURE QWEN_SEED WEB_SEARCH_PROVIDER PARALLEL_SEARCH_MODE TZ".split())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object, mode: int = 0o600) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        os.chmod(temporary, mode)
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def docker(*args: str, capture: bool = True) -> str:
    result = subprocess.run(["docker", *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return (result.stdout or "").strip()


def valid_id(value: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError("Invalid session ID")
    return value


def read_grant(path: Path) -> dict:
    try:
        with path.open() as handle:
            value = json.loads(handle.read(32768))
        if not isinstance(value, dict) or not isinstance(value.get("session_id"), str):
            raise ValueError()
        valid_id(value["session_id"])
        if not isinstance(value.get("token"), str) or not re.fullmatch(r"ivon_container_[A-Za-z0-9_-]{43}", value["token"]):
            raise ValueError()
        return value
    except (ValueError, TypeError, KeyError):
        raise ValueError("Invalid container grant file") from None


def backend(url: str, grant: dict, path: str, method: str = "GET") -> dict:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credential backend requires a plain HTTPS URL")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None
    request = urllib.request.Request(url.rstrip("/") + path, method=method, headers={
        "Authorization": "Bearer " + grant["token"], "X-Session-ID": grant["session_id"],
    })
    try:
        with urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({})).open(request, timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, ValueError):
        raise RuntimeError("Credential backend denied access or is unavailable") from None


def mount_spec(host_path: str, read_only: bool, protected: list[Path]) -> dict:
    path = Path(host_path).expanduser().resolve(strict=True)
    if not path.is_dir() or "," in str(path):
        raise ValueError("Mount source must be a directory with no comma in its path")
    for secret in protected + [Path("/proc"), Path("/sys"), Path("/dev"), Path("/run"), Path("/var/run").resolve()]:
        if path == secret or path in secret.parents or secret in path.parents:
            raise ValueError("Mount source overlaps protected launcher or system data")
    return {"host_path": str(path), "container_path": "/mnt/volumes/" + uuid.uuid4().hex[:12], "read_only": read_only}


class Session:
    def __init__(self, directory: Path, metadata: dict, grant_file: Path | None, backend_url: str | None):
        self.directory, self.metadata = directory, metadata
        self.grant_file, self.backend_url = grant_file, backend_url
        self.mounts = list(metadata.get("mounts", []))
        self.pending_mount = False
        self.protected = [directory.parent.parent] + ([grant_file] if grant_file else [])
        self.conversation = {"messages": [], "continue": False, "interactive": False}
        if metadata.get("conversation"):
            self.conversation = json.loads((directory / metadata["conversation"]).read_text())
        self.control_dir = directory / "live"
        self.control_dir.mkdir(exist_ok=True, mode=0o755)
        self.credentials_dir = self.control_dir / "credentials"
        self.credentials_dir.mkdir(exist_ok=True, mode=0o755)
        self._lock = threading.Lock()

    def refresh_grant(self) -> None:
        if self.grant_file:
            value = read_grant(self.grant_file)
            if value["session_id"] != self.metadata["session_id"]:
                raise ValueError("Replacement grant belongs to another session")
            atomic_json(self.credentials_dir / "grant.json", value, 0o644)

    def dispatch(self, request: dict) -> object:
        with self._lock:
            op = request.get("op")
            if op == "info":
                return {"mounts": self.mounts, "conversation": self.conversation}
            if op == "refresh_grant":
                self.refresh_grant()
                return {}
            if op == "checkpoint":
                state = request["state"]
                if not isinstance(state, dict) or not isinstance(state.get("messages"), list):
                    raise ValueError("Invalid conversation checkpoint")
                atomic_json(self.directory / "pending-conversation.json", state)
                self.conversation = state
                return {}
            if op == "mount":
                if not isinstance(request.get("host_path"), str) or type(request.get("read_only")) is not bool:
                    raise ValueError("Invalid mount request")
                mount = mount_spec(request["host_path"], request["read_only"], self.protected)
                for existing in self.mounts:
                    if existing["host_path"] == mount["host_path"]:
                        if existing["read_only"] != mount["read_only"]:
                            raise ValueError("Mount already exists with different permissions")
                        return {"path": existing["container_path"], "pending": self.pending_mount}
                self.mounts.append(mount)
                self.pending_mount = True
                return {"path": mount["container_path"], "pending": True}
            raise ValueError("Unknown launcher operation")

    def save(self, container: str, config: dict) -> str:
        if docker("inspect", "--format", "{{.State.Running}}", container) != "false":
            raise RuntimeError("Container still running; refusing to snapshot active writers")
        image = docker("commit", container)
        generation = uuid.uuid4().hex
        archive = self.directory / (generation + ".tar")
        temporary = archive.with_suffix(".tar.tmp")
        docker("image", "save", "--output", str(temporary), image)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(archive)
        conversation = generation + ".json"
        atomic_json(self.directory / conversation, self.conversation)
        previous = self.metadata
        updated = {**previous, "image_id": image, "archive": archive.name, "conversation": conversation,
                   "mounts": self.mounts, "config": config, "saved_at": timestamp()}
        atomic_json(self.directory / "metadata.json", updated)
        self.metadata = updated
        # The published metadata now references both durable artifacts. Removal is safe only here.
        docker("rm", container)
        for field in ("archive", "conversation"):
            if previous.get(field) and previous[field] != updated[field]:
                try:
                    (self.directory / previous[field]).unlink(missing_ok=True)
                except OSError:
                    print("Previous snapshot retained; cleanup failed.", file=sys.stderr)
        return image


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(64 * 1024 * 1024 + 1)
            if len(line) > 64 * 1024 * 1024:
                raise ValueError("Control request too large")
            result = self.server.session.dispatch(json.loads(line))
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": str(error) if isinstance(error, ValueError) else "Launcher operation failed"}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class ControlServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def environment() -> tuple[dict[str, str], Path | None]:
    file = Path(os.environ.get("AGENT_ENV_FILE", str(ROOT / ".env"))).resolve()
    values = {}
    if file.is_file():
        for line in file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                if name in CONFIG_KEYS:
                    values[name] = value
    values.update({name: os.environ[name] for name in CONFIG_KEYS if name in os.environ})
    return values, file if file.is_file() else None


def launch(args: argparse.Namespace) -> int:
    if os.getuid() == 0:
        raise ValueError("Launch as your normal host user so model commands cannot run as root")
    state = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "agent-loop"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    sessions = state / "sessions"
    sessions.mkdir(exist_ok=True, mode=0o700)
    if args.list:
        for path in sorted(sessions.glob("*/metadata.json")):
            metadata = json.loads(path.read_text())
            print(metadata["session_id"], metadata.get("saved_at", "unsaved"))
        return 0
    grant_file = Path(args.grant_file).expanduser().resolve(strict=True) if args.grant_file else None
    grant = read_grant(grant_file) if grant_file else None
    if bool(grant_file) != bool(args.backend_url):
        raise ValueError("--grant-file and --backend-url must be supplied together")
    if not grant and not args.local:
        raise ValueError("Supply --grant-file and --backend-url, or use --local for development")
    owner = "local:" + str(os.getuid())
    if grant:
        owner = backend(args.backend_url, grant, "/settings")["user_id"]
    session_id = valid_id(args.resume) if args.resume else grant["session_id"] if grant else str(uuid.uuid4())
    if grant and grant["session_id"] != session_id:
        raise ValueError("Grant belongs to another session")
    directory = sessions / session_id
    if args.resume and not (directory / "metadata.json").is_file():
        raise ValueError("Saved session not found")
    if not args.resume and directory.exists():
        raise ValueError("Session already exists; use --resume")
    directory.mkdir(exist_ok=True, mode=0o700)
    with (directory / "lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Session is already in use") from None
        if (directory / "active.json").exists():
            retained = json.loads((directory / "active.json").read_text())["container_id"]
            raise RuntimeError(f"Session has a retained container {retained}; recover it before resuming")
        metadata = json.loads((directory / "metadata.json").read_text()) if args.resume else {
            "session_id": session_id, "owner": owner, "created_at": timestamp(), "mounts": [], "config": {}}
        if metadata["owner"] != owner:
            raise ValueError("Session belongs to another owner")
        session = Session(directory, metadata, grant_file, args.backend_url)
        config, env_file = environment()
        config = {**metadata["config"], **config}
        protected = [state] + ([grant_file] if grant_file else []) + ([env_file] if env_file else [])
        session.protected = protected
        for path, readonly in [(p, True) for p in args.mount] + [(p, False) for p in args.mount_rw]:
            session.mounts.append(mount_spec(path, readonly, protected))
        for mount in session.mounts:
            # A saved path may have become a symlink or disappeared since the last run.
            checked = mount_spec(mount["host_path"], mount["read_only"], protected)
            if checked["host_path"] != mount["host_path"]:
                raise ValueError("Saved mount source changed")
        session.refresh_grant()
        if env_file and not grant:
            shutil.copyfile(env_file, session.credentials_dir / "environment")
            os.chmod(session.credentials_dir / "environment", 0o644)
        socket_path = session.control_dir / "control.sock"
        socket_path.unlink(missing_ok=True)
        server = ControlServer(str(socket_path), ControlHandler)
        server.session = session
        os.chmod(socket_path, 0o666)  # Host state/ and container /run/agent-loop/ are private.
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        container = None
        try:
            if args.resume:
                docker("image", "load", "--input", str(directory / metadata["archive"]))
                image = metadata["image_id"]
            else:
                target = os.environ.get("AGENT_TARGET", "dev")
                image = docker("build", "--quiet", "--target", target, str(ROOT))
            print(f"Session {session_id}", file=sys.stderr)
            while True:
                session.pending_mount = False
                opts = ["create", "--interactive", "--init", "--memory", "2g", "--memory-swap", "2g", "--cpus", "2", "--pids-limit", "256",
                        "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID", "--cap-add", "KILL", "--security-opt", "no-new-privileges",
                        "--mount", f"type=bind,src={session.control_dir},dst=/run/agent-loop/private/live,readonly",
                        "--env", "AGENT_CONTROL_SOCKET=" + CONTAINER_CONTROL, "--env", "AGENT_SESSION_ID=" + session_id,
                        "--env", "AGENT_HOST_UID=" + str(os.getuid()), "--env", "AGENT_HOST_GID=" + str(os.getgid())]
                if sys.stdin.isatty() and sys.stdout.isatty():
                    opts.append("--tty")
                for name, value in config.items():
                    opts.extend(["--env", name + "=" + value])
                if grant:
                    opts.extend(["--env", "AGENT_CREDENTIAL_URL=" + args.backend_url])
                for mount in session.mounts:
                    checked = mount_spec(mount["host_path"], mount["read_only"], protected)
                    if checked["host_path"] != mount["host_path"]:
                        raise ValueError("Mount source changed")
                    value = f"type=bind,src={mount['host_path']},dst={mount['container_path']}"
                    opts.extend(["--mount", value + (",readonly" if mount["read_only"] else "")])
                if os.environ.get("AGENT_ENTRYPOINT"):
                    opts.extend(["--entrypoint", os.environ["AGENT_ENTRYPOINT"]])
                container = docker(*opts, image, *args.task)
                atomic_json(directory / "active.json", {"container_id": container})
                process = subprocess.Popen(["docker", "start", "--attach", "--interactive", container])
                previous_handlers = {}
                def stop(signum, frame):
                    if process.poll() is None:
                        subprocess.run(["docker", "kill", "--signal", "SIGTERM", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[sig] = signal.signal(sig, stop)
                try:
                    code = process.wait()
                    image = session.save(container, config)
                finally:
                    for sig, handler in previous_handlers.items():
                        signal.signal(sig, handler)
                container = None
                (directory / "active.json").unlink(missing_ok=True)
                if code == 75 and session.pending_mount:
                    continue
                if session.pending_mount:
                    raise RuntimeError("Mount was requested but the agent did not checkpoint for restart")
                local = " --local" if not grant else " --grant-file <fresh-grant.json> --backend-url <url>"
                print(f"Saved session {session_id}. Resume with agent-loop --resume {session_id}{local}", file=sys.stderr)
                return code
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            if grant_file:
                try:
                    current = read_grant(grant_file)
                    backend(args.backend_url, current, "/sessions/" + session_id, "DELETE")
                except Exception:
                    print("Could not revoke container grant; it remains bounded by its expiry.", file=sys.stderr)
            shutil.rmtree(session.control_dir)
            if container:
                print(f"Container {container} retained for recovery; session teardown did not complete.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent-loop in a persistent local container session")
    parser.add_argument("--resume")
    parser.add_argument("--local", action="store_true", help="Use the local development .env instead of a user grant")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--grant-file", default=os.getenv("AGENT_GRANT_FILE"))
    parser.add_argument("--backend-url", default=os.getenv("AGENT_CREDENTIAL_URL"))
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--mount-rw", action="append", default=[])
    parser.add_argument("task", nargs=argparse.REMAINDER)
    try:
        args = parser.parse_args()
        if args.task[:1] == ["--"]:
            args.task = args.task[1:]
        return launch(args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"agent-loop: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
