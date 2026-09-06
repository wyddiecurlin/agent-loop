"""Execution runtimes: the only module in this codebase that touches a filesystem
or spawns a process.

Every tool reaches the outside world through a Runtime, so swapping LocalRuntime
for DockerRuntime (build step 4) or a remote provider changes nothing above this
module. See RUNTIME.md.

Two transfer surfaces, deliberately separate because they have different costs:

  read/write  - one small file, per tool call, inside the loop
  put/get     - whole trees, at setup/teardown, batched into one round trip

Never loop write() over many files. On a remote runtime that is one network
round trip each; put() takes a dict and sends them together.

Paths are always relative to the runtime's root and are resolved by the runtime,
never by the caller. That containment check is the reason `open()` and
`subprocess` must not appear anywhere outside this file.
"""

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
	"""Outcome of one command. A struct, not a string - the tool layer does the formatting.

	exit_code is None when the command timed out (there is no exit status to report).
	"""
	exit_code: int | None
	stdout: str
	stderr: str
	timed_out: bool
	duration_s: float

	@property
	def ok(self) -> bool:
		return self.exit_code == 0


@dataclass
class FileStat:
	path: str  # normalized, relative to the runtime root
	is_dir: bool
	size: int  # 0 for directories


# Environment the sandbox sees. An allowlist, never inheritance: os.environ holds
# OPENAI_API_KEY after load_dotenv(), and a subprocess that inherits it can ship it
# anywhere. A container starts with roughly this set, so LocalRuntime matching it
# keeps the two implementations honest.
DEFAULT_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Runtime(Protocol):
	"""What a tool is allowed to ask of the outside world.

	Ordering contract for list_dir, which implementations must match so tool output
	does not shift when the runtime is swapped:
	  recursive=False -> immediate children; directories first (alphabetical),
	                     then files (alphabetical)
	  recursive=True  -> files only, depth-first, sorted at each level
	Names in `skip` are pruned by basename in both modes.
	"""

	def setup(self) -> None:
		"""Create the execution environment. Cheap for local, boots a container for Docker."""

	def teardown(self) -> None:
		"""Destroy it. Must be safe to call twice."""

	def relpath(self, path: str) -> str:
		"""Normalize `path` and raise PermissionError if it escapes the root. No I/O."""

	def run(self, command: str, *, cwd: str | None = None, timeout_s: float = 30.0,
	        env: Mapping[str, str] | None = None) -> RunResult: ...

	def stat(self, path: str) -> FileStat | None:
		"""FileStat for `path`, or None if it does not exist."""

	def list_dir(self, path: str = ".", *, recursive: bool = False,
	             skip: Iterable[str] = (), max_entries: int = 200) -> list[FileStat]:
		"""At most `max_entries` entries. Ask for one more than you need to detect truncation."""

	def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes: ...

	def read_text(self, path: str) -> str: ...

	def write(self, path: str, content: str | bytes) -> None:
		"""Write `path`, creating parent directories."""

	def remove(self, path: str) -> None:
		"""Delete a file or tree. Missing paths are not an error."""

	def put(self, files: Mapping[str, str | bytes]) -> None:
		"""Bulk upload. One round trip, however many files."""

	def get(self, paths: Iterable[str]) -> dict[str, bytes]:
		"""Bulk download. One round trip, however many files."""

	def snapshot(self) -> str: ...

	def reset(self, snapshot: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

class LocalRuntime:
	"""Runs in this process, confined to `root` by path checks.

	The path checks are argument hygiene, not a boundary: `run()` hands a string to a
	shell, and the shell is free to `cd /`. Whether that matters depends entirely on
	where this process is running (see RUNTIME.md, "Topology"):

	  CLI mode   - the agent is on a developer's machine, so nothing isolates it here.
	               Use DockerRuntime; LocalRuntime is a fallback for development only.
	  Fleet mode - the agent is itself PID 1 inside a disposable container, so the
	               boundary is already one level out and LocalRuntime is the correct
	               production choice. There is nothing left to contain in-process.

	Same class, opposite verdicts. The env allowlist and process-group kill below hold
	in both, so behavior does not shift when the topology does.
	"""

	def __init__(self, root: str | Path | None = None,
	             env_allowlist: Iterable[str] = DEFAULT_ENV_ALLOWLIST):
		self.root = Path(root or Path.cwd()).resolve()
		self._env_allowlist = tuple(env_allowlist)

	# -- lifecycle ----------------------------------------------------------

	def setup(self) -> None:
		self.root.mkdir(parents=True, exist_ok=True)

	def teardown(self) -> None:
		"""No-op: `root` is the developer's own directory and is not ours to delete."""

	def __enter__(self) -> "LocalRuntime":
		self.setup()
		return self

	def __exit__(self, *exc) -> None:
		self.teardown()

	# -- paths --------------------------------------------------------------

	def _resolve(self, path: str) -> Path:
		p = Path(path).resolve() if os.path.isabs(path) else (self.root / path).resolve()
		if p != self.root and self.root not in p.parents:
			raise PermissionError(f"{path!r} is outside the working directory {self.root}")
		return p

	def relpath(self, path: str) -> str:
		return self._rel(self._resolve(path))

	def _rel(self, p: Path) -> str:
		try:
			return str(p.relative_to(self.root)) or "."
		except ValueError:
			return str(p)

	def _stat(self, p: Path) -> FileStat:
		is_dir = p.is_dir()
		return FileStat(path=self._rel(p), is_dir=is_dir, size=0 if is_dir else p.stat().st_size)

	# -- execution ----------------------------------------------------------

	def _env(self, extra: Mapping[str, str] | None) -> dict[str, str]:
		env = {k: os.environ[k] for k in self._env_allowlist if k in os.environ}
		env.setdefault("PATH", os.defpath)
		env.update(extra or {})
		return env

	def run(self, command: str, *, cwd: str | None = None, timeout_s: float = 30.0,
	        env: Mapping[str, str] | None = None) -> RunResult:
		workdir = self._resolve(cwd) if cwd else self.root
		started = time.perf_counter()
		proc = subprocess.Popen(
			command,
			shell=True,
			cwd=workdir,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			env=self._env(env),
			# Own process group, so a timeout can kill grandchildren too. Without this,
			# subprocess's own timeout kills only the shell and leaves `npm test` running.
			start_new_session=True,
		)
		timed_out = False
		try:
			stdout, stderr = proc.communicate(timeout=timeout_s)
		except subprocess.TimeoutExpired:
			timed_out = True
			self._signal_group(proc.pid, signal.SIGTERM)
			try:
				stdout, stderr = proc.communicate(timeout=2.0)
			except subprocess.TimeoutExpired:
				self._signal_group(proc.pid, signal.SIGKILL)
				stdout, stderr = proc.communicate()

		return RunResult(
			exit_code=None if timed_out else proc.returncode,
			stdout=stdout or "",
			stderr=stderr or "",
			timed_out=timed_out,
			duration_s=time.perf_counter() - started,
		)

	@staticmethod
	def _signal_group(pid: int, sig: int) -> None:
		try:
			os.killpg(os.getpgid(pid), sig)
		except (ProcessLookupError, PermissionError):
			pass

	# -- filesystem ---------------------------------------------------------

	def stat(self, path: str) -> FileStat | None:
		p = self._resolve(path)
		return self._stat(p) if p.exists() else None

	def list_dir(self, path: str = ".", *, recursive: bool = False,
	             skip: Iterable[str] = (), max_entries: int = 200) -> list[FileStat]:
		root = self._resolve(path)
		if not root.is_dir():
			raise NotADirectoryError(f"{path!r} is not a directory")
		skip = set(skip)
		entries: list[FileStat] = []

		if recursive:
			for dirpath, dirnames, filenames in os.walk(root):
				dirnames[:] = sorted(d for d in dirnames if d not in skip)
				for name in sorted(filenames):
					if name in skip:
						continue
					entries.append(self._stat(Path(dirpath) / name))
					if len(entries) >= max_entries:
						return entries
		else:
			for p in sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name)):
				if p.name in skip:
					continue
				entries.append(self._stat(p))
				if len(entries) >= max_entries:
					return entries
		return entries

	def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
		p = self._resolve(path)
		if not p.is_file():
			raise FileNotFoundError(f"{path!r} is not a file")
		with p.open("rb") as f:
			return f.read() if max_bytes is None else f.read(max_bytes)

	def read_text(self, path: str) -> str:
		return self.read_bytes(path).decode("utf-8", errors="replace")

	def write(self, path: str, content: str | bytes) -> None:
		p = self._resolve(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		if isinstance(content, str):
			p.write_text(content, encoding="utf-8")
		else:
			p.write_bytes(content)

	def remove(self, path: str) -> None:
		p = self._resolve(path)
		if p == self.root:
			raise PermissionError("refusing to remove the runtime root")
		if p.is_dir():
			shutil.rmtree(p, ignore_errors=True)
		else:
			p.unlink(missing_ok=True)

	# -- bulk transfer ------------------------------------------------------

	def put(self, files: Mapping[str, str | bytes]) -> None:
		for rel, content in files.items():
			self.write(rel, content)

	def get(self, paths: Iterable[str]) -> dict[str, bytes]:
		return {p: self.read_bytes(p) for p in paths}

	# -- state --------------------------------------------------------------

	def snapshot(self) -> str:
		raise NotImplementedError(
			"LocalRuntime does not checkpoint: its root is the developer's real repository, "
			"and reset() there would discard uncommitted work. Use DockerRuntime, whose /work "
			"is disposable."
		)

	def reset(self, snapshot: str | None = None) -> None:
		self.snapshot()  # raises with the explanation above


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def make_runtime(kind: str | None = None, *, root: str | Path | None = None) -> Runtime:
	"""Build the runtime named by `kind` (or AGENT_SANDBOX; default 'local').

	Fails loudly on an unavailable backend rather than falling back. A sandbox we
	think is on is worse than one we know is off.
	"""
	kind = (kind or os.getenv("AGENT_SANDBOX", "local")).lower()
	if kind == "local":
		return LocalRuntime(root)
	if kind == "docker":
		raise NotImplementedError(
			"DockerRuntime is build step 4; AGENT_SANDBOX=docker is not available yet"
		)
	raise ValueError(f"unknown AGENT_SANDBOX {kind!r} (expected 'local' or 'docker')")
