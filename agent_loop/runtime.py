"""The execution runtime: the only module in this codebase that touches a filesystem
or spawns a process.

There is one runtime and one topology. The agent process is itself inside a
container, so when a tool runs `ls`, that `ls` is already in the box - there is no
boundary left to cross and nothing to proxy through. The container is created by the
launcher (`run.sh`) before Python starts; the agent cannot create the box it is
standing in.

That is why this class is named for *where it runs*, not for what it drives: it
contains no Docker code at all. Its guarantee is negative and structural - if this
process is not inside a container, `setup()` refuses to give you a runtime.

Two transfer surfaces, kept separate because the launcher may not always be Docker:

  read/write  - one small file, per tool call, inside the loop
  put/get     - whole trees, at setup/teardown, batched into one call

Paths are always relative to the runtime's root and are resolved here, never by the
caller. That containment check is why `open()` and `subprocess` must not appear
anywhere outside this file - `./test.sh lint` proves it.
"""

import os
import pwd
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


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


# The environment a model-written command sees. An allowlist, never inheritance: the
# agent process legitimately holds API keys for its tools, and no shell command it runs
# may inherit them.
DEFAULT_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ")

# The allowlist alone is not a wall: a command running as the same user could read the
# keys straight out of /proc/1/environ. So model-written commands drop to this user,
# which the image creates. Our own tool code stays root and keeps the keys.
SANDBOX_USER = "sandbox"

CONTAINER_ROOT = "/work"

# Set to "1" to run outside a container anyway. It removes the only boundary there is,
# so it exists for debugging the runtime itself and for nothing else.
UNSAFE_HOST = "AGENT_UNSAFE_HOST"


class DockerRuntime:
	"""What a tool is allowed to ask of the outside world, from inside the container.

	Ordering contract for list_dir, so tool output is stable:
	  recursive=False -> immediate children; directories first (alphabetical),
	                     then files (alphabetical)
	  recursive=True  -> files only, depth-first, sorted at each level
	Names in `skip` are pruned by basename in both modes.
	"""

	def __init__(self, root: str | Path = CONTAINER_ROOT,
	             env_allowlist: Iterable[str] = DEFAULT_ENV_ALLOWLIST):
		self.root = Path(root).resolve()
		self._env_allowlist = tuple(env_allowlist)
		# On a host with AGENT_UNSAFE_HOST there is no such user and no privilege to drop.
		try:
			self._sandbox: pwd.struct_passwd | None = pwd.getpwnam(SANDBOX_USER)
		except KeyError:
			self._sandbox = None

	# -- lifecycle ----------------------------------------------------------

	def setup(self) -> None:
		"""Create the workspace, refusing to exist outside a container.

		This is the structural half of "the agent process is never outside". The other
		half is `run.sh`, which is the only thing that starts one.
		"""
		if not Path("/.dockerenv").exists() and os.getenv(UNSAFE_HOST) != "1":
			raise RuntimeError(
				"agent-loop runs inside a container, and this process is not in one. "
				"Launch it with ./run.sh. To override deliberately, set "
				f"{UNSAFE_HOST}=1 - that removes the only boundary there is."
			)
		self.root.mkdir(parents=True, exist_ok=True)
		os.umask(0o002)  # what the agent writes, the sandbox user can edit (shared group)
		if not (self.root / ".git").exists():
			# A baseline commit so snapshot() always has a parent to hang off. /work is
			# disposable, which is what makes reset() safe here and unsafe anywhere else.
			r = self.run(
				"git init -q -b main . && git config user.email agent@localhost"
				" && git config user.name agent-loop && git commit -q --allow-empty -m baseline"
			)
			if not r.ok:
				raise RuntimeError(f"could not initialise {self.root}: {r.stderr.strip()}")

	def teardown(self) -> None:
		"""No-op: the container's death is the teardown, and it takes /work with it."""

	def __enter__(self) -> "DockerRuntime":
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
		"""Normalize `path` and raise PermissionError if it escapes the root. No I/O."""
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
		if self._sandbox:
			env["HOME"] = self._sandbox.pw_dir  # not root's; the command cannot read it anyway
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
			# Drop to the sandbox user: uid, gid, and no supplementary groups. This needs
			# CAP_SETUID/SETGID, which run.sh keeps; no-new-privileges blocks the way back.
			user=self._sandbox.pw_uid if self._sandbox else None,
			group=self._sandbox.pw_gid if self._sandbox else None,
			extra_groups=[] if self._sandbox else None,
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
		except ProcessLookupError:
			pass  # it finished between the timeout firing and the signal: the outcome we wanted
		# PermissionError is deliberately not caught. The group belongs to the sandbox user,
		# and signalling it needs CAP_KILL; if that is missing, a timeout would silently
		# become "wait for the command to finish", which is worse than failing loudly.

	# -- filesystem ---------------------------------------------------------

	def stat(self, path: str) -> FileStat | None:
		"""FileStat for `path`, or None if it does not exist."""
		p = self._resolve(path)
		return self._stat(p) if p.exists() else None

	def list_dir(self, path: str = ".", *, recursive: bool = False,
	             skip: Iterable[str] = (), max_entries: int = 200) -> list[FileStat]:
		"""At most `max_entries` entries. Ask for one more than you need to detect truncation."""
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
		"""Write `path`, creating parent directories."""
		p = self._resolve(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		if isinstance(content, str):
			p.write_text(content, encoding="utf-8")
		else:
			p.write_bytes(content)

	def remove(self, path: str) -> None:
		"""Delete a file or tree. Missing paths are not an error."""
		p = self._resolve(path)
		if p == self.root:
			raise PermissionError("refusing to remove the runtime root")
		if p.is_dir():
			shutil.rmtree(p, ignore_errors=True)
		else:
			p.unlink(missing_ok=True)

	# -- bulk transfer ------------------------------------------------------

	def put(self, files: Mapping[str, str | bytes]) -> None:
		"""Bulk write. Kept separate from write() because a future launcher may have to
		ship these across a boundary, and then the batching is the difference between one
		round trip and len(files) of them."""
		for rel, content in files.items():
			self.write(rel, content)

	def get(self, paths: Iterable[str]) -> dict[str, bytes]:
		"""Bulk read. See put()."""
		return {p: self.read_bytes(p) for p in paths}

	# -- state --------------------------------------------------------------

	def snapshot(self) -> str:
		"""Commit the workspace and return the sha. Cheap: git is already in the image."""
		r = self.run("git add -A && git commit-tree $(git write-tree) -p HEAD -m snapshot")
		if not r.ok:
			raise RuntimeError(f"snapshot failed: {r.stderr.strip()}")
		return r.stdout.strip()

	def reset(self, snapshot: str | None = None) -> None:
		"""Roll /work back to `snapshot`, or to the baseline commit from setup().

		Safe here only because /work is disposable. The same call against a developer's
		real repository would discard uncommitted work, which is why the agent process
		being inside the container is what makes this method possible at all.
		"""
		target = snapshot or "main"
		r = self.run(f"git reset -q --hard {target} && git clean -qfdx")
		if not r.ok:
			raise RuntimeError(f"reset failed: {r.stderr.strip()}")
