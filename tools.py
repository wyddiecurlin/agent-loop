"""Tool definitions and the registry the agent loop dispatches through.

Tool names use underscores (fs_list, shell_run, ...) because the Responses API only
allows [a-zA-Z0-9_-] in function names.

All filesystem tools are confined to ROOT (the working directory at import time);
paths that resolve outside it are rejected.
"""

import fnmatch
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
	id: str  # call id; echoed back to the model alongside the tool's result
	name: str  # tool name, looked up in the ToolRegistry
	arguments: str  # JSON-encoded arguments; json.loads() before calling the tool


@dataclass
class ToolResult:
	"""Structured outcome of one tool execution.

	ok        - False if the tool failed (bad args, exception, non-zero exit, ...).
	output    - Text for the model. On failure this is the error description.
	metadata  - Always has exit_code, duration_ms, truncated; tools may add more.
	"""
	ok: bool
	output: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_model_output(self) -> str:
		"""String to put in FunctionCallOutputItem.output."""
		return self.output if self.ok else f"error: {self.output}"


MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
	if len(text) <= limit:
		return text, False
	return text[:limit] + f"\n... [truncated {len(text) - limit} chars]", True


def _serialize(value: Any) -> str:
	if isinstance(value, str):
		return value
	try:
		return json.dumps(value)
	except TypeError:
		return str(value)


@dataclass
class Tool:
	"""One callable the model may invoke, plus the metadata the model needs to pick it."""
	name: str
	description: str
	input_schema: dict  # JSON Schema for the arguments object
	execute: Callable[..., Any]  # called with the model's arguments as kwargs; may return a ToolResult or any value

	def schema(self) -> dict:
		"""Responses API function-tool format."""
		return {
			"type": "function",
			"name": self.name,
			"description": self.description,
			"parameters": self.input_schema,
		}


class ToolRegistry:
	"""Name -> Tool map. Produces the schema list for the model and dispatches its tool calls."""

	def __init__(self, tools: list[Tool] | None = None):
		self._tools: dict[str, Tool] = {}
		for tool in tools or []:
			self.register(tool)

	def register(self, tool: Tool) -> Tool:
		if tool.name in self._tools:
			raise ValueError(f"tool {tool.name!r} is already registered")
		self._tools[tool.name] = tool
		return tool

	def get(self, name: str) -> Tool | None:
		return self._tools.get(name)

	def schema(self) -> list[dict]:
		"""What to pass as `tools` to the provider."""
		return [tool.schema() for tool in self._tools.values()]

	def execute(self, call: ToolCall) -> ToolResult:
		"""Run the tool named in `call`. Never raises: failures come back as ok=False.

		A tool may return a ToolResult directly (to attach its own metadata) or any
		plain value, which is serialized to a string. Either way the result gets
		exit_code / duration_ms / truncated filled in and its output capped.
		"""
		started = time.perf_counter()

		def finish(ok: bool, output: str, meta: dict[str, Any] | None = None) -> ToolResult:
			output, truncated = _truncate(output)
			metadata: dict[str, Any] = {"exit_code": 0 if ok else 1}
			metadata.update(meta or {})
			metadata["duration_ms"] = round((time.perf_counter() - started) * 1000)
			metadata["truncated"] = truncated or bool(metadata.get("truncated"))
			return ToolResult(ok=ok, output=output, metadata=metadata)

		tool = self._tools.get(call.name)
		if tool is None:
			return finish(False, f"unknown tool {call.name!r}; available tools: {sorted(self._tools)}")

		try:
			args = json.loads(call.arguments) if call.arguments else {}
		except json.JSONDecodeError as exc:
			return finish(False, f"arguments for {call.name!r} are not valid JSON: {exc}")
		if not isinstance(args, dict):
			return finish(False, f"arguments for {call.name!r} must be a JSON object, got {type(args).__name__}")

		try:
			result = tool.execute(**args)
		except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
			return finish(False, f"{call.name} raised {type(exc).__name__}: {exc}", {"exception": type(exc).__name__})

		if isinstance(result, ToolResult):
			return finish(result.ok, result.output, result.metadata)
		return finish(True, _serialize(result))

	def __contains__(self, name: str) -> bool:
		return name in self._tools

	def __len__(self) -> int:
		return len(self._tools)


# ---------------------------------------------------------------------------
# Filesystem / shell helpers
# ---------------------------------------------------------------------------

ROOT = Path.cwd().resolve()
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
MAX_READ_BYTES = 100_000


def _resolve(path: str) -> Path:
	"""Resolve `path` relative to ROOT and refuse anything that escapes it."""
	p = (ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
	if p != ROOT and ROOT not in p.parents:
		raise PermissionError(f"{path!r} is outside the working directory {ROOT}")
	return p


def _rel(p: Path) -> str:
	try:
		return str(p.relative_to(ROOT)) or "."
	except ValueError:
		return str(p)


def _is_text_file(p: Path) -> bool:
	try:
		with p.open("rb") as f:
			return b"\x00" not in f.read(2048)
	except OSError:
		return False


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def multiply(**params) -> int:
	res = 1
	for _, v in params.items():
		res *= v
	return res


def get_today_date() -> dict:
	from datetime import datetime

	today = datetime.now()
	return {"month": today.month, "day": today.day, "year": today.year}


def substract(a: int, b: int) -> int:
	return a - b


def fs_list(path: str = ".", recursive: bool = False, max_entries: int = 200) -> ToolResult:
	root = _resolve(path)
	if not root.exists():
		raise FileNotFoundError(f"{path!r} does not exist")
	if root.is_file():
		entry = {"path": _rel(root), "type": "file", "size": root.stat().st_size}
		return ToolResult(ok=True, output=json.dumps([entry]), metadata={"count": 1})

	entries: list[dict] = []
	truncated = False
	if recursive:
		for dirpath, dirnames, filenames in os.walk(root):
			dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
			for name in sorted(filenames):
				p = Path(dirpath) / name
				entries.append({"path": _rel(p), "type": "file", "size": p.stat().st_size})
				if len(entries) >= max_entries:
					truncated = True
					break
			if truncated:
				break
	else:
		for p in sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name)):
			if p.name in SKIP_DIRS:
				continue
			if p.is_dir():
				entries.append({"path": _rel(p) + "/", "type": "dir"})
			else:
				entries.append({"path": _rel(p), "type": "file", "size": p.stat().st_size})
			if len(entries) >= max_entries:
				truncated = True
				break

	output = json.dumps(entries)
	if truncated:
		output += f"\n[stopped after {max_entries} entries; raise max_entries or narrow path]"
	return ToolResult(ok=True, output=output, metadata={"path": _rel(root), "count": len(entries), "truncated": truncated})


def fs_read(path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
	p = _resolve(path)
	if not p.is_file():
		raise FileNotFoundError(f"{path!r} is not a file")
	if not _is_text_file(p):
		raise ValueError(f"{path!r} looks binary; refusing to read")

	data = p.read_bytes()
	file_truncated = len(data) > MAX_READ_BYTES
	text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
	lines = text.splitlines()

	start = max(start_line or 1, 1)
	end = min(end_line or len(lines), len(lines))
	body = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(start, end + 1))
	if file_truncated and end == len(lines):
		body += f"\n... [file truncated at {MAX_READ_BYTES} bytes]"
	return ToolResult(
		ok=True,
		output=body,
		metadata={
			"path": _rel(p),
			"total_lines": len(lines),
			"start_line": start,
			"end_line": end,
			"truncated": file_truncated,
		},
	)


def fs_search(
	pattern: str,
	path: str = ".",
	glob: str | None = None,
	regex: bool = True,
	case_sensitive: bool = False,
	max_results: int = 100,
) -> ToolResult:
	root = _resolve(path)
	if not root.exists():
		raise FileNotFoundError(f"{path!r} does not exist")

	flags = 0 if case_sensitive else re.IGNORECASE
	rx = re.compile(pattern if regex else re.escape(pattern), flags)

	files = [root] if root.is_file() else []
	if root.is_dir():
		for dirpath, dirnames, filenames in os.walk(root):
			dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
			for name in sorted(filenames):
				if glob and not fnmatch.fnmatch(name, glob):
					continue
				files.append(Path(dirpath) / name)

	matches: list[str] = []
	truncated = False
	files_scanned = 0
	for f in files:
		if truncated:
			break
		if not _is_text_file(f):
			continue
		files_scanned += 1
		try:
			with f.open("r", encoding="utf-8", errors="replace") as fh:
				for lineno, line in enumerate(fh, 1):
					if rx.search(line):
						matches.append(f"{_rel(f)}:{lineno}: {line.rstrip(chr(10))[:300]}")
						if len(matches) >= max_results:
							truncated = True
							break
		except OSError:
			continue

	output = "\n".join(matches) if matches else "no matches"
	if truncated:
		output += f"\n[stopped after {max_results} matches; raise max_results or narrow the search]"
	return ToolResult(
		ok=True,
		output=output,
		metadata={"matches": len(matches), "files_scanned": files_scanned, "truncated": truncated},
	)


def fs_patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
	p = _resolve(path)

	if not p.exists():
		if old_string != "":
			raise FileNotFoundError(f"{path!r} does not exist; pass old_string='' to create it")
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(new_string, encoding="utf-8")
		return ToolResult(
			ok=True,
			output=f"created {_rel(p)}",
			metadata={"path": _rel(p), "created": True, "bytes": len(new_string.encode())},
		)

	if not p.is_file():
		raise IsADirectoryError(f"{path!r} is not a file")
	if old_string == "":
		raise ValueError("old_string is empty but the file exists; give the exact text to replace")

	text = p.read_text(encoding="utf-8")
	count = text.count(old_string)
	if count == 0:
		raise ValueError(f"old_string not found in {path!r}; it must match the file exactly (including whitespace)")
	if count > 1 and not replace_all:
		raise ValueError(f"old_string matches {count} places in {path!r}; add more context or set replace_all=true")

	replacements = count if replace_all else 1
	new_text = text.replace(old_string, new_string, replacements)
	p.write_text(new_text, encoding="utf-8")
	return ToolResult(
		ok=True,
		output=f"patched {_rel(p)} ({replacements} replacement{'s' if replacements != 1 else ''})",
		metadata={"path": _rel(p), "replacements": replacements},
	)


def shell_run(command: str, cwd: str | None = None, timeout_s: float = 30.0) -> ToolResult:
	workdir = _resolve(cwd) if cwd else ROOT
	try:
		proc = subprocess.run(
			command,
			shell=True,
			cwd=workdir,
			capture_output=True,
			text=True,
			timeout=timeout_s,
		)
	except subprocess.TimeoutExpired as exc:
		stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
		stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
		return ToolResult(
			ok=False,
			output=f"command timed out after {timeout_s}s\nstdout:\n{stdout}\nstderr:\n{stderr}",
			metadata={"exit_code": None, "timed_out": True, "timeout_s": timeout_s},
		)

	parts = [f"exit_code: {proc.returncode}"]
	if proc.stdout:
		parts.append(f"stdout:\n{proc.stdout.rstrip()}")
	if proc.stderr:
		parts.append(f"stderr:\n{proc.stderr.rstrip()}")
	return ToolResult(
		ok=proc.returncode == 0,
		output="\n".join(parts),
		metadata={"exit_code": proc.returncode, "timed_out": False},
	)


# ---------------------------------------------------------------------------
# Task completion
# ---------------------------------------------------------------------------

DONE_TOOL = "done"


def done(answer: str) -> ToolResult:
	"""Terminal tool: the model calls it to submit its final answer, success or failure.

	agent_loop ends the run on the first ok=True result from this tool. A bad call (no
	`answer`, or a blank one) comes back ok=False like any other tool error, so the model
	gets the error and another turn instead of ending the task with nothing to say.
	"""
	if not answer.strip():
		raise ValueError("answer must be a non-empty string holding the full final reply")
	return ToolResult(ok=True, output=answer, metadata={"final": True})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY = ToolRegistry([
	Tool(
		name=DONE_TOOL,
		description=(
			"End the task and submit the final answer. This is the only way to finish, and it "
			"covers BOTH outcomes: if the task succeeded, `answer` is the complete reply for "
			"the user; if it cannot be completed, call this anyway and use `answer` to say what "
			"was tried and why it failed. `answer` must stand on its own - state the actual "
			"values, file names, and command output you found rather than referring back to "
			"earlier steps. Call it exactly once, by itself."
		),
		input_schema={
			"type": "object",
			"properties": {
				"answer": {"type": "string", "description": "The complete final answer, or the reason the task failed."},
			},
			"required": ["answer"],
		},
		execute=done,
	),
	Tool(
		name="multiply",
		description="Use this for all multiplication. Never compute products yourself.",
		input_schema={
			"type": "object",
			"properties": {
				"a": {"type": "number", "description": "First factor."},
				"b": {"type": "number", "description": "Second factor."},
				"c": {"type": "number", "description": "Optional third factor."},
			},
			"required": ["a", "b"],
		},
		execute=multiply,
	),
	Tool(
		name="get_today_date",
		description="Return today's date as month, day, and year.",
		input_schema={"type": "object", "properties": {}},
		execute=get_today_date,
	),
	Tool(
		name="substract",
		description="Use this for all substraction. Never compute yourself.",
		input_schema={
			"type": "object",
			"properties": {
				"a": {"type": "integer", "description": "Minuend."},
				"b": {"type": "integer", "description": "Subtrahend."},
			},
			"required": ["a", "b"],
		},
		execute=substract,
	),
	Tool(
		name="fs_list",
		description=(
			"List files and directories under a path (relative to the working directory). "
			"Skips .git, node_modules, __pycache__, and virtualenvs."
		),
		input_schema={
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Directory to list. Defaults to '.'."},
				"recursive": {"type": "boolean", "description": "Walk subdirectories. Defaults to false."},
				"max_entries": {"type": "integer", "description": "Cap on returned entries. Defaults to 200."},
			},
		},
		execute=fs_list,
	),
	Tool(
		name="fs_read",
		description=(
			"Read a text file and return its contents with line numbers. "
			"Use start_line/end_line to read a slice of a large file."
		),
		input_schema={
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path relative to the working directory."},
				"start_line": {"type": "integer", "description": "First line to return (1-based, inclusive)."},
				"end_line": {"type": "integer", "description": "Last line to return (1-based, inclusive)."},
			},
			"required": ["path"],
		},
		execute=fs_read,
	),
	Tool(
		name="fs_search",
		description=(
			"Search file contents for a pattern (grep). Returns matching lines as file:line: text."
		),
		input_schema={
			"type": "object",
			"properties": {
				"pattern": {"type": "string", "description": "Regex (default) or literal text to find."},
				"path": {"type": "string", "description": "File or directory to search. Defaults to '.'."},
				"glob": {"type": "string", "description": "Only search files whose name matches, e.g. '*.py'."},
				"regex": {"type": "boolean", "description": "Treat pattern as a regex. Defaults to true."},
				"case_sensitive": {"type": "boolean", "description": "Defaults to false."},
				"max_results": {"type": "integer", "description": "Cap on returned matches. Defaults to 100."},
			},
			"required": ["pattern"],
		},
		execute=fs_search,
	),
	Tool(
		name="fs_patch",
		description=(
			"Edit a file by replacing an exact string. old_string must appear exactly once unless "
			"replace_all is true. To create a new file, pass old_string='' and the full contents as new_string. "
			"Read the file first so old_string matches exactly."
		),
		input_schema={
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path relative to the working directory."},
				"old_string": {"type": "string", "description": "Exact text to replace ('' to create a new file)."},
				"new_string": {"type": "string", "description": "Replacement text."},
				"replace_all": {"type": "boolean", "description": "Replace every occurrence. Defaults to false."},
			},
			"required": ["path", "old_string", "new_string"],
		},
		execute=fs_patch,
	),
	Tool(
		name="shell_run",
		description=(
			"Run a shell command in the working directory and return exit code, stdout, and stderr. "
			"Output is truncated; use timeout_s for long-running commands."
		),
		input_schema={
			"type": "object",
			"properties": {
				"command": {"type": "string", "description": "The command line to run via the shell."},
				"cwd": {"type": "string", "description": "Subdirectory to run in. Defaults to the working directory."},
				"timeout_s": {"type": "number", "description": "Kill the command after this many seconds. Defaults to 30."},
			},
			"required": ["command"],
		},
		execute=shell_run,
	),
])
