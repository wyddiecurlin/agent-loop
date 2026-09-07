"""Tool definitions and the registry the agent loop dispatches through.

Tool names use underscores (fs_list, shell_run, ...) because the Responses API only
allows [a-zA-Z0-9_-] in function names.

Tools reach the outside world ONLY through the Runtime they are handed by
build_registry(). Nothing in this module may spawn a process or reach a file on its
own, so that "the agent runs inside the sandbox" is a property of the code rather than
something we remember to arrange. The check that proves it, from the repo root:

	./test.sh lint
"""

import fnmatch
import json
import posixpath
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable

from .runtime import DockerRuntime


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
# Filesystem policy
# ---------------------------------------------------------------------------

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
MAX_READ_BYTES = 100_000
MAX_SEARCH_FILES = 5_000  # cap on files fetched for one fs_search; keeps the batch bounded


def _looks_binary(data: bytes) -> bool:
	return b"\x00" in data[:2048]


# ---------------------------------------------------------------------------
# Tool implementations
#
# Every fs_/shell_ tool takes the Runtime as its first argument; build_registry
# binds it. Nothing here reads or writes a file on its own.
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


def fs_list(runtime: DockerRuntime, path: str = ".", recursive: bool = False, max_entries: int = 200) -> ToolResult:
	st = runtime.stat(path)
	if st is None:
		raise FileNotFoundError(f"{path!r} does not exist")
	if not st.is_dir:
		entry = {"path": st.path, "type": "file", "size": st.size}
		return ToolResult(ok=True, output=json.dumps([entry]), metadata={"count": 1})

	# One extra entry so we can tell "exactly max_entries" from "there were more".
	found = runtime.list_dir(path, recursive=recursive, skip=SKIP_DIRS, max_entries=max_entries + 1)
	truncated = len(found) > max_entries
	found = found[:max_entries]

	entries = [
		{"path": f.path + "/", "type": "dir"} if f.is_dir
		else {"path": f.path, "type": "file", "size": f.size}
		for f in found
	]

	output = json.dumps(entries)
	if truncated:
		output += f"\n[stopped after {max_entries} entries; raise max_entries or narrow path]"
	return ToolResult(ok=True, output=output, metadata={"path": st.path, "count": len(entries), "truncated": truncated})


def fs_read(runtime: DockerRuntime, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
	st = runtime.stat(path)
	if st is None or st.is_dir:
		raise FileNotFoundError(f"{path!r} is not a file")

	# One read, capped: +1 byte tells us whether the file continued past the cap.
	data = runtime.read_bytes(path, max_bytes=MAX_READ_BYTES + 1)
	if _looks_binary(data):
		raise ValueError(f"{path!r} looks binary; refusing to read")
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
			"path": st.path,
			"total_lines": len(lines),
			"start_line": start,
			"end_line": end,
			"truncated": file_truncated,
		},
	)


def fs_search(
	runtime: DockerRuntime,
	pattern: str,
	path: str = ".",
	glob: str | None = None,
	regex: bool = True,
	case_sensitive: bool = False,
	max_results: int = 100,
) -> ToolResult:
	st = runtime.stat(path)
	if st is None:
		raise FileNotFoundError(f"{path!r} does not exist")

	flags = 0 if case_sensitive else re.IGNORECASE
	rx = re.compile(pattern if regex else re.escape(pattern), flags)

	if st.is_dir:
		found = runtime.list_dir(path, recursive=True, skip=SKIP_DIRS, max_entries=MAX_SEARCH_FILES)
		candidates = [f.path for f in found if not glob or fnmatch.fnmatch(posixpath.basename(f.path), glob)]
	else:
		candidates = [st.path] if not glob or fnmatch.fnmatch(posixpath.basename(st.path), glob) else []

	# One batched fetch rather than a read per file: on a remote runtime the
	# difference is one round trip versus len(candidates) of them.
	contents = runtime.get(candidates)

	matches: list[str] = []
	truncated = False
	files_scanned = 0
	for rel in candidates:
		if truncated:
			break
		data = contents.get(rel, b"")
		if _looks_binary(data):
			continue
		files_scanned += 1
		for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
			if rx.search(line):
				matches.append(f"{rel}:{lineno}: {line.rstrip(chr(10))[:300]}")
				if len(matches) >= max_results:
					truncated = True
					break

	output = "\n".join(matches) if matches else "no matches"
	if truncated:
		output += f"\n[stopped after {max_results} matches; raise max_results or narrow the search]"
	return ToolResult(
		ok=True,
		output=output,
		metadata={"matches": len(matches), "files_scanned": files_scanned, "truncated": truncated},
	)


def fs_write(runtime: DockerRuntime, path: str, content: str) -> ToolResult:
	"""Create `path`, or replace it wholesale. Parent directories are created.

	The counterpart to fs_patch, which can only edit text that is already there. Without
	this, writing a new file means smuggling it through `shell_run` in a heredoc, and the
	quoting is a coin flip for a small model - the failure then looks like a reasoning
	error when it was really an escaping one.
	"""
	st = runtime.stat(path)
	if st is not None and st.is_dir:
		raise IsADirectoryError(f"{path!r} is a directory")

	runtime.write(path, content)
	rel = runtime.relpath(path)
	lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
	verb = "overwrote" if st is not None else "wrote"
	return ToolResult(
		ok=True,
		output=f"{verb} {rel} ({len(content.encode())} bytes, {lines} lines)",
		metadata={"path": rel, "created": st is None, "bytes": len(content.encode()), "lines": lines},
	)


def fs_patch(runtime: DockerRuntime, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
	st = runtime.stat(path)

	if st is None:
		if old_string != "":
			raise FileNotFoundError(f"{path!r} does not exist; pass old_string='' to create it")
		runtime.write(path, new_string)
		rel = runtime.relpath(path)
		return ToolResult(
			ok=True,
			output=f"created {rel}",
			metadata={"path": rel, "created": True, "bytes": len(new_string.encode())},
		)

	if st.is_dir:
		raise IsADirectoryError(f"{path!r} is not a file")
	if old_string == "":
		raise ValueError("old_string is empty but the file exists; give the exact text to replace")

	text = runtime.read_text(path)
	count = text.count(old_string)
	if count == 0:
		raise ValueError(f"old_string not found in {path!r}; it must match the file exactly (including whitespace)")
	if count > 1 and not replace_all:
		raise ValueError(f"old_string matches {count} places in {path!r}; add more context or set replace_all=true")

	replacements = count if replace_all else 1
	runtime.write(path, text.replace(old_string, new_string, replacements))
	return ToolResult(
		ok=True,
		output=f"patched {st.path} ({replacements} replacement{'s' if replacements != 1 else ''})",
		metadata={"path": st.path, "replacements": replacements},
	)


def shell_run(runtime: DockerRuntime, command: str, cwd: str | None = None, timeout_s: float = 30.0) -> ToolResult:
	result = runtime.run(command, cwd=cwd, timeout_s=timeout_s)

	if result.timed_out:
		return ToolResult(
			ok=False,
			output=f"command timed out after {timeout_s}s\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
			metadata={"exit_code": None, "timed_out": True, "timeout_s": timeout_s},
		)

	parts = [f"exit_code: {result.exit_code}"]
	if result.stdout:
		parts.append(f"stdout:\n{result.stdout.rstrip()}")
	if result.stderr:
		parts.append(f"stderr:\n{result.stderr.rstrip()}")
	return ToolResult(
		ok=result.ok,
		output="\n".join(parts),
		metadata={"exit_code": result.exit_code, "timed_out": False},
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

def build_registry(runtime: DockerRuntime, allow: Iterable[str] | None = None) -> ToolRegistry:
	"""Every tool that touches the world is bound to `runtime` here.

	This is the enforcement point. There is no module-level registry, so no caller can
	obtain a filesystem tool without first deciding which runtime it acts on.

	`allow` narrows the set: with it, the model is handed only those tools plus `done`,
	which is never removable because it is the loop's only exit. A caller that wants the
	agent to have no shell says so here, and the tool is then absent from the schema the
	model ever sees - not refused at call time, which would still leave it discoverable.
	"""
	bind = lambda fn: partial(fn, runtime)  # noqa: E731

	tools = [
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
			execute=bind(fs_list),
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
			execute=bind(fs_read),
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
			execute=bind(fs_search),
		),
		Tool(
			name="fs_write",
			description=(
				"Create a file, or replace an existing one entirely, with `content`. Parent directories "
				"are created. This is the tool for writing a new file - do not build one with shell "
				"redirection or a heredoc. To change part of a file that already exists, use fs_patch."
			),
			input_schema={
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "File path relative to the working directory."},
					"content": {"type": "string", "description": "The complete contents of the file."},
				},
				"required": ["path", "content"],
			},
			execute=bind(fs_write),
		),
		Tool(
			name="fs_patch",
			description=(
				"Edit an existing file by replacing an exact string. old_string must appear exactly once "
				"unless replace_all is true. Read the file first so old_string matches exactly. "
				"To create a new file or rewrite one from scratch, use fs_write instead."
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
			execute=bind(fs_patch),
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
			execute=bind(shell_run),
		),
	]

	if allow is not None:
		keep = set(allow) | {DONE_TOOL}
		unknown = keep - {t.name for t in tools}
		if unknown:
			raise ValueError(f"unknown tool(s) in allow: {sorted(unknown)}")
		tools = [t for t in tools if t.name in keep]

	return ToolRegistry(tools)
