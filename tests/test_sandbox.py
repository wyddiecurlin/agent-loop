# AI_OWNED
"""Runtime conformance + escape. Runs INSIDE the container, like everything else.

	./test.sh sandbox

The conformance cases pin the runtime's contract. The escape cases each assert a
denial, and only a real boundary passes them - which is the point of there being
exactly one topology: these now run in the same place the agent does.
"""

import os
import time

from agent_loop.runtime import DockerRuntime
from agent_loop.tools import SKIP_DIRS

FIXTURE = {
	"t/notes.txt": "alpha\nbeta\nbanana\n",
	"t/app.py": "print('hi')\n",
	"t/data/config.json": '{"v": 1}\n',
	"t/data/deep/x.txt": "x\n",
	"t/__pycache__/junk.pyc": b"\x00\x01binary",
}
CASES = []


def case(fn):
	CASES.append((fn.__name__[2:].replace("_", " "), fn))
	return fn


def raises(exc, fn, *a, **kw) -> bool:
	try:
		fn(*a, **kw)
	except exc:
		return True
	return False


# -- conformance ------------------------------------------------------------

@case
def t_write_and_read_text_and_bytes(rt):
	rt.write("t/a/b/deep.txt", "hello\n")
	assert rt.read_text("t/a/b/deep.txt") == "hello\n"       # parents created
	rt.write("t/blob.bin", bytes(range(256)))
	assert rt.read_bytes("t/blob.bin") == bytes(range(256))
	assert rt.read_bytes("t/notes.txt", max_bytes=5) == b"alpha"


@case
def t_read_missing_or_a_directory_raises(rt):
	assert raises(FileNotFoundError, rt.read_bytes, "t/nope.txt")
	assert raises(FileNotFoundError, rt.read_bytes, "t/data")


@case
def t_stat_reports_file_dir_and_missing(rt):
	f, d = rt.stat("t/notes.txt"), rt.stat("t/data")
	assert (f.path, f.is_dir, f.size) == ("t/notes.txt", False, 18), f
	assert (d.path, d.is_dir, d.size) == ("t/data", True, 0), d
	assert rt.stat("t/nope") is None


@case
def t_list_dir_orders_dirs_first_then_files(rt):
	assert [f.path for f in rt.list_dir("t")] == [
		"t/__pycache__", "t/data", "t/app.py", "t/notes.txt"]


@case
def t_list_dir_recursive_is_files_depth_first(rt):
	assert [f.path for f in rt.list_dir("t", recursive=True)] == [
		"t/app.py", "t/notes.txt", "t/__pycache__/junk.pyc",
		"t/data/config.json", "t/data/deep/x.txt"]


@case
def t_list_dir_prunes_skip_and_caps_entries(rt):
	assert [f.path for f in rt.list_dir("t", recursive=True, skip=SKIP_DIRS)] == [
		"t/app.py", "t/notes.txt", "t/data/config.json", "t/data/deep/x.txt"]
	assert len(rt.list_dir("t", recursive=True, max_entries=2)) == 2
	assert raises(NotADirectoryError, rt.list_dir, "t/notes.txt")


@case
def t_relpath_normalizes_and_refuses_to_escape(rt):
	assert rt.relpath("t/data/../notes.txt") == "t/notes.txt"
	for outside in ("../secrets", "t/../../etc/passwd", "/etc/passwd"):
		assert raises(PermissionError, rt.relpath, outside), outside


@case
def t_remove_handles_file_tree_missing_and_root(rt):
	rt.remove("t/notes.txt")
	rt.remove("t/data")
	rt.remove("t/never/existed")                              # not an error
	assert rt.stat("t/notes.txt") is None and rt.stat("t/data") is None
	assert raises(PermissionError, rt.remove, ".")


@case
def t_put_and_get_round_trip_bytes(rt):
	rt.put({"t/x/one.txt": "1", "t/x/y/two.bin": b"\x00\x02"})
	assert rt.get(["t/x/one.txt", "t/x/y/two.bin"]) == {
		"t/x/one.txt": b"1", "t/x/y/two.bin": b"\x00\x02"}
	assert rt.get([]) == {}


@case
def t_run_separates_streams_cwd_and_env(rt):
	r = rt.run("echo out; echo err >&2; exit 3")
	assert (r.exit_code, r.ok, r.timed_out) == (3, False, False), r
	assert r.stdout.strip() == "out" and r.stderr.strip() == "err", r
	assert rt.run("cat config.json", cwd="t/data").stdout.strip() == '{"v": 1}'
	assert rt.run("echo $GREETING", env={"GREETING": "hi"}).stdout.strip() == "hi"
	# A different user, but the same workspace: it can edit what the agent wrote.
	assert rt.run("echo more >> t/notes.txt && mkdir -p t/new && echo x > t/new/f").ok
	assert rt.read_text("t/notes.txt").endswith("more\n") and rt.read_text("t/new/f") == "x\n"


@case
def t_run_times_out_and_kills_grandchildren(rt):
	r = rt.run("sleep 30", timeout_s=1)
	assert r.timed_out and r.exit_code is None and r.duration_s < 15, r
	rt.run("(sleep 2; echo leaked > t/leak.txt) & sleep 30", timeout_s=1)
	time.sleep(4)
	assert rt.stat("t/leak.txt") is None, "a grandchild outlived the timeout"


# -- escape: each case asserts a denial --------------------------------------

@case
def t_escape_model_commands_cannot_reach_the_agents_keys(rt):
	"""Our tool code holds the keys; the model's commands must not. Two layers: the
	env allowlist keeps them out of the command's environment, and running as a
	different user keeps them out of /proc, which the allowlist alone cannot do."""
	assert os.getenv("QWEN_API_KEY"), "the agent should hold the token"
	env = rt.run("env").stdout
	assert "QWEN_API_KEY" not in env and os.environ["QWEN_API_KEY"] not in env
	assert rt.run("id -un").stdout.strip() == "sandbox"
	assert not rt.run("cat /proc/1/environ").ok, "/proc must be closed, not just env"
	assert not rt.run("su -c id root").ok, "and there must be no way back up"


@case
def t_escape_the_host_filesystem_is_not_there(rt):
	assert not rt.run("ls /Users").ok
	assert not rt.run("cat /work/../.env").ok
	assert "Darwin" not in rt.run("uname -a").stdout


@case
def t_the_network_is_open(rt):
	"""Egress is unrestricted by decision: the agent may call any server. The gateway
	is the one host it *needs*, so that is the one asserted."""
	import urllib.request
	urllib.request.urlopen(os.environ["QWEN_BASE_URL"] + "/models", timeout=10)
	assert rt.run("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 5)\"").ok


@case
def t_escape_the_memory_limit_holds(rt):
	assert not rt.run("python3 -c 'b = bytearray(3 * 10**9)'", timeout_s=60).ok


@case
def t_escape_snapshot_and_reset_roll_the_workspace_back(rt):
	sha = rt.snapshot()
	rt.write("t/notes.txt", "clobbered")
	rt.write("t/extra.txt", "untracked")
	rt.reset(sha)
	assert rt.read_text("t/notes.txt") == FIXTURE["t/notes.txt"]
	assert rt.stat("t/extra.txt") is None


def main() -> int:
	rt = DockerRuntime()
	rt.setup()
	failed = 0
	for label, fn in CASES:
		try:
			rt.remove("t")
			rt.put(FIXTURE)
			fn(rt)
			print(f"[PASS] {label}")
		except Exception as exc:  # noqa: BLE001 - a failed assertion is a failed case
			failed += 1
			print(f"[FAIL] {label} -- {type(exc).__name__}: {exc}")
	print(f"\n[RESULT] {len(CASES) - failed}/{len(CASES)} sandbox cases passed")
	return 1 if failed else 0


if __name__ == "__main__":
	raise SystemExit(main())
