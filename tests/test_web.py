# AI_OWNED
"""The web tools, with the network replaced by fixtures. Runs inside the container.

	AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_web [--live]

Every case drives the real WebClient over an httpx MockTransport, so what is tested is
the pipeline - URL policy, redirects, size and type caps, HTML to markdown, the
extraction prompt, the tool formatting - and not a copy of it. `--live` adds one Brave
search, one real fetch, and one extraction through the self-hosted model.
"""

import json
import os
import sys

from agent_loop import web
from agent_loop.providers import QWEN_MODEL, ModelTurn
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import ToolCall, build_registry
from agent_loop.web import Fetched, Redirected, SearchHit, WebClient, to_markdown, validate

httpx = web.httpx

CASES = []


def case(fn):
	CASES.append((fn.__name__[2:].replace("_", " "), fn))
	return fn


def raises(exc, fn, *a, **kw) -> str:
	"""The message of the exception `fn` raised, or '' if it did not."""
	try:
		fn(*a, **kw)
	except exc as e:
		return str(e)
	return ""


# -- fixtures ----------------------------------------------------------------

ADDRESSES = {
	"example.com": ["93.184.216.34"],
	"www.example.com": ["93.184.216.34"],
	"other.org": ["203.0.113.5"],
	"internal.corp": ["10.0.0.7"],
	"metadata.google.internal": ["169.254.169.254"],
	"169.254.169.254": ["169.254.169.254"],
	"loop.example.com": ["127.0.0.1"],
	"api.search.brave.com": ["1.1.1.1"],
}

PAGE = """<!doctype html><html><head><title>Graph Breaks &amp; You</title>
<style>body{color:red}</style><script>alert(1)</script></head>
<body><nav><a href="/">Home</a><a href="/docs">Docs</a></nav>
<h1>Graph breaks</h1>
<p>torch.compile   splits the   graph when it meets <code>print()</code>.
See <a href="/tutorials/breaks.html">the tutorial</a>.</p>
<ul><li>first</li><li>second &lt;item&gt;</li></ul>
<pre><code>def f(x):
    return x + 1
</code></pre>
<table><tr><th>flag</th><th>effect</th></tr><tr><td>fullgraph</td><td>error on break</td></tr></table>
<a href="/img"><img src="x.png"></a>
<footer>Copyright</footer></body></html>"""

BRAVE = {
	"web": {
		"results": [
			{"title": "TorchDynamo &amp; graph breaks", "url": "https://example.com/docs",
			 "description": "How <strong>graph breaks</strong> happen", "age": "March 3, 2026"},
			{"title": "Second", "url": "https://other.org/x", "description": ""},
		]
	}
}


def handler(request: httpx.Request) -> httpx.Response:
	url, path = str(request.url), request.url.path
	if request.url.host == "api.search.brave.com":
		handler.last_search = request
		return httpx.Response(200, json=BRAVE)
	if request.url.host != "example.com" and request.url.host != "www.example.com":
		return httpx.Response(404)
	routes = {
		"/page": httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=PAGE.encode()),
		"/plain": httpx.Response(200, headers={"content-type": "text/plain"}, content=b"  just text  "),
		"/moved": httpx.Response(301, headers={"location": "https://www.example.com/page"}),
		"/away": httpx.Response(302, headers={"location": "https://other.org/x"}),
		"/inward": httpx.Response(302, headers={"location": "https://internal.corp/admin"}),
		"/forever": httpx.Response(302, headers={"location": "/forever"}),
		"/stub": httpx.Response(200, headers={"content-type": "text/html"}, content=(
			b'<html><head><meta http-equiv="refresh" content="0; url=page"></head></html><script>x()</script>')),
		"/stub-away": httpx.Response(200, headers={"content-type": "text/html"}, content=(
			b'<meta HTTP-EQUIV="Refresh" content="5; URL=\'https://other.org/x\'">')),
		"/png": httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"),
		"/big": httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 2_000),
		"/missing": httpx.Response(404, content=b"nope"),
	}
	handler.last_fetch = url
	return routes.get(path, httpx.Response(404))


class FakeExtractor:
	"""Stands in for the qwen provider: records the request, answers with a constant."""

	def __init__(self):
		self.messages = None

	def generate(self, messages, model, tools, **kw):
		self.messages, self.model, self.tools = messages, model, tools
		return ModelTurn(text="  the answer  ", tool_calls=None, usage=None, stop_reason="completed")


def client(**kw) -> WebClient:
	return WebClient(api_key=kw.pop("api_key", "k"),
	                 http=httpx.Client(transport=httpx.MockTransport(handler)),
	                 extractor=kw.pop("extractor", FakeExtractor()), **kw)


# -- URL policy ---------------------------------------------------------------

@case
def t_validate_upgrades_http_and_strips_userinfo():
	assert validate("http://user:pw@Example.com/a?b=1#frag") == "https://example.com/a?b=1"
	assert validate("https://example.com") == "https://example.com/"
	assert validate("https://example.com:8443/x") == "https://example.com:8443/x"


@case
def t_validate_refuses_other_schemes_and_bare_hosts():
	assert "only http(s)" in raises(ValueError, validate, "ftp://example.com/f")
	assert "only http(s)" in raises(ValueError, validate, "example.com/no-scheme")
	assert "not a public hostname" in raises(ValueError, validate, "http://localhost:9000/v1/models")
	assert "not a public hostname" in raises(ValueError, validate, "http://qwen/v1")


@case
def t_validate_refuses_private_loopback_and_link_local():
	assert "10.0.0.7" in raises(ValueError, validate, "https://internal.corp/")
	assert "127.0.0.1" in raises(ValueError, validate, "https://loop.example.com/")
	assert "169.254.169.254" in raises(ValueError, validate, "http://169.254.169.254/latest/meta-data/")
	assert "169.254.169.254" in raises(ValueError, validate, "http://metadata.google.internal/")
	assert "does not resolve" in raises(ValueError, validate, "https://no.such.host.invalid/")


# -- fetch --------------------------------------------------------------------

@case
def t_fetch_html_becomes_markdown():
	page = client().fetch("http://example.com/page")
	assert isinstance(page, Fetched) and page.url == "https://example.com/page"
	md = page.markdown
	assert md.startswith("# Graph Breaks & You"), md[:60]
	assert "# Graph breaks\n" in md
	assert "torch.compile splits the graph when it meets `print()`" in md  # whitespace collapsed
	assert "[the tutorial](https://example.com/tutorials/breaks.html)" in md  # links absolute
	assert "- first\n- second <item>" in md
	assert "```\ndef f(x):\n    return x + 1\n```" in md  # code kept verbatim
	assert "\n| flag | effect |\n| fullgraph | error on break |" in md
	for gone in ("alert(1)", "color:red", "Home", "Copyright", "[](", "Docs"):
		assert gone not in md, gone


@case
def t_fetch_plain_text_is_returned_as_is():
	page = client().fetch("https://example.com/plain")
	assert page.markdown == "just text" and page.content_type == "text/plain"


@case
def t_fetch_follows_same_host_redirects_only():
	page = client().fetch("https://example.com/moved")
	assert page.url == "https://www.example.com/page" and page.markdown.startswith("# Graph")
	r = client().fetch("https://example.com/away")
	assert isinstance(r, Redirected) and (r.url, r.location) == ("https://example.com/away", "https://other.org/x")


@case
def t_fetch_rechecks_policy_on_every_hop():
	# A cross-host hop goes back to the model, and fetching it is refused like any URL.
	r = client().fetch("https://example.com/inward")
	assert isinstance(r, Redirected) and r.location == "https://internal.corp/admin"
	assert "10.0.0.7" in raises(ValueError, client().fetch, r.location)
	assert "redirected more than" in raises(ValueError, client().fetch, "https://example.com/forever")


@case
def t_fetch_treats_meta_refresh_as_a_redirect():
	page = client().fetch("https://example.com/stub")
	assert page.url == "https://example.com/page" and page.markdown.startswith("# Graph Breaks"), page
	r = client().fetch("https://example.com/stub-away")
	assert isinstance(r, Redirected) and r.location == "https://other.org/x", r


@case
def t_fetch_refuses_non_text_oversize_and_http_errors():
	assert "not text" in raises(ValueError, client().fetch, "https://example.com/png")
	before = web.MAX_BODY_BYTES
	web.MAX_BODY_BYTES = 1_000
	try:
		assert "larger than" in raises(ValueError, client().fetch, "https://example.com/big")
	finally:
		web.MAX_BODY_BYTES = before
	assert "404" in raises(httpx.HTTPStatusError, client().fetch, "https://example.com/missing")


@case
def t_markdown_survives_broken_html():
	assert to_markdown("<p>unclosed <b>bold <a href='x'>link") == "unclosed bold [link](x)"
	assert to_markdown("<title> T </title><h1>T</h1><p>body") == "# T\n\nbody"  # no duplicate title
	assert to_markdown("") == ""
	assert to_markdown("<script>only</script>") == ""


# -- search -------------------------------------------------------------------

@case
def t_search_sends_the_key_and_parses_results():
	hits = client(api_key="secret").search("graph breaks", limit=2, allowed_domains=["example.com", "other.org"])
	req = handler.last_search
	assert req.headers["x-subscription-token"] == "secret"
	assert req.url.params["q"] == "graph breaks site:example.com OR site:other.org"
	assert req.url.params["count"] == "2" and req.url.params["text_decorations"] == "false"
	assert hits == [
		SearchHit("TorchDynamo & graph breaks", "https://example.com/docs", "How <strong>graph breaks</strong> happen", "March 3, 2026"),
		SearchHit("Second", "https://other.org/x", "", ""),
	]


@case
def t_search_without_a_key_names_the_variable():
	key = os.environ.pop(web.BRAVE_KEY_ENV, None)
	try:
		assert web.BRAVE_KEY_ENV in raises(RuntimeError, client(api_key=None).search, "x")
	finally:
		if key is not None:
			os.environ[web.BRAVE_KEY_ENV] = key


# -- extraction ---------------------------------------------------------------

@case
def t_extract_sends_page_and_question_with_thinking_off():
	fake = FakeExtractor()
	answer = client(extractor=fake).extract("PAGE " * 20_000, "what is it?")
	assert answer == "the answer"
	assert fake.tools is None
	system, user = fake.messages
	assert system["role"] == "system" and "untrusted" in system["content"]
	assert user["content"].startswith("Question: what is it?\n\nPage:\nPAGE")
	assert len(user["content"]) <= web.MAX_PAGE_CHARS + 40  # capped, not the whole page


@case
def t_extractor_defaults_to_qwen_with_thinking_off():
	c = WebClient(http=httpx.Client(transport=httpx.MockTransport(handler)))
	assert c._extractor is None  # built on first use, so a registry needs no qwen env
	fake = FakeExtractor()
	c._extractor = fake
	c.extract("p", "q")
	assert fake.model == os.getenv("QWEN_MODEL", QWEN_MODEL)


# -- the tools ----------------------------------------------------------------

def call(reg, name, **args):
	return reg.execute(ToolCall(id="1", name=name, arguments=json.dumps(args)))


@case
def t_registry_binds_both_tools_and_formats_search():
	reg = build_registry(DockerRuntime(), web=client())
	assert "web_search" in reg and "web_fetch" in reg
	names = {t["name"] for t in reg.schema()}
	assert {"web_search", "web_fetch", "done"} <= names
	r = call(reg, "web_search", query="graph breaks")
	assert r.ok and r.output == (
		"1. TorchDynamo & graph breaks\n   https://example.com/docs  (March 3, 2026)\n   How <strong>graph breaks</strong> happen\n"
		"2. Second\n   https://other.org/x"
	), r.output
	assert r.metadata["hits"] == 2


@case
def t_fetch_tool_returns_page_answer_or_redirect():
	reg = build_registry(DockerRuntime(), web=client())
	r = call(reg, "web_fetch", url="http://example.com/page")
	assert r.ok and r.output.startswith("# Graph Breaks & You") and r.metadata["chars"] == len(r.output)
	r = call(reg, "web_fetch", url="http://example.com/page", prompt="what splits?")
	assert r.ok and r.output.startswith("the answer\n\nsource: https://example.com/page (") and r.metadata["extracted"]
	r = call(reg, "web_fetch", url="https://example.com/away")
	assert r.ok and r.output.startswith("redirect: https://other.org/x") and r.metadata["redirect"] == "https://other.org/x"


@case
def t_fetch_tool_errors_are_readable():
	reg = build_registry(DockerRuntime(), web=client())
	r = call(reg, "web_fetch", url="http://localhost:9000/v1/models")
	assert not r.ok and "not a public hostname" in r.output
	r = call(reg, "web_fetch", url="https://example.com/missing")
	assert not r.ok and "404" in r.output
	r = call(reg, "web_search", query="x", allowed_domains="one.org")  # a bare string still filters
	assert r.ok and handler.last_search.url.params["q"] == "x site:one.org"


@case
def t_allow_can_leave_the_web_out():
	reg = build_registry(DockerRuntime(), allow=["fs_read"], web=client())
	assert "web_search" not in reg and "web_fetch" not in reg


# -- live ---------------------------------------------------------------------

def live() -> None:
	"""One real search, fetch, and extraction. Needs the key and the qwen box."""
	c = WebClient()
	hits = c.search("python 3.12 what's new", limit=3, allowed_domains=["docs.python.org"])
	assert hits and all("python.org" in h.url for h in hits), hits
	print(f"  search: {len(hits)} hits, first {hits[0].url}")
	page = c.fetch(hits[0].url)
	assert isinstance(page, Fetched) and len(page.markdown) > 1_000, page
	print(f"  fetch: {len(page.markdown)} chars of markdown")
	answer = c.extract(page.markdown, "Which PEP introduced the type parameter syntax? One sentence.")
	print(f"  extract: {answer[:200]!r}")
	assert "695" in answer, answer


def fake_addresses(host: str) -> list[str]:
	if host not in ADDRESSES:
		raise web.socket.gaierror(f"{host}: fixture has no address")
	return ADDRESSES[host]


def main(argv: list[str]) -> int:
	resolve = web._addresses
	web._addresses = fake_addresses
	failed = 0
	for label, fn in CASES:
		try:
			fn()
			print(f"[PASS] {label}")
		except Exception as exc:  # noqa: BLE001 - a failed assertion is a failed case
			failed += 1
			print(f"[FAIL] {label} -- {type(exc).__name__}: {exc}")
	if "--live" in argv:
		web._addresses = resolve
		try:
			live()
			print("[PASS] live search, fetch, extract")
		except Exception as exc:  # noqa: BLE001
			failed += 1
			print(f"[FAIL] live -- {type(exc).__name__}: {exc}")
	print(f"\n[RESULT] {len(CASES) + ('--live' in argv) - failed}/{len(CASES) + ('--live' in argv)} web cases passed")
	return 1 if failed else 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
