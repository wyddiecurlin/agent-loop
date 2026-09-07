# AI_OWNED
"""Proof that the catalog, the request it builds, and the fall-over are right, offline.

	./test.sh providers          the offline cases below
	./test.sh providers --live   also ask each platform whether the ids exist

Six models on two platforms, and the ways they disagree are not decorative: GLM 5.3 and
Kimi K3 reject a request that asks them not to reason, the same model costs different
money depending on who serves it, and the two platforms call the same weights by
different names - which is the one fact the fallback stands on. Each of those is one
assertion here, because each is a 400, a wrong invoice, or a fallback that silently runs
a different model if the catalog drifts.

The one thing this file cannot prove offline is that a model id still exists. `--live`
does that, and needs the keys.
"""

import inspect
import os
import sys

from openai import APIStatusError, APITimeoutError
from openai.resources.chat.completions import Completions

from agent_loop.providers import (
	BACKENDS,
	CATALOG,
	PRICING,
	ChatProvider,
	CostTracker,
	FallbackProvider,
	Usage,
	alias_for,
	can_fail_over,
	compute_cost,
	default_model,
	make_provider,
	resolve_model,
)

try:  # matches providers.py: openai>=3 vendors its transport as httpx2
	import httpx2 as httpx
except ModuleNotFoundError:
	import httpx

# The six the platforms were chosen for, plus the two Flash tiers we get for free.
REQUESTED = ("deepseek-v4-pro", "glm-5.3", "glm-5.3-flash", "kimi-k3", "qwen-3.7-plus", "qwen-3.8-max")
ALWAYS_REASONS = ("glm-5.3", "glm-5.3-flash", "kimi-k3")
HOSTED = ("fireworks", "together")


def check(label: str, passed: bool | None, detail: str = "") -> bool | None:
	"""None means "not run". A skip that prints PASS is a forged score, which is the one
	thing every test file in this repo exists to make impossible."""
	print(f"[{'SKIP' if passed is None else 'PASS' if passed else 'FAIL'}] {label}"
	      + (f" -- {detail}" if detail else ""))
	return passed


# -- a client that answers without a socket -----------------------------------

class _Fn:
	def __init__(self, name, args):
		self.name, self.arguments = name, args


class _TC:
	def __init__(self):
		self.id, self.type, self.function = "call_1", "function", _Fn("done", '{"answer":"x"}')


class _Msg:
	content = None
	tool_calls = [_TC()]


class _Choice:
	message, finish_reason = _Msg(), "tool_calls"


class _Usage:
	prompt_tokens, completion_tokens = 1000, 100
	prompt_tokens_details = type("d", (), {"cached_tokens": 900})()
	completion_tokens_details = type("d", (), {"reasoning_tokens": 40})()


class _Resp:
	choices, usage = [_Choice()], _Usage()


# The SDK validates keyword names against its own signature and raises before a request
# is sent, so a fake that accepts anything is a fake that cannot catch the most likely
# mistake: putting a non-OpenAI parameter (top_k) at the top level. It cost a real
# TypeError on the first live Fireworks call, so the fake now rejects what the SDK does.
SDK_PARAMS = set(inspect.signature(Completions.create).parameters) - {"self"}


class _Raw:
	"""What with_raw_response.create() hands back: headers, and the body behind .parse()."""

	def __init__(self, headers: dict):
		self.headers, self._body = headers, _Resp()

	def parse(self):
		return self._body


class FakeClient:
	"""Records the kwargs generate() built, so the request shape is testable.

	Shaped like the real client down to `with_raw_response`, because that is the only path
	to the response headers and the headers are where Fireworks reports cache hits.
	"""

	def __init__(self, headers: dict | None = None):
		self.seen: dict = {}
		self.headers = {} if headers is None else headers
		outer = self

		class _Raw_:
			def create(self, **kw):
				if bad := sorted(set(kw) - SDK_PARAMS):
					raise TypeError(f"Completions.create() got unexpected keyword arguments {bad}")
				outer.seen = kw
				return _Raw(outer.headers)

		class _Completions:
			with_raw_response = _Raw_()

		self.chat = type("c", (), {"completions": _Completions()})()


def status_error(code: int) -> APIStatusError:
	req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
	return APIStatusError("boom", response=httpx.Response(code, request=req), body=None)


class FailingClient:
	"""A client that raises instead of answering, so the fall-over path is reachable."""

	def __init__(self, exc: Exception):
		outer = self

		class _Raw_:
			def create(self, **kw):
				raise outer.exc

		class _Completions:
			with_raw_response = _Raw_()

		self.exc = exc
		self.chat = type("c", (), {"completions": _Completions()})()


def pair(exc: Exception) -> tuple[FallbackProvider, FakeClient]:
	"""Fireworks that always raises `exc`, paired with a Together that always answers."""
	second = FakeClient()
	return FallbackProvider(
		ChatProvider(backend="fireworks", client=FailingClient(exc)),
		ChatProvider(backend="together", client=second),
	), second


def request_for(backend: str, alias: str, **kw) -> dict:
	"""The kwargs ChatProvider would put on the wire for one model on one platform."""
	client = FakeClient()
	p = ChatProvider(backend=backend, client=client, **kw)
	p.generate([{"role": "user", "content": "hi"}], resolve_model(alias, backend),
	           [{"name": "done", "description": "", "parameters": {}}], tool_choice="required")
	return client.seen


# -- live ---------------------------------------------------------------------

def live_ids(backend_name: str) -> tuple[bool | None, str]:
	"""Ask the platform for its model list and check every id we would send is in it.

	This is the half of the catalog that documentation cannot settle: a price can be read
	off a page, but an id is only true if the server says so.
	"""
	from openai import OpenAI  # local: this path is the only one that needs a socket

	b = BACKENDS[backend_name]
	key = os.getenv(b.api_key_env)
	if not key:
		return None, f"no {b.api_key_env}, ids unverified"
	client = OpenAI(base_url=os.getenv(b.base_url_env, b.base_url), api_key=key, timeout=60, max_retries=0)
	served = {m.id for m in client.models.list().data}
	missing = sorted(m.id for m in CATALOG[backend_name].values() if m.id not in served)
	return not missing, ("all ids served" if not missing else f"not served: {', '.join(missing)}")


def main(argv: list[str]) -> int:
	results = []

	# 1. Every model the platform choice was made for is actually in the catalog, on the
	#    platform we chose. A silent gap here is a run that quietly used something else.
	for alias in REQUESTED:
		results.append(check(f"fireworks serves {alias}", alias in CATALOG["fireworks"]))

	# 2. Both platforms serve all six, which is what makes the pair a fallback rather than
	#    a routing table. If one ever drops a model this fails here, not at 3am in a shard.
	results.append(check("both platforms serve every requested model",
	                     all(a in CATALOG["fireworks"] and a in CATALOG["together"] for a in REQUESTED)))
	#    An alias a provider does not serve must raise, never substitute: an eval that
	#    thinks it measured Kimi and measured GLM is worse than one that crashed.
	try:
		resolve_model("qwen-3.7-plus", "qwen")
		results.append(check("a provider refuses a model it lacks", False, "resolved instead of raising"))
	except ValueError as exc:
		results.append(check("a provider refuses a model it lacks", "does not serve" in str(exc)))

	# 3. An id the catalog has never seen passes through untouched, so a model released
	#    after this file was written is a MODEL= away and not a code change.
	unknown = "accounts/fireworks/models/something-new"
	results.append(check("an unlisted id passes through", resolve_model(unknown, "fireworks") == unknown))

	# 4. Prices are keyed by (platform, id), not id. Fireworks and Together happen not to
	#    collide today; Baseten, evaluated and dropped, served GLM 5.3 under the identical
	#    string `zai-org/GLM-5.3` at a different cached rate, and the id-keyed map written
	#    first billed it at Together's. The key stays a pair so the next platform cannot
	#    bring the bug back, and this asserts the property that made it a bug.
	rows = [((name, m.id), (m.input, m.cached_input, m.output))
	        for name, models in CATALOG.items() for m in models.values()]
	results.append(check("PRICING covers every (platform, id) pair", all(k in PRICING for k, _ in rows)))

	# 5. The models that cannot stop reasoning are never asked to. This is a 400 from the
	#    platform, not a soft degradation, so it is worth one assertion per platform.
	for backend in HOSTED:
		for alias in ALWAYS_REASONS:
			if alias not in CATALOG[backend]:
				continue
			spec = CATALOG[backend][alias]
			floor_ok = "none" not in spec.reasoning
			sent = request_for(backend, alias).get("reasoning_effort")
			results.append(check(f"{backend}/{alias} is never sent effort=none",
			                     floor_ok and sent != "none", f"sent {sent!r}"))

	# 6. ...and the ones that can are, by default, because reasoning tokens are output
	#    tokens and this loop pays for them on every one of up to 60 steps.
	sent = request_for("fireworks", "deepseek-v4-pro").get("reasoning_effort")
	results.append(check("a toggleable model defaults to effort=none", sent == "none", f"sent {sent!r}"))
	sent = request_for("fireworks", "deepseek-v4-pro", thinking=True).get("reasoning_effort")
	results.append(check("thinking=1 asks for the top of the ladder", sent == "max", f"sent {sent!r}"))
	sent = request_for("fireworks", "kimi-k3", reasoning_effort="high").get("reasoning_effort")
	results.append(check("REASONING_EFFORT overrides the floor", sent == "high", f"sent {sent!r}"))

	# 7. The two dialects stay on their own side of the seam. vLLM's chat-template flag is
	#    not a parameter Fireworks knows, and reasoning_effort is not one vLLM reads.
	fw = request_for("fireworks", "glm-5.3")
	qw = request_for("qwen", "qwen3.5-9b")
	results.append(check("hosted gets reasoning_effort, not chat_template_kwargs",
	                     "reasoning_effort" in fw and "chat_template_kwargs" not in fw.get("extra_body", {})))
	results.append(check("vLLM gets chat_template_kwargs, not reasoning_effort",
	                     "reasoning_effort" not in qw and "enable_thinking" in qw["extra_body"]["chat_template_kwargs"]))
	#    top_k is not an OpenAI parameter on any platform, whatever the platform's own
	#    docs say: the SDK raises on the name before the server is reached.
	results.append(check("top_k rides in extra_body everywhere, never top-level",
	                     all(r.get("extra_body", {}).get("top_k") == 20 and "top_k" not in r
	                         for r in (fw, qw))))
	results.append(check("no request carries a keyword the SDK would reject",
	                     all(not (set(r) - SDK_PARAMS) for r in (fw, qw))))

	# 8. An always-reasoning model needs room to think before it writes a tool call. At
	#    8192 the reasoning eats the budget and the turn returns truncated, which reads
	#    from the loop as a model that cannot follow instructions.
	results.append(check("always-reasoning models get a bigger output cap",
	                     all(CATALOG[b][a].max_output >= 32_768
	                         for b in HOSTED for a in ALWAYS_REASONS if a in CATALOG[b])))
	results.append(check("max_tokens on the wire follows the catalog",
	                     request_for("fireworks", "glm-5.3")["max_tokens"] == 32_768))

	# 9. Seed and sampling reach every Chat Completions backend, hosted or not. Whatever
	#    a shared serverless batch does to determinism, not sending the seed guarantees
	#    the worse answer.
	results.append(check("seed and sampling are sent to hosted backends too",
	                     fw.get("seed") == 0 and fw.get("temperature") == 0.6 and fw.get("top_p") == 0.95))

	# 10. Cost is charged at the serving platform's rate, not the model's "usual" one.
	#     Same model, same tokens, two invoices - which is the entire reason to compare
	#     platforms rather than assume open weights cost the same everywhere.
	toks = (1_000_000, 900_000, 100_000)
	fw_cost = compute_cost(CATALOG["fireworks"]["deepseek-v4-pro"].id, *toks, provider="fireworks")
	tg_cost = compute_cost(CATALOG["together"]["deepseek-v4-pro"].id, *toks, provider="together")
	results.append(check("the same model costs less on fireworks (deeper cache discount)",
	                     fw_cost < tg_cost, f"${fw_cost:.4f} vs ${tg_cost:.4f}"))
	results.append(check("an unpriced model is free, not a crash",
	                     compute_cost("nope", 10, 0, 10, "fireworks") == 0.0))
	# The same id on the wrong platform must not borrow that platform's price.
	results.append(check("a price is never borrowed across platforms",
	                     compute_cost("Qwen/Qwen3.7-Plus", 10, 0, 10, "fireworks") == 0.0))

	# 11. Usage is parsed off the chat response, including the cached and reasoning
	#     subsets - the two numbers the platform comparison actually turns on.
	client = FakeClient()
	turn = ChatProvider(backend="fireworks", client=client).generate(
		"hi", CATALOG["fireworks"]["glm-5.3-flash"].id, None)
	u = turn.usage
	results.append(check("usage carries cached and reasoning subsets",
	                     u.input_tokens == 1000 and u.cached_input_tokens == 900 and u.reasoning_tokens == 40
	                     and u.cost_usd > 0, str(u)))
	results.append(check("tool calls survive the translation",
	                     turn.tool_calls and turn.tool_calls[0].name == "done"))

	#     ...and on Fireworks the body's cached_tokens is a lie. It is present and always
	#     0 while the real count rides in a header, so believing the body bills a whole
	#     agent run at the uncached rate. _Usage above says 900 cached; the header says 950
	#     and must win, and the cost must fall accordingly.
	priced = ChatProvider(backend="fireworks", client=FakeClient()).generate(
		"hi", CATALOG["fireworks"]["deepseek-v4-pro"].id, None)
	hdr = ChatProvider(backend="fireworks", client=FakeClient({"fireworks-cached-prompt-tokens": "950"})
	                   ).generate("hi", CATALOG["fireworks"]["deepseek-v4-pro"].id, None)
	results.append(check("the cache header beats the body's cached_tokens",
	                     hdr.usage.cached_input_tokens == 950 and priced.usage.cached_input_tokens == 900))
	results.append(check("...and a cache hit actually lowers the bill",
	                     hdr.usage.cost_usd < priced.usage.cost_usd,
	                     f"${hdr.usage.cost_usd:.8f} vs ${priced.usage.cost_usd:.8f}"))
	junk = ChatProvider(backend="fireworks", client=FakeClient({"fireworks-cached-prompt-tokens": "?"})
	                    ).generate("hi", CATALOG["fireworks"]["deepseek-v4-pro"].id, None)
	results.append(check("an unparseable header falls back to the body",
	                     junk.usage.cached_input_tokens == 900))

	# 12. A hosted platform without its key fails at construction. The alternative is a
	#     401 on request 200 of a shard, after the run has already cost wall-clock.
	for name in HOSTED:
		saved = os.environ.pop(BACKENDS[name].api_key_env, None)
		try:
			make_provider(name)
			results.append(check(f"{name} without a key refuses to start", False, "constructed anyway"))
		except RuntimeError as exc:
			results.append(check(f"{name} without a key refuses to start", "not set" in str(exc)))
		finally:
			if saved is not None:
				os.environ[BACKENDS[name].api_key_env] = saved

	# 13. MODEL= takes a portable alias, so one env var moves a whole eval between
	#     platforms without touching the dataset or the harness.
	saved_p, saved_m = os.environ.get("PROVIDER"), os.environ.get("MODEL")
	try:
		os.environ["PROVIDER"], os.environ["MODEL"] = "together", "kimi-k3"
		results.append(check("MODEL= resolves an alias per provider",
		                     default_model() == "moonshotai/Kimi-K3", default_model()))
		os.environ["PROVIDER"] = "fireworks"
		results.append(check("...and the same alias on another provider",
		                     default_model() == "accounts/fireworks/models/kimi-k3", default_model()))
	finally:
		for k, v in (("PROVIDER", saved_p), ("MODEL", saved_m)):
			os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

	# 14. The fall-over. Only the alias travels: Together has never heard of
	#     `accounts/fireworks/models/glm-5p3`, so a fallback that resent the id would 404
	#     on both platforms and look like an outage.
	fb, second = pair(status_error(503))
	turn = fb.generate("hi", CATALOG["fireworks"]["glm-5.3"].id, None)
	results.append(check("a 503 on fireworks is served by together",
	                     second.seen.get("model") == "zai-org/GLM-5.3", second.seen.get("model")))
	results.append(check("the fall-over is counted, not silent", turn.usage.fallback_calls == 1))

	# 15. ...and billed at the platform that actually served it. Together's cached rate for
	#     DeepSeek V4 Pro is 3x Fireworks': a fallback charged at the primary's price is a
	#     cost report that quietly under-states every degraded run.
	fb, second = pair(status_error(503))
	turn = fb.generate("hi", CATALOG["fireworks"]["deepseek-v4-pro"].id, None)
	expect = compute_cost(CATALOG["together"]["deepseek-v4-pro"].id, 1000, 900, 100, "together")
	results.append(check("a fallback call is billed at the platform that served it",
	                     abs(turn.usage.cost_usd - expect) < 1e-12,
	                     f"{turn.usage.cost_usd} vs {expect}"))

	# 16. What is worth failing over on. A 400 is a malformed request and will be just as
	#     malformed on the second platform, so falling over on it pays twice for the same
	#     rejection and buries the bug.
	results.append(check("503, 429, timeouts and dead keys fail over",
	                     all(can_fail_over(e) for e in (status_error(503), status_error(429),
	                                                    status_error(401), status_error(404),
	                                                    APITimeoutError(request=httpx.Request("POST", "https://x.invalid"))))))
	results.append(check("a malformed request does not", not can_fail_over(status_error(400))))
	fb, second = pair(status_error(400))
	try:
		fb.generate("hi", CATALOG["fireworks"]["glm-5.3"].id, None)
		results.append(check("a 400 propagates instead of falling over", False, "fell over anyway"))
	except APIStatusError as exc:
		results.append(check("a 400 propagates instead of falling over",
		                     exc.status_code == 400 and not second.seen))

	#     The dated build is what is pinned, on both platforms: `deepseek-v4-pro` also
	#     resolves on Fireworks and floats, which would change what a stored number means.
	results.append(check("the flagship is pinned to a dated build",
	                     all(CATALOG[b]["deepseek-v4-pro"].id.endswith("0813") for b in HOSTED)))

	# 17. A model the catalog cannot map raises the original error rather than guessing.
	#     Silently running whatever the second platform has under a similar name is the
	#     one failure mode worse than the outage it is covering for.
	fb, second = pair(status_error(503))
	try:
		fb.generate("hi", "accounts/fireworks/models/something-new", None)
		results.append(check("an unmappable model raises rather than guessing", False, "fell over anyway"))
	except APIStatusError:
		results.append(check("an unmappable model raises rather than guessing", not second.seen))
	results.append(check("alias_for round-trips a known id",
	                     alias_for("zai-org/GLM-5.3", "together") == "glm-5.3"
	                     and alias_for("nope", "together") is None))

	# 18. The count survives the tracker, so a run's JSON says how much of it was degraded.
	tracker = CostTracker()
	tracker.record(type("t", (), {"usage": Usage(10, 0, 5, 0, 0.1, 1)})())
	tracker.record(type("t", (), {"usage": Usage(10, 0, 5, 0, 0.1, 0)})())
	results.append(check("fallback_calls sums through the tracker",
	                     tracker.total.fallback_calls == 1 and "fallback=1" in tracker.summary(),
	                     tracker.summary()))

	# 19. Wiring. FALLBACK=none is off, an unknown name is an error rather than a silent
	#     single-platform run, and PROVIDER=together does not pair with itself.
	saved = {k: os.environ.get(k) for k in ("PROVIDER", "FALLBACK", "FIREWORKS_API_KEY", "TOGETHER_API_KEY")}
	try:
		os.environ["FIREWORKS_API_KEY"] = os.environ["TOGETHER_API_KEY"] = "test-key"
		os.environ.pop("FALLBACK", None)
		results.append(check("fireworks pairs with together by default",
		                     isinstance(make_provider("fireworks"), FallbackProvider)))
		results.append(check("together alone does not pair with itself",
		                     isinstance(make_provider("together"), ChatProvider)))
		os.environ["FALLBACK"] = "none"
		results.append(check("FALLBACK=none turns it off",
		                     isinstance(make_provider("fireworks"), ChatProvider)))
		os.environ["FALLBACK"] = "baseten"
		try:
			make_provider("fireworks")
			results.append(check("an unknown FALLBACK is an error", False, "accepted it"))
		except ValueError as exc:
			results.append(check("an unknown FALLBACK is an error", "unknown FALLBACK" in str(exc)))
		# Asked for explicitly and keyless: an error. Defaulted and keyless: a warning and
		# no cover, because that run is still the run the operator described.
		os.environ["FALLBACK"] = "together"
		del os.environ["TOGETHER_API_KEY"]
		try:
			make_provider("fireworks")
			results.append(check("an explicit keyless FALLBACK is an error", False, "accepted it"))
		except RuntimeError as exc:
			results.append(check("an explicit keyless FALLBACK is an error", "not set" in str(exc)))
		os.environ.pop("FALLBACK")
		results.append(check("a defaulted keyless fallback degrades to one platform",
		                     isinstance(make_provider("fireworks"), ChatProvider)))
	finally:
		for k, v in saved.items():
			os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

	if "--live" in argv:
		print("\n-- live: does the platform still serve these ids? --")
		for name in HOSTED:
			ok, detail = live_ids(name)
			results.append(check(f"{name} serves every catalogued id", ok, detail))

	ran = [r for r in results if r is not None]
	skipped = len(results) - len(ran)
	print(f"\n{sum(ran)}/{len(ran)} passed" + (f", {skipped} skipped" if skipped else ""))
	return 0 if all(ran) else 1


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
