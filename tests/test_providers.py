"""Proof that the catalog and the request it builds are right, without a network.

	./test.sh providers          the offline cases below
	./test.sh providers --live   also ask each platform whether the ids exist

Six models, three platforms, and the ways they disagree are not decorative: GLM 5.3 and
Kimi K3 reject a request that asks them not to reason, Baseten does not serve either Qwen
tier, and the same model costs three different prices depending on who is serving it. Each
of those is one assertion here, because each of them is a 400, a wrong model, or a wrong
invoice if the catalog drifts.

The one thing this file cannot prove offline is that a model id still exists. `--live`
does that, and needs the keys.
"""

import os
import sys

from agent_loop.providers import (
	BACKENDS,
	CATALOG,
	PRICING,
	ChatProvider,
	compute_cost,
	default_model,
	make_provider,
	resolve_model,
)

# The six the platforms were chosen for, plus the two Flash tiers we get for free.
REQUESTED = ("deepseek-v4-pro", "glm-5.3", "glm-5.3-flash", "kimi-k3", "qwen-3.7-plus", "qwen-3.8-max")
ALWAYS_REASONS = ("glm-5.3", "glm-5.3-flash", "kimi-k3")
HOSTED = ("fireworks", "together", "baseten")


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


class FakeClient:
	"""Records the kwargs generate() built, so the request shape is testable."""

	def __init__(self):
		self.seen: dict = {}
		outer = self

		class _Completions:
			def create(self, **kw):
				outer.seen = kw
				return _Resp()

		self.chat = type("c", (), {"completions": _Completions()})()


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

	# 2. Baseten's Model APIs carry no Qwen tier. Asking for one must raise, not fall back:
	#    an eval that thinks it measured Qwen and measured GLM is worse than a crash.
	for alias in ("qwen-3.7-plus", "qwen-3.8-max"):
		try:
			resolve_model(alias, "baseten")
			results.append(check(f"baseten refuses {alias}", False, "resolved instead of raising"))
		except ValueError as exc:
			results.append(check(f"baseten refuses {alias}", "does not serve" in str(exc)))

	# 3. An id the catalog has never seen passes through untouched, so a model released
	#    after this file was written is a MODEL= away and not a code change.
	unknown = "accounts/fireworks/models/something-new"
	results.append(check("an unlisted id passes through", resolve_model(unknown, "fireworks") == unknown))

	# 4. Two platforms serve GLM 5.3 under the identical id `zai-org/GLM-5.3` at different
	#    cached rates, so a price keyed by id alone silently bills one of them at the
	#    other's rate. This asserts the pair key that makes that impossible, and it is
	#    here because the flat map was the first thing written and it was wrong.
	rows = [((name, m.id), (m.input, m.cached_input, m.output))
	        for name, models in CATALOG.items() for m in models.values()]
	shared = {i for (_, i), _ in rows if len({n for (n, j), _ in rows if j == i}) > 1}
	results.append(check("an id served by two platforms is priced twice", bool(shared),
	                     f"shared ids: {', '.join(sorted(shared))}"))
	results.append(check("PRICING covers every (platform, id) pair", all(k in PRICING for k, _ in rows)))
	results.append(check("...and prices them apart",
	                     PRICING[("together", "zai-org/GLM-5.3")]["cached_input"]
	                     != PRICING[("baseten", "zai-org/GLM-5.3")]["cached_input"]))

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
	results.append(check("top_k is top-level on hosted, extra_body on vLLM",
	                     fw.get("top_k") == 20 and qw.get("extra_body", {}).get("top_k") == 20 and "top_k" not in qw))

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
	                     compute_cost("Qwen/Qwen3.7-Plus", 10, 0, 10, "baseten") == 0.0))

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
