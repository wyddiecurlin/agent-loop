# Providers — which serverless platform, and why

Six models were asked for: **DeepSeek V4 Pro (0813), GLM 5.3, GLM 5.3 Flash, Qwen 3.7
Plus, Qwen 3.8 Max, Kimi K3**. None of them run on our box — the largest is 2.4T
parameters — so they come from someone else's GPUs. Three were compared: Fireworks,
Together, Baseten.

**Fireworks is the primary, Together is its fallback, Baseten was dropped.** Fireworks
and Together serve the same six models over the same wire, which is what makes the pair
work: a request Fireworks will not take is re-sent to Together as the same weights under
a different name, per request, and the run says so afterwards.

Prices below are USD per 1M tokens, read off each platform's own pricing page on
2026-09-07. They move; `./test.sh providers --live` re-checks the *ids*, which is the
half that documentation cannot settle.

## Coverage: why Baseten went

| model | Fireworks | Together | Baseten |
|---|---|---|---|
| DeepSeek V4 Pro (0813) | ✅ | ✅ | ✅ |
| DeepSeek V4 Flash | ✅ | ✅ | ✅ |
| GLM 5.3 | ✅ | ✅ | ✅ |
| GLM 5.3 Flash | ✅ | ✅ | ✅ |
| Kimi K3 | ✅ | ✅ | ✅ |
| **Qwen 3.7 Plus** | ✅ | ✅ | ❌ |
| **Qwen 3.8 Max** | ✅ | ✅ | ❌ |

Baseten's Model APIs catalog is ~16 models and carries neither Qwen tier (its
"Qwen3.8-27B" is a different model, not a smaller serving of the same one). You can
deploy any weights on a dedicated Baseten GPU, but that bills by the minute whether or
not a request arrives — the opposite of what an eval that runs for ten minutes a day
wants. It was also, on price, only ever ahead on GLM 5.3, and a third platform that
covers five of six models is a routing table, not a fallback.

Qwen 3.7 Plus is worth a note: it has **no open weights**, it is Alibaba's API-only
tier. That it appears on Fireworks and Together at all is a reselling arrangement, not
the usual open-weights story, and it is the one model here that could disappear from a
third-party platform without a version bump. Qwen 3.8 Max did ship open weights
(Qwen3.8-2.4T-A95B, August 2026), which is why Together lists it under that name.

## Price: on the workload this repo actually runs

Not list price — the bill for **one real HumanEval run**, using the token counts from
`evals/results/humaneval.json`: 1,906,965 input tokens of which **1,646,304 (86.3%) were
cache hits**, and 116,848 output tokens.

| model | Fireworks | Together |
|---|---|---|
| glm-5.3-flash | $0.147 | $0.147 |
| deepseek-v4-flash | $0.146 | **$0.119** |
| qwen-3.7-plus | **$0.423** | $0.760 |
| deepseek-v4-pro | **$0.879** | $1.021 |
| glm-5.3 | $1.307 | $1.307 |
| qwen-3.8-max | $1.634 | $1.634 |
| kimi-k3 | $3.029 | $3.029 |

That 86.3% is the number the platform choice turns on, and it is not an accident: an
agent loop resends the entire transcript every step, so by step 20 almost every input
token is one the server has already seen. **Headline input price is nearly irrelevant
here; the cached-input rate is the bill.** Fireworks prices DeepSeek V4 Pro's cache at
$0.044 against Together's $0.13 — 3x — and that alone is the 14% gap in the table.

It also sets the price of the fallback: a degraded run costs up to ~20% more, never a
different order of magnitude. That is cheap enough that failing over eagerly is the
right default, and expensive enough that it has to be visible. Both are true below.

## Everything else

| | Fireworks | Together |
|---|---|---|
| wire | OpenAI Chat Completions + Anthropic Messages | OpenAI Chat Completions |
| tool calling | yes, all six | yes |
| `seed` | documented | accepted |
| `top_k`, `temperature`, `top_p` | documented, top-level | yes |
| reasoning control | `reasoning_effort` (none/low/medium/high/xhigh/max/adaptive, bool, or an int budget) + `thinking` object + `reasoning_history` | `reasoning_effort` |
| cached input | 3–10% of input on the flagships | ~10–20% |
| batch | 50% off both directions | yes |
| rate limits | **published ceilings**: 64.8M total-prompt TPM / 16.2M uncached / 648k generated for <400B models | dynamic, unpublished, ramps with steady traffic |
| overload | 503 Service Overloaded even inside your limit; Priority tier (~1.25–1.5x) reduces it | 429 with `x-ratelimit-reset` |
| speed (DeepSeek V4 Pro) | 183 t/s | **208 t/s**, lowest TTFT |
| SLA | 99.9% uptime | — |

Two of those rows decided the ordering as much as price:

**Published rate limits.** `./evals.sh --shards 24` goes from zero to 24 concurrent
requests in about 600ms and stays there for ten minutes. Together's limits are dynamic
and explicitly grow with "steady, successful traffic" — the one thing a burst eval never
provides. Fireworks publishes fixed ceilings you can check a run against before spending
the wall clock. This is also why Together is the *second* platform and not the first: a
cold fallback absorbing an overflow is exactly the traffic its ramp handles worst, which
is an argument for keeping the fallback rare, not for dropping it.

**Reasoning control is per-model, and getting it wrong is a 400.** GLM 5.3 and Kimi K3
reason unconditionally: `reasoning_effort="none"` and `thinking:{"type":"disabled"}` are
both rejected, and their default is `max`. DeepSeek and the Qwen tiers can be told to
stop. This is why the catalog carries a `reasoning` tuple per model rather than one
global default — the loop sends each model the cheapest effort it will actually accept,
and `./test.sh providers` asserts that "none" never reaches the two that reject it.

## The fallback

```
PROVIDER=fireworks              -> FallbackProvider(fireworks, together)   [default]
PROVIDER=fireworks FALLBACK=none -> fireworks alone
PROVIDER=together                -> together alone
```

**Only the alias travels.** `accounts/fireworks/models/glm-5p3` means nothing to
Together, so the id goes back through the catalog to `glm-5.3` and forward again to
`zai-org/GLM-5.3`. A fallback that re-sent the id would 404 on both platforms and look
like an outage. If the catalog cannot map the id — an unlisted model set by hand — the
original error is raised rather than a quietly different model being run.

**What it falls over on** is "the platform will not serve this", not "this request is
wrong": 5xx, timeouts, transport errors, 401/403 (dead key), 404 (model absent),
408/429. A **400 propagates untouched** — it will be just as malformed on the second
platform, so falling over pays twice for the same rejection and buries the bug.

**It is eager.** The first 503 goes to Together rather than sleeping through a backoff;
at 24 shards Fireworks load-sheds with 503 *inside* your rate limit, and that is the
case this exists for. `with_retries` still wraps the pair from the loop, so both
platforms failing is what gets retried — up to four attempts at the pair.

**It is never silent.** Every fall-over prints to stderr and increments
`Usage.fallback_calls`, which rides the same accumulator as cost — so it sums through
`CostTracker`, through the per-task deltas in `evals/harness.py`, through the merge
across 24 shards, and lands in the run's JSON:

```json
"usage": { "input_tokens": 1906965, "cost_usd": 0.879, "fallback_calls": 3 }
```

A mistyped `FIREWORKS_API_KEY` sends an entire eval to the more expensive platform and
succeeds while doing it. Without that counter the only evidence would be the invoice.

## What this cost in code

One class, not three. `ChatProvider` is the old `QwenProvider` generalised: after the
base URL and the key, a hosted platform differs from our vLLM in exactly two ways, and
`Backend` is those two fields.

```
reasoning_style   "effort"        -> reasoning_effort=<value>              hosted
                  "chat_template" -> chat_template_kwargs.enable_thinking  vLLM
top_k             top-level                                                hosted
                  extra_body (not an OpenAI parameter)                     vLLM
```

`CATALOG[provider][alias] -> Model` holds the id, the three prices, the reasoning ladder,
the context window and the output cap. `PRICING` is derived from it, keyed by
**`(provider, id)`** — not by id. Fireworks and Together happen not to collide today, but
Baseten served GLM 5.3 under the identical string `zai-org/GLM-5.3` at a different cached
rate, and the id-keyed table written first billed it at Together's — a number that looks
entirely plausible. `./test.sh providers` caught that; the pair key stays so the next
platform cannot bring it back.

`MODEL` takes a portable alias, which is also what the fallback resolves through:

```
PROVIDER=fireworks MODEL=kimi-k3         ./run.sh "..."
PROVIDER=together  MODEL=glm-5.3         ./evals.sh --dataset humaneval --shards 24
PROVIDER=fireworks MODEL=deepseek-v4-pro REASONING_EFFORT=high ./evals.sh --dataset mbpp
```

An alias a platform does not serve raises, rather than falling back. An eval that thinks
it measured Qwen and measured GLM is worse than one that crashed.

## Known traps

- **The catalog's prices are transcribed, not fetched.** Fireworks' own Qwen 3.7 Plus
  launch post says $0.50/$3.00 while its pricing page says $0.40/$1.60; the page is
  taken as canonical here. Every cost number this repo prints inherits that uncertainty,
  and no test can catch it — only re-reading the page can.
- **The model ids are the weakest link.** Fireworks spells decimals with `p`
  (`glm-5p3`), and whether DeepSeek carries the `-0813` suffix in the id differs by
  platform. `./test.sh providers --live` asks each platform's `/v1/models` whether every
  id we would send exists; run it once after adding the keys, before trusting a run.
- **A fallback changes the experiment.** Two platforms serving "the same" open weights
  are not bitwise identical — different kernels, different batching, possibly different
  quantisation. A run with `fallback_calls > 0` is a mixed sample, and for a benchmark
  number that matters, `FALLBACK=none` and a rerun beat a footnote.
- **The seed buys less here than on our own box.** Continuous batching already made runs
  only near-reproducible; on a shared serverless backend the batch contains other
  tenants' traffic. Still worth sending, still not a promise. Compare runs paired.
- **8192 output tokens truncates a model that cannot stop reasoning.** GLM 5.3 and Kimi
  K3 spend the budget thinking before they write a tool call, so the catalog gives them
  32,768 and `MAX_OUTPUT_TOKENS` overrides it. A truncated turn reads from the loop as a
  model that cannot follow instructions, which is the most expensive kind of wrong.

Sources: [Fireworks serverless pricing](https://docs.fireworks.ai/serverless/pricing),
[Fireworks rate limits](https://docs.fireworks.ai/serverless/rate-limits),
[Fireworks chat completions API](https://docs.fireworks.ai/api-reference/post-chatcompletions),
[Fireworks reasoning guide](https://docs.fireworks.ai/guides/reasoning),
[Qwen 3.7 Plus on Fireworks](https://fireworks.ai/blog/qwen-3p7-plus),
[Together pricing](https://www.together.ai/pricing),
[Together serverless models](https://docs.together.ai/docs/serverless-models),
[Together rate limits](https://docs.together.ai/docs/rate-limits),
[Baseten pricing](https://www.baseten.co/pricing/) and
[Model APIs](https://docs.baseten.co/development/model-apis/overview) (evaluated, dropped),
[Artificial Analysis: DeepSeek V4 Pro providers](https://artificialanalysis.ai/models/deepseek-v4-pro/providers).
