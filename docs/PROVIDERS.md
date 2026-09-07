# Providers — which serverless platform, and why

Six models were asked for: **DeepSeek V4 Pro (0813), GLM 5.3, GLM 5.3 Flash, Qwen 3.7
Plus, Qwen 3.8 Max, Kimi K3**. None of them run on our box — the largest is 2.4T
parameters — so they come from someone else's GPUs. Three candidates: Fireworks,
Together, Baseten.

**The pick is Fireworks**, and all three are wired up anyway, because they speak the same
wire and the difference between them is one env var. `PROVIDER=baseten ./evals.sh ...`
is the whole cost of keeping the comparison honest.

Prices below are USD per 1M tokens, read off each platform's own pricing page on
2026-09-07. They move; `./test.sh providers --live` re-checks the *ids*, which is the
half that documentation cannot settle.

## Coverage: only one platform serves all six

| model | Fireworks | Together | Baseten |
|---|---|---|---|
| DeepSeek V4 Pro (0813) | ✅ | ✅ | ✅ |
| DeepSeek V4 Flash | ✅ | ✅ | ✅ |
| GLM 5.3 | ✅ | ✅ | ✅ |
| GLM 5.3 Flash | ✅ | ✅ | ✅ |
| Kimi K3 | ✅ | ✅ | ✅ |
| **Qwen 3.7 Plus** | ✅ | ✅ | ❌ |
| **Qwen 3.8 Max** | ✅ | ✅ | ❌ |

Baseten's Model APIs catalog is ~16 models and carries neither Qwen tier (it has a
Qwen3.8-27B, which is a different model, not a smaller serving of the same one). That
is not fatal — you can deploy any weights on a dedicated Baseten GPU — but a dedicated
deployment is billed by the minute whether or not a request arrives, which is the
opposite of what an eval that runs for ten minutes a day wants.

Qwen 3.7 Plus is worth a note: it has **no open weights**, it is Alibaba's API-only tier.
That it appears on Fireworks and Together at all is a reselling arrangement, not the
usual open-weights story, and it is the one model here that could disappear from a
third-party platform without a version bump. Qwen 3.8 Max did ship open weights
(Qwen3.8-2.4T-A95B, August 2026), which is why Together lists it under that name.

## Price: on the workload this repo actually runs

Not list price — the bill for **one real HumanEval run**, using the token counts from
`evals/results/humaneval.json`: 1,906,965 input tokens of which **1,646,304 (86.3%) were
cache hits**, and 116,848 output tokens.

| model | Fireworks | Together | Baseten |
|---|---|---|---|
| glm-5.3-flash | $0.147 | $0.147 | $0.147 |
| deepseek-v4-flash | $0.146 | $0.119 | **$0.114** |
| qwen-3.7-plus | **$0.423** | $0.760 | — |
| deepseek-v4-pro | **$0.879** | $1.021 | $1.021 |
| glm-5.3 | $1.307 | $1.307 | **$1.110** |
| qwen-3.8-max | **$1.634** | $1.634 | — |
| kimi-k3 | $3.029 | $3.029 | $3.029 |

That 86.3% is the number the platform choice turns on, and it is not an accident: an
agent loop resends the entire transcript every step, so by step 20 almost every input
token is one the server has already seen. **Headline input price is nearly irrelevant
here; the cached-input rate is the bill.** Fireworks prices DeepSeek V4 Pro's cache at
$0.044 against Together's $0.13 — 3x — and that alone is the 14% gap in the table.

Nobody wins everything. Baseten is genuinely cheaper on GLM 5.3 ($0.14 cache vs $0.26)
and on DeepSeek V4 Flash. Together is not uniquely cheapest on anything. Fireworks wins
the two flagship agent models and is the only platform that can run all six, which is
what makes it the default rather than a per-model routing table nobody will maintain.

## Everything else

| | Fireworks | Together | Baseten |
|---|---|---|---|
| wire | OpenAI Chat Completions + Anthropic Messages | OpenAI Chat Completions | OpenAI Chat Completions + Anthropic Messages |
| tool calling | yes, all six | yes | yes, all models |
| `seed` | documented | accepted | accepted |
| `top_k`, `temperature`, `top_p` | documented, top-level | yes | yes |
| reasoning control | `reasoning_effort` (none/low/medium/high/xhigh/max/adaptive, bool, or an int budget) + `thinking` object + `reasoning_history` | `reasoning_effort` | `reasoning_effort` |
| cached input | 3–10% of input on the flagships | ~10–20% | ~10% |
| batch | 50% off both directions | yes | — |
| rate limits | **published ceilings**: 64.8M total-prompt TPM / 16.2M uncached / 648k generated for <400B models | dynamic, unpublished, ramps with steady traffic | per-account |
| overload | 503 Service Overloaded even inside your limit; Priority tier (~1.25–1.5x) reduces it | 429 with `x-ratelimit-reset` | — |
| speed (DeepSeek V4 Pro) | 183 t/s | **208 t/s**, lowest TTFT | fastest on the *max*-effort variant (164 t/s vs 83) |
| SLA | 99.9% uptime | — | Enterprise tier |

Two of those rows decided it as much as price:

**Published rate limits.** `./evals.sh --shards 24` goes from zero to 24 concurrent
requests in about 600ms and stays there for ten minutes. Together's limits are dynamic
and explicitly grow with "steady, successful traffic" — which is the one thing a burst
eval never provides. Fireworks publishes fixed ceilings you can check a run against
before spending the wall clock.

**Reasoning control is per-model, and getting it wrong is a 400.** GLM 5.3 and Kimi K3
reason unconditionally: `reasoning_effort="none"` and `thinking:{"type":"disabled"}` are
both rejected, and their default is `max`. DeepSeek and the Qwen tiers can be told to
stop. This is why the catalog in `providers.py` carries a `reasoning` tuple per model
rather than a single global default — the loop sends each model the cheapest effort it
will actually accept, and `./test.sh providers` asserts that "none" never reaches the two
that reject it.

## What this cost in code

One class, not three. `ChatProvider` is the old `QwenProvider` generalised: after the
base URL and the key, a hosted platform differs from our vLLM in exactly two ways, and
`Backend` is those two fields.

```
reasoning_style   "effort"        -> reasoning_effort=<value>            hosted
                  "chat_template" -> chat_template_kwargs.enable_thinking  vLLM
top_k             top-level                                              hosted
                  extra_body (not an OpenAI parameter)                   vLLM
```

`CATALOG[provider][alias] -> Model` holds the id, the three prices, the reasoning ladder,
the context window and the output cap. `PRICING` is derived from it, keyed by
**`(provider, id)`** — not by id. That is not fastidiousness: Together and Baseten both
serve GLM 5.3 under the identical string `zai-org/GLM-5.3` at different cached rates, so
an id-keyed price table bills one of them at the other's rate and reports a number that
looks entirely plausible. The flat map was written first and `./test.sh providers` caught
it.

`MODEL` takes a portable alias, so moving a whole eval between platforms is a prefix:

```
PROVIDER=fireworks MODEL=kimi-k3         ./run.sh "..."
PROVIDER=baseten   MODEL=glm-5.3         ./evals.sh --dataset humaneval --shards 24
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
- **The seed buys less here than on our own box.** Continuous batching already made
  runs only near-reproducible; on a shared serverless backend the batch contains other
  tenants' traffic. Still worth sending, still not a promise. Compare runs paired.
- **8192 output tokens truncates a model that cannot stop reasoning.** GLM 5.3 and Kimi
  K3 spend the budget thinking before they write a tool call, so the catalog gives them
  32,768 and `MAX_OUTPUT_TOKENS` overrides it. A truncated turn reads from the loop as a
  model that cannot follow instructions, which is the most expensive kind of wrong.
- **503 is not an error to surface.** Fireworks load-sheds with 503 *inside* your rate
  limit, so `with_retries` treats every 5xx as transient. At 24 shards this will happen.

Sources: [Fireworks serverless pricing](https://docs.fireworks.ai/serverless/pricing),
[Fireworks rate limits](https://docs.fireworks.ai/serverless/rate-limits),
[Fireworks chat completions API](https://docs.fireworks.ai/api-reference/post-chatcompletions),
[Fireworks reasoning guide](https://docs.fireworks.ai/guides/reasoning),
[Qwen 3.7 Plus on Fireworks](https://fireworks.ai/blog/qwen-3p7-plus),
[Together pricing](https://www.together.ai/pricing),
[Together serverless models](https://docs.together.ai/docs/serverless-models),
[Together rate limits](https://docs.together.ai/docs/rate-limits),
[Baseten pricing](https://www.baseten.co/pricing/),
[Baseten Model APIs](https://docs.baseten.co/development/model-apis/overview),
[Artificial Analysis: DeepSeek V4 Pro providers](https://artificialanalysis.ai/models/deepseek-v4-pro/providers),
[Artificial Analysis: Baseten](https://artificialanalysis.ai/providers/baseten).
