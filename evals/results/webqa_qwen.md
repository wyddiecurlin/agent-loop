# Brave vs Parallel: five-question smoke benchmark

Brave and Parallel Fast tied on answer accuracy, but Brave supplied cited support more reliably (12/15 versus 10/15). Fast cost about one fifth as much in search fees and had similar overall agent latency. Brave returned searches faster and Qwen reproduced its source quotes more accurately. Advanced solved one additional trial, with more input tokens and a higher tail latency. Keep Brave as the default while the dataset is this small; Parallel Fast is a useful optional backend.

September 10, 2026; configured self-hosted `qwen3.5-9b`, thinking disabled, same model for page extraction. [Raw results and tool traces](webqa_qwen.json), [tasks](../data/webqa.jsonl), [runner](../webqa.py).

| Agent result | Brave | Parallel Fast | Parallel Advanced |
|---|---:|---:|---:|
| Trials (five questions × three repeats) | 15 | 15 | 15 |
| Correct answer | 12/15 | 12/15 | 13/15 |
| Correct answer + cited supporting result | 12/15 | 10/15 | 13/15 |
| Also supplied a verified verbatim quote | 12/15 | 7/15 | 9/15 |
| Median task time | 4.65s | 4.48s | 4.73s |
| p95 task time (nearest rank) | 7.52s | 9.88s | 10.34s |
| Search / fetch calls | 17 / 12 | 16 / 10 | 16 / 8 |
| Tool errors | 6 | 3 | 5 |
| Model input tokens, including extraction | 96,776 | 93,714 | 111,231 |
| Model output tokens, including extraction | 4,178 | 4,751 | 4,177 |
| Tool output characters | 34,019 | 90,444 | 125,199 |
| Estimated search fees for these 15 trials | $0.085 | $0.016 | $0.080 |

Fees use published list prices, counting attempted calls: [Brave $5/1K](https://brave.com/search/api/), [Parallel Fast $1/1K and Advanced $5/1K](https://parallel.ai/pricing), requesting five results. They exclude credits, self-hosted inference/hosting costs, and preliminary development checks. The recorded comparison, including fixed searches and the no-web control, took 247.37 seconds of task execution and an estimated $0.236 in search fees. This is not a total compute bill.

| Fixed-query search only (one query per question) | Brave | Parallel Fast | Parallel Advanced |
|---|---:|---:|---:|
| Supporting result among five returned hits | 5/5 | 5/5 | 5/5 |
| Median search time | 0.165s | 0.588s | 1.434s |
| Total returned tool characters | 10,130 | 33,904 | 40,966 |
| Search fees | $0.025 | $0.005 | $0.025 |

These queries were curated before the scored comparison. Pilot searches had already warmed some queries, so these are not cold-cache latency measurements. The identical queries isolate retrieval; the agent trials let the model choose searches and fetches freely within the limits.

| Task | Brave | Fast | Advanced |
|---|---:|---:|---:|
| Adapted BrowseComp 244: paper's first author | 0/3 | 0/3 | 1/3 |
| Adapted BrowseComp 266: exhibition title | 3/3 | 3/3 | 3/3 |
| Adapted BrowseComp 790: band identification | 3/3 | 3/3 | 3/3 |
| Python ZIP metadata encoding version | 3/3 | 3/3 | 3/3 |
| SQLite compound SELECT limit | 3/3 | 1/3 | 3/3 |

The borrowed/adapted slice scores 6/9, 6/9, and 7/9 respectively; the two documentation questions score 6/6, 4/6, and 6/6 respectively. These are custom adaptations, not official BrowseComp results. The paper-author task remains a stress case: Qwen frequently fetched blocked ResearchGate pages or PDFs even though search results already contained the answer. The existing fetcher rejects PDFs. Eight failed trials stopped at the search/fetch budget; two more Fast trials answered 500 correctly but cited a build-specific SQLite setting that did not establish the default. No task reached the 60-second deadline.

The no-web control answered only the Python version question correctly (1/5). It also supplied invented citations, which did not count because no supporting tool result existed. That question partly tests evidence retrieval rather than new factual knowledge for this model; the four other questions remain useful tests of information gained through tools.

The automated scorer checks exact answer aliases and a cited result from a reviewed domain containing the answer and task-specific evidence terms. All 35 passing agent answers were manually reviewed against their cited tool text. Source support is separate from quote fidelity: Parallel's exhibition snippets supported the answers, but Qwen rearranged or paraphrased passages inside its quote fields. One Advanced author answer included a valid ResearchGate citation and an incorrect `.qxd` URL; a pass means at least one supporting citation, not that every citation is valid.

During the audit, the ResearchGate author record was cross-checked against the publisher/PubMed, and the [WOMEX artist/label entry](https://www.womex.com/virtual/eka3/jadal/arabic_rocks) was independently verified. Both were added as equivalent sources. The final audit tightened SQLite evidence to require the default value rather than a build-specific setting, removing two Fast passes; a regression test covers that distinction. Every saved trace was rescored offline. The JSON retains the initial stricter summary and the audit note. No questions were removed or rewritten after seeing provider results.

Limits: three searches, two fetches, eight agent turns, 60 seconds per task; five hits per search; 2,048 output tokens per agent turn and 1,024 per extraction. No model retries or provider fallback. Provider order rotates between questions/repeats. Three repeats show stability on five questions, not statistical confidence over general web use. The larger 30-question suite remains future work.

Validation: 21 offline web contract cases; eight evaluator/grading/budget tests; 5/5 canonical grader checks; live Parallel domain-restricted search, fetch, and extraction; repository boundary lint; diff whitespace check.
