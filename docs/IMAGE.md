# Image input — image_view

One `image_view(source)` tool for image understanding, with a shared converter and
small changes to the existing tool registry, loop, and provider adapters.
Face recognition and client integration (including Mimo) are out of scope.

**Inputs.** Accept a URL, a local path, or bytes. The Python entry point accepts `bytes`;
the JSON tool accepts a base64 data URL (`data:image/jpeg;base64,...`). Fetch URLs before conversion so remote
images obey the same compression policy as files. Runtime owns local reads; reuse the
web client's URL validation, redirect checks, timeouts, and bounded downloads. Detect
formats from decoded content rather than trusting filenames or HTTP content types.

**Conversion.** Convert common photo formats: JPEG, PNG, WebP, GIF (first frame), BMP,
TIFF, HEIC/HEIF, AVIF, and camera RAW (including DNG, CR2/CR3, NEF, ARW, and RAF where
the decoder supports them). Use Pillow with HEIF support and a LibRaw-backed decoder;
keep conversion in memory. Correct orientation, flatten transparency, remove metadata,
and emit JPEG. Reject corrupt inputs and unsupported camera variants with a useful
error; bound encoded input, decoded dimensions, and PDF rendering work.

Optimize for recognizing image content rather than inspecting individual pixels.
Always aggressively compress inputs over **500 KB (500,000 bytes)**: start at a
512-pixel longest edge and JPEG quality 40, then lower quality and dimensions until
output is at most **50 KB**. Apply the dimension cap to smaller inputs too, since a
small encoded file can still have large dimensions. Never enlarge a small image.
Compression alone does not determine vision
token cost: resizing and the model's supported low-detail mode also matter. Do not
silently retry at full resolution.

**PDFs.** Automatically split a PDF into ordered page images and apply the same
converter. Batch pages by the active model/provider's remaining image, payload, and
context budgets, accounting for images already in history and the configured fallback.
Process every batch automatically, keeping page-numbered text observations before
releasing its image payload and continuing. Do not merely split pages into messages
within one oversized request, silently truncate a document, or require manual page
ranges. Explicit page selection may narrow the work. Report failed pages and reject
encrypted or malformed documents clearly.

**Loop and wire format.** Keep tool output text small; carry image attachments separately
in `ToolResult.metadata`. Append one image-bearing user message after **all** pending
tool results, preserving call/result pairing. Keep plain string messages compatible;
serialize text/image parts for OpenAI Responses and Chat Completions. Estimate image
cost as vision tokens rather than base64 text. See
[COMPACTION.md](COMPACTION.md): after compaction, history retains text observations and
references, never image bytes or automatically restored image parts.

**Model specifications.** Extend `Model` in [providers.py](../agent_loop/providers.py)
with explicit vision support, maximum images per request, and applicable image/payload
limits, keyed by provider and exact model ID. Record the source and verification date.
Use `0` for verified lack of image support and `None` for an unknown maximum; unknown
must never mean unlimited. A configured conservative operating cap is distinct from a
verified provider maximum. Validate the entire outgoing request before every send,
including retries and fallback; apply the stricter applicable limits. Unsupported or
unverified endpoints must return an actionable tool error, not discard images or
silently substitute a different model.

`image_formats` lists native accepted encodings on each exact provider/model spec;
`None` means the endpoint is unverified and `()` means no accepted image formats.
`rejected_image_formats` lists confirmed rejections, with `"*"` meaning all formats.
Formats absent from both lists remain unverified. Names are lowercase: `jpeg` includes
JPG, `tiff` includes TIF, and `gif` means a single-frame image. These fields describe
the image payload channel, not separate provider file APIs. `image_view` still accepts
the source formats above and converts them to JPEG before sending; PDF pages are
rendered, and RAW support comes from the converter.

[Native format results](image-format-validation.jsonl), checked 2026-09-10, record
original encodings sent directly to every reachable vision endpoint. Accepted probes
must identify the blue rectangle; decoder/MIME rejections are recorded separately
from endpoint failures. HEIF/HEIC probes use HEVC; other codecs and animated images
remain unverified. The verified native capabilities are:

| Provider / models | Accepted | Rejected |
|---|---|---|
| Fireworks: GLM 5.3 Flash, Kimi K3, Qwen 3.8 Max | JPEG, PNG, WebP, GIF, BMP, TIFF, PPM, HEIF, HEIC, AVIF | PDF |
| Together: GLM 5.3 Flash, Kimi K3 | JPEG, PNG, WebP, GIF, BMP, TIFF, PPM | HEIF, HEIC, AVIF, PDF |
| Together: Qwen 3.7 Plus | JPEG, PNG, WebP, GIF, BMP, TIFF, PPM, HEIF, HEIC, AVIF | PDF |
| Local: Qwen 3.5 9B | JPEG, PNG, WebP, GIF, BMP, TIFF, PPM, AVIF | HEIF, HEIC, PDF |
| OpenAI: GPT 5.4 Nano, GPT 5 Nano, GPT 5 Mini, GPT 5 | JPEG, PNG, WebP, GIF | BMP, TIFF, PPM, HEIF, HEIC, AVIF, PDF; RAW excluded by documented supported types |
| Fireworks and Together: DeepSeek V4 Pro, DeepSeek V4 Flash, GLM 5.3; Together: Qwen 3.8 Max | None | All image formats (text-only endpoints) |
| Fireworks: Qwen 3.7 Plus | Unverified (404) | Unverified |

Native RAW encodings remain unverified outside OpenAI's documented exclusion and
text-only endpoints. Fireworks accepts more encodings in these live probes than its
guide lists; the spec records those observed capabilities. Together's capabilities
vary by exact endpoint, so they must not be inferred from another model on the platform.

**Provider research (2026-09-10).** These are serving constraints, not proof that every
catalog entry supports vision. Finish exact-model verification before enabling it.

| Provider | Published image-count constraint | Other constraints / verification |
|---|---|---|
| Fireworks | 30 per request | Less than 10 MB of base64 image data; direct URL images must be under 5 MB and download within 1.5 seconds. [Vision guide](https://docs.fireworks.ai/guides/querying-vision-language-models). |
| Together | No numeric maximum found in the reviewed guide | Verify each exact endpoint and record a conservative tested operating cap until a maximum is established. [Vision guide](https://docs.together.ai/docs/inference/vision/overview), [model catalog](https://docs.together.ai/docs/serverless/models). |
| OpenAI Responses | 1,500 per request | Up to 512 MB total payload; model-specific image processing and context limits still apply. [Image input requirements](https://developers.openai.com/api/docs/guides/images-vision#image-input-requirements). |
| Local Qwen / vLLM | Deployment-specific `--limit-mm-per-prompt` | Current vLLM documents a default of 999 per modality; read the actual deployment configuration rather than assuming that default. [Engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/#multimodalconfig). |

The [Fireworks specification for DeepSeek V4 Pro 0813](https://fireworks.ai/models/deepseek-ai/deepseek-v4-pro-0813)
explicitly says image input is unsupported. Test the exact pinned endpoints; a newer
vision variant is a different model and cannot establish support for these IDs.

**Acceptance.** Keep implementation clear, consistent, concise, and limited to the
shared image path. Cover URL/path/bytes equivalence, malformed input, common formats
and actual RAW fixtures, orientation, compression above 500 KB, and automatic PDF
batching with page order and complete coverage. Verify both provider wire formats,
whole-history count/payload limits, fallback with a lower limit, preserved tool pairs,
and absence of image bytes after compaction/resume. Compare image answers with a blind
control so a successful API response alone does not count as vision support.

Run live tool tests for **every catalogued provider/model pair**, including supported
streaming and non-streaming paths, and publish per-pair results with exact IDs, date,
image count, and pass/fail/unsupported/unverified status. Exercise the configured limit
and local rejection above it. Missing credentials, an unreachable endpoint, or a
text-only model must remain visible in the results; none counts as a passing vision
test. The current catalog matrix is:

| Model alias | Providers to test |
|---|---|
| `deepseek-v4-pro` | Fireworks, Together |
| `deepseek-v4-flash` | Fireworks, Together |
| `glm-5.3` | Fireworks, Together |
| `glm-5.3-flash` | Fireworks, Together |
| `kimi-k3` | Fireworks, Together |
| `qwen-3.7-plus` | Fireworks, Together |
| `qwen-3.8-max` | Fireworks, Together |
| `qwen3.5-9b` | Local Qwen / vLLM |
| `gpt-5.4-nano` | OpenAI |
| `gpt-5-nano` | OpenAI |
| `gpt-5-mini` | OpenAI |
| `gpt-5` | OpenAI |

**Validation recorded with this change.** [Per-endpoint results](image-validation.jsonl)
cover all 19 exact catalog IDs on 2026-09-10 (PDT), with fallback disabled to isolate
each provider. Eleven endpoints passed red/blue image probes and the streamed
`image_view` → `done` flow: Fireworks GLM 5.3 Flash, Kimi K3, Qwen 3.8 Max; local Qwen;
all four OpenAI models; and Together GLM 5.3 Flash, Kimi K3, Qwen 3.7 Plus.
Each also passed at its operating batch size (four
images locally, eight on the hosted endpoints). These tests do not establish the
provider's absolute maximum; those limits come from the cited documentation. Blind prompts sometimes guessed a color, so changing the
actual image from red to blue was also required to verify visual input affected the
answer. Fireworks DeepSeek Pro/Flash and GLM 5.3 rejected images; Fireworks Qwen 3.7
Plus returned 404. With credentials configured, Together's two DeepSeek endpoints,
GLM 5.3, and Qwen 3.8 Max also explicitly rejected images. All seven Together pairs
were verified; its three vision endpoints use a tested eight-image operating cap,
while their absolute maximum remains unverified in the reviewed documentation. Unsupported endpoints have a
zero-image limit. A configured fallback that cannot accept images also prevents image
requests; use `FALLBACK=none` to select a vision-capable primary alone.

Together Qwen 3.7 Plus requires streaming: the adapter collects its stream into the
same `ModelTurn` when callers request a non-streamed result, including PDF summaries.
Together Qwen 3.8 Max accepts only `low`, `medium`, and `xhigh` reasoning effort;
the catalog now records that ladder independently of Fireworks.

The local launch script was checked: it sets four images per prompt, reflected in the
local model spec. Keep that spec in sync when changing the deployment. Offline tests
cover conversion, URL bounds, both wire formats, complete-history limits, the stricter
fallback limit, tool-call pairing, and PDF batches. Actual Canon CR2 and Nikon NEF
samples from [rawpy's test collection](https://github.com/letmaik/rawpy/tree/main/test)
converted to 21,285 and 6,967 bytes respectively, with a 512-pixel longest edge. A
live nine-page PDF test on local Qwen correctly identified pages 2, 5, and 9 across
automatic batches.
A nine-page PDF also passed on Qwen 3.7 Plus through the actual Fireworks 404 fallback:
Together served four calls, including both page-summary batches, and returned `2, 5, 9`.
Compaction itself remains planned; this change documents its image-removal contract.

Run the checks inside the container:

```sh
AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m unittest tests.test_images tests.test_context
AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_images --live
AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_images --live-tool
AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_images --live-formats
```
