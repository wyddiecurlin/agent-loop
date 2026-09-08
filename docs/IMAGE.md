# Image input — vision in the loop

One seam, two tools, one converter. The goal is that *"what is wrong in this screenshot?"*
and *"compare these three photos"* work on a HEIC off a phone and a CR2 off a camera, on
the same six models the text loop already runs on.

```
./run.sh --image a.heic --image b.jpg "which one is sharper?"   session-level
image_view(paths, prompt=None)                                  agent-initiated
pdf_read(path, pages=None)                                      one image per page
```

No image generation, no video, no OCR tier, no face recognition.

## The one hard constraint

**A tool result cannot carry an image.** `FunctionCallOutputItem.output` is a string, and
neither the Responses API nor Chat Completions accepts image content in a `tool`-role
message. So `image_view` cannot return the picture.

The fix is an attachment seam: a `ToolResult` may set `metadata["attachments"]`, its text
output stays a string (`attached 2 images: a.heic -> 1568x1176 jpeg, b.jpg -> ...`), and
the loop appends **one extra user message** carrying the image parts directly after the
tool output. ~15 lines in `loop.py`, and the main model sees actual pixels.

Anthropic's API *does* allow image blocks inside `tool_result` — which is why Claude Code's
Read tool returns an image directly and needs no seam. The constraint is the OpenAI-shaped
APIs', not vision's. Fallback for a text-only main model, same signature: a second cheap call
to a vision model that returns *text* (WEB.md's `web_fetch(prompt=...)` pattern). Worse at
"read the exact number in this chart", which is why it is the fallback.

## The wire (`providers.py`)

`Message.content` becomes `str | list[Part]`, `Part` being `{"type":"input_text"}` or
`{"type":"input_image","image_url":...}` — Responses shapes, already this repo's internal
language. A bare `str` keeps working, so no existing call site, eval, or stored history
changes. `OpenAIProvider` passes parts through; `ChatProvider` gets one branch in
`_to_chat_messages` emitting `{"type":"image_url","image_url":{"url":...}}`.

`image_url` is a `data:image/jpeg;base64,...` URI, or an `https://` URL passed straight
through to the provider — we never open a socket, so `./test.sh lint` stays green.

## The converter (`images.py`)

Bytes in, bytes out, no filesystem — so the lint leaves it alone (`Image.open(BytesIO(...))`
does not match the `open(` pattern; the `.` is what saves it). Three wheels, no apt:
`pillow`, plus `pillow-heif` and `rawpy`, which carry libheif and LibRaw. `rawpy` is a
**guarded import** — a ~30MB wheel for a format most runs never see should report itself
missing, not fail at startup.

| in | handling |
|---|---|
| jpeg png | **passed through untouched** when already within both caps |
| heic heif avif / tiff bmp | Pillow (+ `pillow-heif`) -> RGB |
| cr2 nef arw dng raf … | `rawpy`; embedded JPEG preview first, demosaic only if absent |
| webp gif | always converted — spotty vLLM support, and GIF is animated |
| pdf | rasterized to one JPEG per page (below) |
| svg video | refused with a message; a renderer is a different dependency class |

```
longest edge -> 1568       what Anthropic's server resizes to anyway; tokens ~ w*h/750
under both caps?           pass through untouched — no re-encode, no loss
otherwise, in order:       format-preserving compression at full size
                           -> resize to 1568 -> compression ladder again
                           -> last resort 1000px JPEG q20
```

Output keeps the **source format** where it can, JPEG only as the fallback — Claude Code's
rule: `imageResizer.ts` tries palette PNG (`compressionLevel: 9, palette: true`, then
`colors: 64`) before it will turn a PNG into a JPEG, because transparency and crisp
screenshot text are what conversion destroys.

**Why a 500KB target and not Claude Code's 3.75MB.** Its cap exists to dodge the Anthropic
API's 5MB base64 *rejection* (`constants/apiLimits.ts`); bytes under that limit cost no
tokens. Ours exists because the base64 is re-uploaded in the request body on **every turn**
and this loop runs up to 120 of them — 60MB at 500KB, 450MB at 3.75MB.

The chosen rung goes in the text output, so a misread image can be told from a mangled one.
So does Claude Code's label: *"original 3024x4032, displayed at 1176x1568. Multiply
coordinates by 2.57 to map to original image."* Without it, a model asked to point at
something answers in the wrong space.

## PDF

A PDF page is an image, so this is one step in front of the pipeline above: rasterize at
100 DPI (a Letter page lands at 850x1100, already inside the cap), through the same
converter, out through the same seam. `pdf_read(path, pages=None)` — `"3-5"` or `[1,7]`,
default first 5, 5 per call. Anthropic, Gemini and OpenAI take PDFs natively; fireworks,
together and qwen take none, so rasterizing locally is the only path, not an optimization.

No text-layer extraction, even at ~800 tokens a page instead of ~2500: a page holding a
chart *and* a paragraph passes any "is there text here" test, takes the text path, and the
chart vanishes with nobody told. A page cap is a simpler lever and it fails loudly.

One wheel, `pypdfium2` (BSD, bundles PDFium) — page count and rasterizer in-process. Not
poppler (a subprocess would route through `runtime.run`), not PyMuPDF (AGPL). Validate
`%PDF-` before the file enters the conversation, as Claude Code does: once an invalid
document block is in the history every later call 400s, and our history is append-only, so
that is a whole `AgentRun` lost rather than one bad turn.

## Caps, gating, getting a file in

`image_view(paths: [str])` takes a list; `--image` repeats. **8 per call, 20 per run**, and
the run refuses rather than silently dropping. Every attachment carries its source path so
the model can say *which* image it means.

`Model` gains `vision: bool`, verified per platform by `./test.sh providers --live`, never
assumed. A text-only model refuses **before** the request — "glm-5.3-flash has no vision;
try: …" — not as a 400 in eval task 137. `FallbackProvider` needs nothing: same weights.

`run.sh` mounts nothing today and `_resolve` confines tools to `/work`, so `--image`
bind-mounts the file read-only at `/work/inputs/<name>` and passes the container path — the
smallest honest version of the roadmap's **mountable folders**, and its seed.

## The prompt cache, explained

Cross-checked against `../claude-code-source`: `services/api/claude.ts`
(`addCacheBreakpoints`, `stripExcessMediaItems`), `services/api/promptCacheBreakDetection.ts`,
`utils/toolResultStorage.ts`, `services/compact/`.

**The mental model.** A prompt cache is a *prefix* cache. The server hashes your messages
from the start and reuses its KV pages for the longest run of tokens byte-identical to last
time. So there are only two operations: **append** — free, everything before it stays
cached — and **edit something already sent** — expensive, every token after the edit point
is recomputed. Nothing else matters.

**So an image does not break the cache.** One attached at turn 12 leaves turns 1–11 cached.
Its tokens (~w*h/750, ~2.5k for a 1568px photo) are paid once and then ride in the prefix
like any others. Images are only expensive if you *move* them.

What Claude Code does, and what we copy:

- **One cache breakpoint, on the last message.** `addCacheBreakpoints` sets
  `markerIndex = messages.length - 1`: a single `cache_control` marker, always at the tail.
  Its comment says why not more — a second marker keeps KV pages alive at a position nothing
  will ever resume from. The Chat Completions platforms cache automatically so we have no
  marker to place, but the shape it enforces — *a prefix that only ever grows at the tail* —
  is the whole trick, and that part is ours to keep or lose.
- **Images are exempt from in-place rewriting.** `toolResultStorage.ts` skips persistence for
  image blocks ("they need to be sent as-is") and `collectCandidatesFromMessage` drops any
  block with `hasImageBlock` from the microcompact candidate set. The eviction scheme an
  earlier draft of this doc proposed — strip an image's parts after N turns, keep its text
  line — is exactly what Claude Code refuses to do: it invalidates everything downstream of
  the edit and costs more than the tokens it saves.
- **`[image]` is a summarizer placeholder, never a history entry.**
  `compact.ts:stripImagesFromMessages` swaps image blocks for the text `[image]` — but it
  `.map()`s into a throwaway array at the call site of the *summarization* request and the
  live history is untouched. The summarizer writes prose and does not need pixels; the main
  model does. Attaching `[image]` instead of the image, or downgrading an old one to save
  bytes, is an **edit to the cached prefix** and recomputes everything after it. Our
  equivalent label already exists in the right place: the `function_call_output` string
  (`attached 2 images: a.heic -> ...`) is the caption, the appended user message is the pixels.
- **The one path that does drop images is a boundary, not an in-place edit.**
  `stripExcessMediaItems` drops oldest-first past `API_MAX_MEDIA_PER_REQUEST = 100` — a hard
  API limit to avoid rejection, not a cost tactic, and far above what a session holds. So:
  when context truly runs out, compact once, explicitly, and re-seed the prefix.
- **The breaks that actually happen are upstream of the conversation.** A whole module exists
  to find them, and what it hashes is the system prompt, the per-tool schemas, the model, the
  beta headers, the effort setting, the extra body params — never the messages. A statistic
  recorded there: 77% of tool-caused breaks were a tool *description* changing with no tool
  added or removed. For us that means **`image_view`'s description must not vary with what is
  attached**, and neither must the system prompt.

The rules that fall out are all "don't":

- **Append only.** Never drop, summarize in place, or reorder.
- **Convert deterministically.** Same file, same bytes, same tokens. Memoize on
  `sha256(source) + rung` so a second look cannot re-encode to something subtly different.
- **Nothing variable in the text the model sees.** No durations, no temp paths, no
  timestamps. (`ToolResult.metadata` is safe: only `output` is sent.)
- **Deduplicate `image_view`.** A second call on the same path returns the text line and
  attaches nothing. `AgentRun.repeated_calls()` already counts this.
- **Measure.** `CACHED_TOKENS_HEADER` is already wired; whether a platform caches image
  tokens at all is a number to read, not to assume.

**The one thing a cache hit does not buy is upload.** The KV cache lives on the server; the
request body still carries every base64 payload on every turn. That, not tokens, is why the
converter targets 500KB and why the run cap is 20 images.

## Proving it works

**`./test.sh vision`** — no model, no network. Fixture HEIC, CR2, PNG, a text PDF, a scanned
PDF, and two lies (a `.png` that is a zip, a `.pdf` with no `%PDF-`) through the real
pipeline. Asserts: both wire shapes exactly; every output decodable and **under 500KB and
1568px whatever went in**; an in-budget JPEG byte-identical; the ladder's chosen rung per
fixture; a 3-page PDF yielding 3 attachments at 850x1100; the page budget and the text-only
refusal; and a bare `str` message serialized byte-for-byte as it is today.

**`./evals.sh --dataset imgqa`** — ~20 `(images, question, expected)` tasks, half multi-image,
with the mandatory control: a **blind run** with the images withheld. Any task answered blind
measures the weights, not vision, and leaves the set.

## Known traps

- **`/work/inputs/` is under `snapshot()`.** `git add -A` commits the blob. Needs a
  `.gitignore` line and a `SKIP_DIRS` entry, as WEB.md plans for `.web/`.
- **A data URI floods stderr.** The verbose trace dumps whole turns as JSON; elide payloads.
- **An image is untrusted input.** A screenshot reading "ignore previous instructions" is the
  injection surface WEB.md already flags; the attached message says so.
- **Generation loss compounds.** Never feed an `image_view` output back into `image_view`.
- **RAW previews lie.** The embedded JPEG is the camera's rendering, not sensor data — fine
  for "what is in this photo", wrong for "is this over-exposed". Log which path ran.

## Later

Crops and tiling for very large screenshots; a PDF text layer, if a real task ever runs out
of context where 800 tokens a page would have saved it; image *generation*; video. **Face
recognition** is a consent question before it is a plumbing one, and stays out until that is
answered.
