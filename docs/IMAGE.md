# Image input — vision in the loop

One seam, one tool, one converter. The goal is that *"what is wrong in this screenshot?"*
and *"compare these three photos"* work on a HEIC off a phone and a CR2 off a camera, on
the same six models the text loop already runs on.

```
./run.sh --image a.heic --image b.jpg "which one is sharper?"   session-level
image_view(paths, prompt=None)                                  agent-initiated
```

No image generation, no video, no OCR tier, no face recognition. Reasons below.

## The one hard constraint

**A tool result cannot carry an image.** `FunctionCallOutputItem.output` is a string, and
neither the Responses API nor Chat Completions accepts image content in a `tool`-role
message. So `image_view` cannot return the picture.

The fix is an attachment seam: a `ToolResult` may set `metadata["attachments"]`, its text
output stays a string (`attached 2 images: a.heic -> 1568x1176 jpeg, b.jpg -> ...`), and
the loop appends **one extra user message** carrying the image parts directly after the
tool output. ~15 lines in `loop.py`, and the main model sees actual pixels.

Fallback for a text-only main model, same tool and signature: a second cheap call to a
vision model that returns *text*, the `web_fetch(prompt=...)` pattern from WEB.md. Worse
at "read the exact number in this chart", which is why it is the fallback and not the path.

## The wire: content parts (`providers.py`)

`Message.content` becomes `str | list[Part]`, `Part` being `{"type":"input_text"}` or
`{"type":"input_image","image_url":...}` — Responses shapes, already this repo's internal
language. A bare `str` keeps working, so no existing call site, eval, or stored history
changes.

| provider | translation |
|---|---|
| `OpenAIProvider` | parts pass through unchanged |
| `ChatProvider` | one branch in `_to_chat_messages`: `{"type":"image_url","image_url":{"url":...}}` |

`image_url` is a `data:image/jpeg;base64,...` URI for a local file, or an `https://` URL
passed **straight through to the provider** — we never open a socket, so `./test.sh lint`
stays green and no `web.py` dependency appears. Hosted platforms fetch server-side; if the
self-hosted vLLM will not, remote URLs there wait on WEB.md's fetch seam.

## The converter: `images.py`

Bytes in, bytes out, no filesystem — so it is a new module the lint leaves alone
(`Image.open(BytesIO(...))` does not match the `open(` pattern; the `.` is what saves it).

```
sniff magic bytes
  jpeg png webp gif      -> pass through if within caps
  heic heif avif         -> pillow-heif -> RGB
  tiff bmp               -> Pillow -> RGB
  cr2 nef arw dng raf... -> rawpy: embedded JPEG preview first, demosaic only if absent
downscale longest edge to 1568        (~1.1-1.6k tokens; above it the model gains nothing)
re-encode JPEG q85, or PNG when alpha matters
still over 4MB -> q70, then 1024px, then refuse with a message the model can act on
```

Providers accept png/jpeg/webp/gif only, so **every other format is converted, not
rejected** — that is the whole point of the module. Three pip wheels, no apt:
`pillow`, `pillow-heif`, `rawpy` (LibRaw and libheif ride along in the wheels).

## Multiple images

`image_view(paths: [str])` takes a list; `--image` repeats. Caps: 8 per call, 20 per run,
and the run refuses rather than silently dropping. Every attachment is labelled with its
source path in the text output, so the model can say *which* image it means.

## Capability gating

`Model` gains `vision: bool`, verified per platform by `./test.sh providers --live`, never
assumed. A text-only model refuses **before** the request — "glm-5.3-flash has no vision;
try: ..." — not as a 400 in eval task 137. `FallbackProvider` needs nothing: same weights.

## Getting a host image in

`run.sh` mounts nothing today and `_resolve` confines tools to `/work`, so `--image`
bind-mounts the file read-only at `/work/inputs/<name>` and passes the container path.
The smallest honest version of the roadmap's **mountable folders**, and its seed.

## Proving it works

**`./test.sh vision`** — no model, no network. Fixture HEIC, CR2, PNG, and a lie (a `.png`
that is a zip) through the real pipeline: both wire shapes asserted exactly, conversion
output decodable at the expected size, caps enforced, text-only model refused, and a bare
`str` message serialized byte-for-byte as it is today.

**`./evals.sh --dataset imgqa`** — ~20 `(images, question, expected)` tasks, half of them
multi-image, with the mandatory control: a **blind run** with the images withheld. Any task
answered blind measures the weights, not vision, and leaves the set.

## Known traps

- **Images break the prompt cache** and are re-sent every turn. Ten `image_view` calls is a
  different bill from ten `fs_read`s. Cap per run; consider dropping an image's parts after
  N turns and keeping its text line.
- **`/work/inputs/` is under `snapshot()`.** `git add -A` commits the blob. Needs a
  `.gitignore` line and a `SKIP_DIRS` entry, as WEB.md plans for `.web/`.
- **A data URI floods stderr.** The verbose trace dumps whole turns as JSON; elide payloads.
- **An image is untrusted input.** A screenshot reading "ignore previous instructions" is
  the injection surface WEB.md already flags. The attached message says so.
- **RAW previews lie.** The embedded JPEG is the camera's rendering, not the sensor data —
  fine for "what is in this photo", wrong for "is this over-exposed". Log which path ran.

## Later

Pillow-based crops and tiling for very large screenshots; PDF pages as images; image
*generation*; video. **Face recognition** is on the roadmap line but is a consent and
identification question before it is a plumbing one, and stays out until that is answered.
