# Voice — talking to the agent from the terminal

`./voice.sh` turns the interactive chat into a conversation. You talk; the agent answers out
loud, says what it is doing while it works ("Alright, I'm using the shell run tool."), and
stops the moment you talk over it. The agent itself is unchanged and still runs in its
container. What is new is a thin audio program on the host and a line protocol into the
container.

```
 host (your Mac)                                                container (./run.sh)
 ┌───────────────────────────────────────────────┐   stdin     ┌──────────────────────────────┐
 │ mic ─ Silero VAD ─ end of turn ─ Whisper ─────┼───────────▶ │ voice/bridge.py ─▶ agent_loop │
 │ speaker ◀─ Qwen3-TTS ◀─ answer ───────────────┼◀─ stdout ── │   {"type":"answer","text":…} │
 │         ◀─ "Okay, file read tool now." ◀──────┼◀─ stdout ── │   {"type":"tool","name":…}   │
 └───────────────────────────────────────────────┘             └──────────────────────────────┘
```

Why the split: Docker on macOS has no audio device passthrough, so a microphone can only be
read by a process on the host. `voice/client.py` is that process, and the one exception to
"nothing runs on the host". It never sees the agent's tools, files or keys. It turns sound
into a line of text and a line of text into sound; everything in between is `./run.sh`.

## Running it

```
./voice.sh                                  talk. Enter interrupts, a typed line is a turn, Ctrl-C quits
./voice.sh --voice serena --lang en         another preset voice; what you speak, for Whisper
./voice.sh --instructions "casual, warm"    a style hint for the synthesizer
./voice.sh --endpoint-ms 450                end your turn after less silence (faster, cuts in more)
./voice.sh --list-devices                   then --mic / --speaker by name or index
PROVIDER=qwen QWEN_BASE_URL=https://api.lemontree.media/v1 ./voice.sh    the lowest-latency model
```

Needs `uv` (`brew install uv`); it installs the client's five dependencies (sounddevice,
numpy, onnxruntime, httpx, websockets) in its own cache the first time. The first run also
fetches the 2 MB Silero model and synthesizes the filler clips into `~/.cache/agent-voice/`.
macOS asks for microphone permission for your terminal app once, when the mic opens.

Wear a headset or AirPods. See *Limits* for why.

## Pieces

| file | runs | does |
|---|---|---|
| `voice/client.py` | host, via `uv run` | microphone, speaker, the turn-taking loop, spawns `./run.sh` |
| `voice/bridge.py` | container, as `AGENT_ENTRYPOINT=python ./run.sh -m voice.bridge` | one `agent_loop` turn per JSON line, history carried over |
| `voice/turns.py` | both | the decisions: endpointing, barge-in, what to say while waiting. Pure, tested |
| `voice/vad.py` | host | Silero VAD v5 through onnxruntime, one probability per 32 ms frame |
| `voice/speech.py` | host | Whisper and Qwen3-TTS against the gateway (`~/Documents/data/wikis/api-lemontree-media.md`) |
| `voice.sh` | host | `uv run voice/client.py` |

Two lines elsewhere make it possible: `run.sh` always attaches stdin (`--interactive`) so a
pipe reaches the container, and the Dockerfile copies `voice/` into the image.

## The protocol

The client writes one JSON object per line to the container's stdin and reads one per line
from its stdout. stderr stays the loop's quiet trace and is shown dimmed in the terminal.

```
→ {"type": "prompt", "text": "list the files"}      one turn; the conversation carries over
→ {"type": "heard",  "text": "There are three"}     you cut the last answer off after this much
← {"type": "ready"}
← {"type": "tool", "name": "fs_list"}                a tool call started
← {"type": "answer", "ok": true, "text": "...", "steps": 2, "stop_reason": "done"}
```

The bridge adds a *Voice mode* section to the system prompt: one to three spoken sentences,
no markdown or paths, anything long goes to a file. After a `heard`, the next prompt is
prefixed with what the user actually heard, so the model does not repeat itself or assume
you heard the rest. Tool events come from the trace itself: with `verbose=False` the loop
prints `  [fs_list]` for every call, and the bridge wraps its own stderr to turn those lines
into events. Nothing in `agent_loop/` changed for this.

## What makes it feel like a person

**Knowing when you are done.** Silero VAD gives a speech probability every 32 ms. Three
frames over 0.5 open a turn (a click does not); the turn closes after 600 ms under 0.35, with
hysteresis in between so a hesitant frame does not end it. A 380 ms pre-roll buffer keeps the
first syllable. Turns under 250 ms of speech are dropped as coughs, and Whisper's stock
outputs for near-silence ("Thank you.", "you", anything in brackets) are dropped too.

**Not waiting for the transcription.** At 250 ms of silence the audio so far is already on
its way to Whisper. If you keep talking the result is discarded and fired again at the next
pause; if the turn ends, the text is usually back before the 600 ms closes. Whisper takes
~130 ms and costs nothing, so a wasted request is cheaper than 130 ms of dead air.

**Signs of life while it works.** Voice has no spinner, so the client speaks one:

| when | what | rule |
|---|---|---|
| 400 ms after your turn, nothing yet | "Mm-hm." "Um, let me think." | once |
| a tool call starts | "Alright, I'm using the file read tool." | the first at once; then 2.5 s apart, the same tool again only after 8 s |
| 12 s with nothing said | "Still on it." | at most three times |

The lines are fixed text, synthesized once and cached as PCM, so they play instantly. Tool
names become words (`fs_read` → "file read", `get_today_date` → "get today's date"). A line
is never started while the answer is playing, and an answer waits up to 600 ms for a line
to finish rather than clip it.

**Interrupting.** The microphone stays live while the agent speaks. Speech over 0.6 for
250 ms cuts playback at once, drops the TTS socket (a fresh one answers in ~150 ms; draining
the rest would take longer), and your words, already buffered from before the cut, become
the next turn. The client splits the answer into sentences and sends each as its own
utterance on the same socket (the server treats one input as one utterance and synthesizes
six times faster than real time, so the next sentence lands while the previous one plays);
the speaker counts bytes played, so the client knows which sentences you heard and tells
the bridge. Speech that starts while the
agent is talking but never reaches the barge-in threshold (an "uh-huh") is ignored rather
than transcribed. In v1 an interruption stops the *speech*, not the agent's work: a turn
already running finishes, and what you said queues behind it.

## Latency

End of your speech to the agent's first sound, chat-style turn, gateway over Tailscale:

| stage | now | with Smart Turn |
|---|---|---|
| silence that ends the turn | 600 ms | ~200 ms |
| Whisper | ~0 (already in flight) | ~0 |
| agent turn: one `done` call, Qwen thinking off | 600–1200 ms | same |
| TTS first audio + network | ~170 ms (132–165 ms measured on the box) | same |
| **total**, filler heard at 400 ms | **1.4–2.0 s** | **1.0–1.6 s** |

Tool-using tasks take as long as they take; the tool announcements and the dimmed trace
cover that.

## Knobs

Flags above. In code, `voice/turns.py` holds every threshold as a dataclass field:
`EndpointConfig` (start/end probabilities and frames, `speculate_ms`, `end_ms`,
`min_speech_ms`), `BargeInConfig` (`prob`, `sustain_ms`) and `NarratorConfig`
(`ack_after_s`, `gap_s`, `repeat_tool_after_s`, `reassure_every_s`, `max_reassure`). The
filler and announcement lines are the `ACKS`, `ANNOUNCE`, `REASSURE` tuples next to them.
`AGENT_VOICE_CACHE` moves the cache.

## Limits, and what comes next

- **Speakers.** The agent's own voice is speech to the VAD. With a headset the leak is small
  and the stricter barge-in threshold absorbs it; with laptop speakers every answer would cut
  itself off. PortAudio cannot reach macOS's system echo canceller. Next: an echo gate that
  compares the microphone's energy envelope with the PCM being played (~40 lines), so
  speaker mode works too.
- **Fixed 600 ms endpoint.** Pipecat's open Smart Turn v3 (ONNX, ~65 ms on CPU) judges
  from the audio whether you finished or paused mid-thought; with it the wait can drop to
  ~200 ms when you clearly finished. The single biggest naturalness gain left.
- **Interrupting compute.** Cancelling a running agent turn needs a hook in the loop; until
  then, interruptions cut speech only.
- **The gateway's `/v1/tts/stream` alias** hangs on the websocket handshake through nginx
  (the backend also refuses it with 403); `/v1/audio/speech/stream` works on both, and is
  what the client uses.

## Testing

```
AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m tests.test_voice
```

covers the endpointer, barge-in, narrator, heard-text and transcript filters, and the bridge
protocol end to end with a stubbed loop (tool events, history, the interruption note, a
crash as an answer). The audio and network halves have no sound card in the container; they
were exercised on the GPU box against the local services (`voice_smoke.py` in the session
scratchpad, easy to recreate from `voice/speech.py`'s surface): websocket first audio
132–165 ms per sentence, a cancel after 400 ms leaves 0.64 s of audio and a dropped socket,
Silero at 0.14 ms per frame with the endpointer opening 150 ms into speech and closing
570 ms after it, Whisper at 178 ms for a 7 s utterance with a correct transcript and an
empty one for silence.
