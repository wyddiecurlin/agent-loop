# Voice — talking to the agent from the terminal and iOS

`./voice.sh` turns the interactive chat into a conversation. You talk; a small robot answers
out loud, says in its own words what it is about to do when that will take a moment ("Let
me check the current listings."), and stops the moment you talk over it. The agent itself
is unchanged and still runs in its container. What is new is a thin audio program on the
host, a line protocol into the container, and one effect chain that makes every sound
out of the speaker the same robot.

```
 host (your Mac)                                                container (./run.sh)
 ┌───────────────────────────────────────────────┐   stdin     ┌──────────────────────────────┐
 │ mic ─ Silero VAD ─ end of turn ─ Whisper ─────┼───────────▶ │ voice/bridge.py ─▶ agent_loop │
 │ speaker ◀─ robot ◀─ Qwen3-TTS ◀─ answer ──────┼◀─ stdout ── │   {"type":"answer","text":…} │
 │         ◀─ robot ◀─ Qwen3-TTS ◀─ preamble ────┼◀─ stdout ── │   {"type":"preamble","text":…}│
 │         ◀─ robot ◀─ "Um," (cached) ◀──────────┼◀─ stdout ── │   {"type":"tool","name":…}   │
 └───────────────────────────────────────────────┘             └──────────────────────────────┘
```

Why the split: Docker on macOS has no audio device passthrough, so a microphone can only be
read by a process on the host. `voice/client.py` is that process, and the one exception to
"nothing runs on the host". It never sees the agent's tools, files or keys. It turns sound
into a line of text and a line of text into sound; everything in between is `./run.sh`.

## Running it

```
./voice.sh                                  talk. Enter interrupts, a typed line is a turn, Ctrl-C quits
./voice.sh --voice ryan --lang en           another preset under the robot; what you speak, for Whisper
./voice.sh --robot 0.6                      less robot (0 is the plain preset, 1 the default)
./voice.sh --instructions "..."             replace the synthesizer's delivery notes
./voice.sh --endpoint-ms 450                end your turn after less silence (faster, cuts in more)
./voice.sh --list-devices                   then --mic / --speaker by name or index
PROVIDER=qwen QWEN_BASE_URL=https://api.lemontree.media/v1 ./voice.sh    the lowest-latency model
```

Needs `uv` (`brew install uv`); it installs the client's five dependencies (sounddevice,
numpy, onnxruntime, httpx, websockets) in its own cache the first time. The first run also
fetches the 2 MB Silero model and synthesizes the filler and hold clips into
`~/.cache/agent-voice/`.
macOS asks for microphone permission for your terminal app once, when the mic opens.

Laptop speakers are fine: the client knows what it just played and refuses to hear it
again (see *The echo gate*). To interrupt, speak up a little over the robot.

## Pieces

| file | runs | does |
|---|---|---|
| `voice/client.py` | host, via `uv run` | microphone, speaker, the turn-taking loop, spawns `./run.sh` |
| `voice/robot.py` | host | the robot: one effect chain over every byte the speaker plays |
| `voice/echo.py` | host (tested in the container) | the echo gate: our own voice coming back is silence |
| `voice/bridge.py` | container, as `AGENT_ENTRYPOINT=python ./run.sh -m voice.bridge` | one `agent_loop` turn per JSON line, history carried over, the preamble call beside it |
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
→ {"type": "clock",  "tz": "America/Los_Angeles"}   the host's zone, once, so the clock in the
                                                    system prompt reads like yours
→ {"type": "prompt", "text": "list the files"}      one turn; the conversation carries over
→ {"type": "heard",  "text": "There are three"}     you cut the last answer off after this much
← {"type": "ready"}
← {"type": "preamble", "text": "One sec, I'll look."} the model's own line, while the turn runs
← {"type": "tool", "name": "fs_list"}                a tool call started
← {"type": "answer", "ok": true, "text": "...", "steps": 2, "stop_reason": "done"}
```

The bridge adds a *Voice mode* section to the system prompt: one to three spoken sentences,
in character, no markdown or paths, no written laughter or emoji, anything long goes to a
file. After a `heard`, the next prompt is prefixed with what the user actually heard, so the
model does not repeat itself or assume you heard the rest. Tool events come from the trace
itself: with `verbose=False` the loop prints `  [fs_list]` for every call, and the bridge
wraps its own stderr to turn those lines into events.

**The preamble.** The moment a prompt arrives, the bridge makes a second, tool-less model
call on a thread beside the real turn, with the same character and the recent conversation,
and asks one thing: will this take a moment, and if so, what is the one casual sentence to
say meanwhile? Small talk and anything answerable at once get `NONE`; "what's in theaters?"
gets "Let me check the current listings." The model's `NONE` is not trusted on its own
(a small model will happily "preamble" a reply to small talk): the line is held until the
real turn calls its first tool, and dropped if the turn finishes without one or finishes
first. When it was said it is written into the history as the assistant's own line, so the
answer does not say it again. The call
is made with thinking off and a token cap: left as the main loop has it, a self-hosted
model with `QWEN_THINKING=1` reasoned for twenty seconds about a twelve-word job.

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

**Signs of life while it works.** Voice has no spinner. Three things fill the gap, in
order, and none of them is a sentence about machinery:

| when | what | who decides |
|---|---|---|
| 450 ms after your turn, nothing heard yet | "Uh," "Um," "Hmm." "Mm," or, deliberately, nothing | the client, from a cached clip; once |
| the turn will take a moment | "Let me see what's out this week." | the model, live (the preamble above) |
| 7 s of quiet after the last thing you heard, then every 15 s | "Still digging through the web, hold on a sec." / "Still going through the files, one sec." / "Still running that, hang on." | the client, from a cached clip, by the family of the last tool; at most three |

Tool calls are never announced. The fixed lines are synthesized once and cached as PCM, so
they play instantly; the preamble is streamed like an answer, and the answer queues behind
it rather than cutting it off. A clip is never started while an answer is playing, and an
answer waits up to 600 ms for a clip to finish rather than clip it.

**The robot.** The synthesizer decides the words and the delivery (its instructions describe
a small, cheerful male robot that never laughs); `voice/robot.py` decides the timbre, and it
is applied to every byte the speaker plays, so a filler, the preamble and an answer are one
voice. The chain: a band-pass for the small speaker in its chest, a slow shallow vibrato for
the analog warble, a low ring-modulator carrier mixed under the dry voice for the metal, a
soft clip, fewer bits, a whisper of hiss. It carries its state across chunks, keeps the byte
count (so the offsets `heard` relies on still mean the same thing) and is deterministic:
the same input is the same robot every time. `--robot` scales it, `--voice` picks the preset
underneath (aiden by default; `GET /v1/audio/voices` on the gateway lists the rest). Renders
from the gateway are occasionally flaky: the same line has come back as forty seconds of
sound, which the stream's duration guard cuts off.

**The echo gate.** A container cannot reach macOS's echo canceller, and a laptop's
speakers are inches from its microphone, so without help every answer is heard again as
a turn. The full cancellation problem is not needed to stop that. The speaker callback
reports the level of every block as it leaves; the gate predicts the echo's level at the
microphone one lag later, scaled by the room's gain, and a microphone frame during
playback (plus a 300 ms tail) counts as speech only when it exceeds four times that
prediction plus the estimated room floor. Rejected frames are replaced with zeros
**before** the recurrent VAD, preroll, and transcription buffers. Overriding only the
VAD probability is insufficient: it leaves the robot's voice in VAD state and in audio
that can later be transcribed. Lag and gain start from a measured
prior (130 ms and gain 0.25) and are re-fitted every second
of playback by cross-correlating the two envelopes; the boot chirp at startup is the
first fit. The cost is that interrupting takes a little more voice than with a headset:
the gate is a level test, and a whisper under a loud robot is the robot. `--no-echo-gate`
turns it off, for a headset or for debugging; the `echo` event in the run log shows the
fitted lag, gain, floor and how many frames were suppressed. Two rules close the gaps the
gate itself cannot: an utterance that began while the robot was talking and never reached
the barge-in threshold is dropped, and so is one that opened on a noise a moment before
the robot spoke and had no real speech before it, since the rest of what it holds is the
robot. Measured with the laptop's own speakers at 55% volume: two turns, 261 frames
suppressed, no interruption and no phantom turn.

**No laughs.** Written laughter in any form (haha, ahahaha, lol, 哈哈, emoji) is deleted from
the text before synthesis, whatever the model wrote; the character forbids it in the system
prompt, the voice rules repeat it, and the synthesizer is told never to laugh, giggle,
chuckle, snort or sigh.

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
| **total**, filler heard at 450 ms | **1.4–2.0 s** | **1.0–1.6 s** |

Tool-using tasks take as long as they take; the preamble lands about 300 ms after the prompt
(measured on the qwen box, before the first tool call) and the hold lines cover the rest.

## Knobs

Flags above. In code, `voice/turns.py` holds every threshold as a dataclass field:
`EndpointConfig` (start/end probabilities and frames, `speculate_ms`, `end_ms`,
`min_speech_ms`), `BargeInConfig` (`prob`, `sustain_ms`), `NarratorConfig`
(`filler_after_s`, `hold_after_s`, `hold_every_s`, `max_hold`) and, in `voice/echo.py`,
`EchoConfig` (`lag_s`, `gain`, `margin`, `tail_s`). The filler and hold lines
are `STATUS_LINES`, per language. The robot's numbers are `RobotConfig` at the top of
`voice/robot.py`. The preamble prompt is `PREAMBLE_PROMPT` in `voice/bridge.py`.
`AGENT_VOICE_CACHE` moves the cache.

## Mobile iOS / Mimo

`../pet-moment/mobile-app/gateway/voice_worker.py` adapts the same `voice.client.run`
loop to the iOS app. The phone streams 16 kHz mono PCM16 and plays 24 kHz robot audio.
The harness still owns VAD, endpointing, STT, TTS, and barge-in. Do not run a second
desktop `voice/client.py` listener for the mobile app: it creates a separate conversation
and can hear the phone's output without driving its eyes.

The audio path is:

```
iOS voice processing → remote microphone → harness echo gate → foreground focus
                    → Silero VAD / endpointing → STT → agent
agent → TTS → robot effect → phone player → playback acknowledgement → echo reference
```

**Echo protection.** iOS uses one AVAudioEngine with voice processing enabled on its
input/output path. Bypass remains off, including after audio-route recovery. Automatic
gain control is disabled so quiet room conversation is not raised toward the nearby
speaker's level during pauses. The mobile adapter also enables the existing harness
EchoGate; iOS processing alone was insufficient in the Simulator setup.

The remote speaker stores a small RMS envelope of each post-robot PCM chunk. Only the
phone's `dataPlayedBack` acknowledgement feeds that envelope into EchoGate. The adapter
subtracts the chunk duration to estimate block start from completion time; synthesis or
network arrival does not mean audio was heard. Stale, duplicate, and nonboundary
acknowledgements are ignored. Stopping playback drops queued envelopes while preserving
the acoustic tail of audio already played. A reentrant lock serializes reference updates
and gate decisions across the transport and harness threads. This uses arrival-clock
estimates, so device/network latency and room acoustics still need route-specific testing.

**Background conversation.** Echo rejection is not background-speaker separation. The
mobile adapter's `gateway/foreground.py` applies level-based foreground focus after echo
detection. Its threshold is the greater of a configured minimum RMS and three times a
rolling background estimate. Two strong 32 ms frames open focus. A 96 ms lookahead
preserves a soft word onset; a 320 ms hold preserves softer syllables and brief pauses.
Rejected audio becomes zeros before both VAD and recognition, with sample timing intact.
The adapter's optional `Mic.process(frame, is_echo=...)` hook returns the focused frame
and its suppression flag. The standard desktop microphone has no focus hook and keeps
its existing behavior. No additional turn or interruption state machine is introduced.

| Gateway setting | Behavior |
|---|---|
| `MIMO_FOREGROUND_MIN_RMS=0.006` | Default foreground threshold floor on normalized PCM. |
| `MIMO_FOREGROUND_MIN_RMS=0.012` | Stronger focus used for the current Mac Simulator room with background conversation. |
| `MIMO_VOICE_DIAGNOSTICS=1` | Opt-in numeric microphone levels, focus counts/thresholds, echo fit/suppression counts, and allowlisted stage names. |

Set these on the **mobile gateway process**, then reconnect the app to create a worker
with the new configuration. The minimum must be finite and between 0.0005 and 0.1.
Raising it rejects more distant speech but can miss soft or far-away user speech; lowering
it does the reverse. The filter does not identify the owner and cannot reliably separate
two people speaking at similar levels. Do not keep raising it until all voices disappear.
Validate normal nearby speech and interruptions as well as idle room rejection.
On supported physical devices, the user can also select Apple's
[Voice Isolation microphone mode](https://support.apple.com/101993) for additional
background filtering. Simulator results do not establish physical-device microphone or
acoustic echo-cancellation quality.

**Recovery and diagnostics.** Mimo rebuilds its microphone format/converter after an
audio-engine configuration change. A microphone-buffer watchdog triggers bounded
recovery instead of leaving a stopped engine labelled “I'm listening.” Recovery keeps
the harness epoch, reschedules unacknowledged playback, and invalidates old callbacks.
Mute, sleep, and backgrounding stop capture and cancel recovery. The mobile gateway's
diagnostics exclude PCM, transcript text, credentials, and upstream traces. Increasing
microphone byte counts prove transport only; focus/VAD/transcript stages distinguish
filtering from actual recognition.

Mobile regressions live beside the adapter:

```
cd ../pet-moment/mobile-app
gateway/.venv/bin/python -m unittest discover -s gateway -v
```

They cover echoed audio being silenced before VAD, foreground onset/tail preservation,
quiet chatter rejection, invalid input, playback-reference epochs, and stronger user
speech interrupting the actual harness after TTS has finished generating. The iOS
`testLiveSpeakerDoesNotBecomeAnotherUserTurn` is opt-in with `MIMO_LIVE_TEST_ECHO=1`,
`MIMO_LIVE_TEST_ENDPOINT`, and `MIMO_LIVE_TEST_TOKEN` in the test environment. It captures
the real microphone, plays a fixed phrase, and watches for another user turn for 15
seconds after speech begins. A nearby human speaking during it legitimately fails this
check; it is not proof of echo by itself. Test human speech and acoustic barge-in
separately. See the mobile README for playback, microphone-recovery, and animation tests.

Validation on 2026-09-09: 21 mobile gateway regressions and 17 host-audio regressions
passed. The live Simulator playback/no-extra-turn test passed at minimum RMS 0.012.
Earlier runs in the room with background conversations at the lower threshold produced
additional turns; those failures do not by themselves distinguish echo from people.
The successful run establishes rejection for the observed setup, not owner recognition
or acoustic barge-in quality on every device. Normal user speech needs a separate check.

## Limits, and what comes next

- **Speakers.** Handled by the echo gate above, at the level of energy, not of the
  signal: it stops the robot hearing itself, it does not subtract it. Talking quietly over a
  loud robot is not heard until the robot pauses. A real canceller (NLMS on the signal)
  would recover that; it is more code than it is worth until someone misses it.
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

covers the endpointer, barge-in, narrator, echo gate (a synthetic room, learned from wrong
priors), heard-text, transcript and laughter filters, the
preamble call against a fake provider, and the bridge protocol end to end with a stubbed
loop (the preamble beside a turn and a late one never emitted, tool events, history, the
clock, the interruption note, a crash as an answer). `uv run tests/test_voice_audio.py`
covers the host half with fake sockets and speakers, the robot chain included (chunked
equals whole, byte count kept, silence in is silence out). The audio and network halves
have no sound card in the container; they were exercised on the GPU box against the local
services (`voice_smoke.py` in the session scratchpad, easy to recreate from
`voice/speech.py`'s surface): websocket first audio
132–165 ms per sentence, a cancel after 400 ms leaves 0.64 s of audio and a dropped socket,
Silero at 0.14 ms per frame with the endpointer opening 150 ms into speech and closing
570 ms after it, Whisper at 178 ms for a 7 s utterance with a correct transcript and an
empty one for silence.
