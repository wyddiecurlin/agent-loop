---
name: debug-voice
description: Debug a failed or odd voice session — the Mimo phone app or ./voice.sh — from the per-run logs on the laptop or on lemontree. Use when the app said "I hit a problem", "Mimo couldn't finish that thought", "My voice cut out", "The voice connection stopped", or "Could not start the voice harness", or when a turn was slow, silent, or interrupted itself.
---

# Debugging a voice session

Read the logs. Do not reproduce on a phone until you have. Report the finding, then fix.

## Where the log is

Every session writes one run directory through `voice/log.py` (RunLog):

| how it ran | run directory |
|---|---|
| `./voice.sh` on this Mac | `logs/voice/<timestamp>/`, `logs/voice/latest` symlink |
| the Mimo app, through the gateway on lemontree | `~/Documents/agent-loop/logs/voice/<timestamp>/` on the box, same `latest` symlink |

The gateway's worker (`pet-moment/mobile-app/gateway/voice_worker.py`, class `RemoteLog`)
mirrors everything the harness logs into that RunLog. It only started doing so on
2026-09-09; sessions before that left nothing. Knobs, set in the gateway's environment:
`MIMO_VOICE_LOG=0` disables the mirror, `MIMO_VOICE_LOG_AUDIO=1` also keeps each
utterance as WAV (off by default).

Start here:

```
ssh lemontree 'cat ~/Documents/agent-loop/logs/voice/latest/session.log'
ssh lemontree 'ls -t ~/Documents/agent-loop/logs/voice | head'     # pick an older run by time
```

## What is in a run directory

- `session.log` — one timestamped line per event, human order. Read this first.
- `events.jsonl` — the same records as JSON, for `jq`/grep.
- `turns/NNN.json` — the bridge's trace of one agent turn: prompt, every message the loop
  added (tool calls and outputs), steps, stop reason, timing, usage, `error` if any.
- `audio/uNNN.wav` — each utterance as sent to Whisper (laptop runs, or with audio on).

Event kinds worth grepping, in the order a bad turn produces them:

```
grep -E ' (start|session|prompt|agent.err|turn|answer|error|exit) ' session.log
```

- `start` / `session` — args, provider, model, pid, max_output_tokens, mode.
- `prompt` — what STT heard and the loop was asked.
- `agent.err` — the container's stderr, line by line. Tracebacks from the loop or the
  provider land here, before the failed turn.
- `turn ... ok=False error=...` — the bridge's verdict; the full trace is in the named
  `turns/NNN.json`.
- `answer ok=False text=...` — the real error text. The phone never sees it.
- `error where=tts|stt|filler|main` — a client-side exception, traceback follows.
- `exit reason=...` — how the session ended (`exit 0`, `sigterm`, `crash: X`).

Latency: each `answer` carries `wait_s`; `preamble` carries `after_s`; `vad.*`/`stt.*`
events time the ear. An `agent.event` with `type=context_cleared` marks the loop dropping history.

## What the phone's message means

| the phone said | what happened | look at |
|---|---|---|
| "I hit a problem. The details are in the terminal." | the loop returned `ok: false` | `answer ok=False`, the `turn` before it, `agent.err` |
| "Mimo couldn't finish that thought." | same, filtered by the gateway's `public_event` | same |
| "My voice cut out. You can keep talking." | TTS raised | `error where=tts` and the traceback; TTS unit on the box |
| "I missed that. Could you say it again?" | STT raised or returned nothing | `error where=stt`, `stt.*` events; STT unit |
| "The voice connection stopped." | any other client exception | `error where=main`, `exit reason=crash` |
| "Could not start the voice harness." | the worker died before `client.run` | there may be no run dir; run the worker by hand (below) |
| "All voice companions are busy" | the gateway's session slot is taken | `journalctl`, stale worker processes |

## The gateway itself

The worker's stderr goes to /dev/null and the phone only gets an allowlist of events, so
the run dir is the only full record. What journald has is the gateway's own lines:
per-turn `speech=` and `emotion=` metrics, startup, HTTP errors.

```
ssh lemontree 'journalctl --user -u mimo-gateway -o short-iso -n 300 --no-pager'
ssh lemontree 'systemctl --user is-active mimo-gateway vllm stt tts'
```

Run the worker by hand when it will not start (needs the gateway's venv, prints JSON events):

```
ssh lemontree 'cd ~/Documents/pet-moment/mobile-app/gateway && .venv/bin/python voice_worker.py ~/Documents/agent-loop'
```

## Before blaming the code

1. The box's checkout is behind. The gateway sends `max_output_tokens`/`mode` on every
   prompt; a bridge older than 0684963 does not know them.
   `ssh lemontree 'cd ~/Documents/agent-loop && git fetch -q && git log --oneline HEAD..origin/main'`
   must print nothing.
2. The gateway on the box is behind too. `~/Documents/pet-moment` is a clone of
   LemonTree-Media-LLC/pet-moment; the same `git log HEAD..origin/main` check applies there.
3. vLLM is queueing: `curl -s http://100.89.9.78:9000/metrics | grep num_requests_waiting`
   from the box. Five voice sessions fill the KV cache.
4. The docker group. The unit wraps python in `sg docker` because the user manager started
   before mdeng joined the group; after a reboot that is harmless.

## Redeploy after a fix

Commit and push both repos first; the box pulls, nothing is rsync'd.

```
ssh lemontree 'cd ~/Documents/agent-loop && git pull --ff-only'     # agent-loop: no restart needed
../pet-moment/mobile-app/gateway/deploy-gateway.sh --test            # gateway: pull, test, restart, health 200
```

`deploy-gateway.sh` refuses to run with unpushed gateway commits and prints the unit's
status and last journal lines if the restart or the health check fails.

The container is rebuilt by `run.sh` on every session, so a pull takes effect on the next
turn; the worker imports `voice.client` from the checkout per connection, so only gateway
changes need the restart. Full topology and smoke tests: `docs/DEPLOY.md`.

## Reproducing on the laptop

`./voice.sh` runs the same client against the same box services and writes the same log
locally, with audio kept. Use it when the failure needs the utterance WAV. `voice.sh --help`
lists the knobs (`--log-dir`, `--no-log-audio`, `--lang`, `--max-turn-ms`).
