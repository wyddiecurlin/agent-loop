# Deploy — where the loop runs when a phone is the client

One box. The agent containers, the Mimo gateway and the three models all run on the home
GPU machine (`lemontree`, RTX 3090 Ti, Tailscale-only). The phone makes one hop; every
model call is a localhost hop.

```
  iPhone ──Tailscale/HTTPS──▶ nginx  mimo.lemontree.media  ─▶ gateway :8787 (systemd --user)
                                                                 │  spawns run.sh per session
                                                                 ▼
                                             ┌─ agent-loop container ───────────────┐
                                             │  voice.bridge  (PID 1)               │
                                             │  LLM ─▶ 172.17.0.1:9000  (vLLM)      │
                                             └──────────────────────────────────────┘
                                                 gateway process ─▶ STT :9001, TTS :9002
```

Measured 2026-09-09 from a Mac on the tailnet, through nginx: a one-shot task returns in
2.4 s including container start; a warm voice session is `ready` in 1.6–1.7 s; a text turn
reaches its first spoken sentence in 1.1–1.2 s. Turn latency is now the model's generation
time, not the network (see `docs/VOICE.md`, Latency).

## Why the gateway sits on the GPU box

The gateway does not use the GPU. It is there because of what it *does*, and because of
what the phone cannot do:

- **Something must create the container.** The agent never creates the box it stands in
  (`docs/RUNTIME.md`), and a phone cannot run `docker run`. The gateway is the launcher for
  the mobile client: it spawns `run.sh -m voice.bridge` per session and holds the stdin /
  stdout pipe that *is* the bridge protocol.
- **It runs the voice harness.** VAD, endpointing, the echo gate, STT and TTS calls and the
  robot effect all live in `voice/client.py`, which needs a Python process with the audio
  math, not a phone. The phone only ships PCM in and plays PCM out.
- **It makes four model calls per turn.** STT, the loop, the preamble and the emotion
  classifier, then the TTS stream. Wherever the gateway runs, those calls originate there.
  Next to the models they are loopback; on the Mac each one crossed the tailnet, and the Mac
  had to be awake for the phone to work at all.
- **It is the auth boundary.** One bearer token, checked on every route; provider keys stay
  in the container's `.env` and never reach the phone.

So the requirement is "an always-on Linux host with Docker, as close to the models as
possible", and the GPU box is the only machine that is all three. A second small box on the
LAN would also work, at one LAN hop per model call, if the GPU box should ever do nothing
but serve models.

## Why not Modal

The models are on the 3090 at $0 per token. Orchestration on Modal would call home over the
internet for every model request and need the API exposed publicly. Moving the models to
Modal means three GPU functions kept warm (`min_containers=1`, about $0.80/h per L4), because
a vLLM cold start is 30–90 s and fatal for voice. Modal is a rung on the scale ladder below,
not the base.

## What is on the box

| piece | where | notes |
|---|---|---|
| this repo | `~/Documents/agent-loop`, branch `main` | `run.sh` rebuilds the image per session; warm rebuild 0.5 s |
| `.env` | same dir, mode 600 | `PROVIDER=qwen`, `QWEN_BASE_URL=http://172.17.0.1:9000/v1`, `QWEN_THINKING=0`; Fireworks/Together keys kept for fallback |
| gateway | `~/Documents/pet-moment/mobile-app/gateway`, Python 3.12 venv via `uv` | rsync'd from the Mac; the directory is untracked in pet-moment |
| gateway unit | `~/.config/systemd/user/mimo-gateway.service` | `127.0.0.1:8787`; token and `VOICE_API` in `~/.config/mimo/gateway.env` |
| nginx vhost | `~/vllm/deploy/mimo.lemontree.media` → `/etc/nginx/sites-enabled/` | Tailscale ACL, wildcard cert; `/v1/voice` through `ws-proxy.conf`, the rest through `qwen-proxy.conf` |
| root steps | `sudo ~/vllm/deploy/install-mimo-nginx.sh` | ufw `docker0 → :9000`, `libportaudio2`, vhost, dnsmasq entry |
| models | `systemd --user` units `vllm`, `stt`, `tts` | documented on the box in `~/Documents/data/wikis/api-lemontree-media.md` |

Two things that bit:

- **The container reaches vLLM on the Docker bridge, not through nginx.** nginx admits only
  `100.64.0.0/10`, and a container's source address is `172.17.x`. vLLM listens on
  `0.0.0.0:9000`, so `172.17.0.1:9000` works once ufw allows `in on docker0 to any port 9000`.
  STT and TTS are called by the gateway process, outside the container, so `VOICE_API` stays
  `https://api.lemontree.media/v1`.
- **The user manager predates the `docker` group.** `systemd --user` services inherit the
  groups the manager had at boot; `mdeng` joined `docker` later, so a login shell could run
  `docker` and the unit could not. `ExecStart` wraps Python in `sg docker -c 'exec …'`.
  `sg` forks, so `KillMode=control-group` so Python still gets SIGTERM and shuts its workers
  down. A reboot makes the wrapper unnecessary; it stays harmless.

## Redeploy

```
ssh lemontree 'cd ~/Documents/agent-loop && git pull --ff-only'
rsync -a --exclude .venv --exclude __pycache__ \
  ../pet-moment/mobile-app/gateway/ lemontree:~/Documents/pet-moment/mobile-app/gateway/
ssh lemontree 'systemctl --user restart mimo-gateway && cd ~/Documents/pet-moment/mobile-app/gateway && .venv/bin/python -m unittest test_server'
```

Smoke test from any tailnet machine, with the token from `~/.config/mimo/gateway.env`:

```
curl -s -H "Authorization: Bearer $TOKEN" https://mimo.lemontree.media/health
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  https://mimo.lemontree.media/v1/task -d '{"text":"reply with the single word pong"}'
```

`/v1/task` runs one `./run.sh "<text>"` in a fresh container and returns its result JSON
without the per-call provider records. `/v1/turn` is the session form; `/v1/voice` is the
phone's WebSocket.

## Access for others

- **Tailscale users.** Invite them to the tailnet; split DNS and the nginx ACL then just work.
  No new infrastructure.
- **Cloudflare Tunnel** to a public `mimo.lemontree.media` → `127.0.0.1:8787` for someone who
  cannot run Tailscale. The zone is already on Cloudflare; WebSockets pass through. Keep the
  bearer token, and add per-user tokens to the gateway before going beyond friends.

## Scale ladder

The wall is VRAM: 23.5 of 24 GB in use (LLM 13.5 GB at a 0.60 budget, TTS 8.8 GB, STT
1.2 GB). The KV cache holds ~45k tokens, so about five concurrent voice sessions, each
running three streams (loop, preamble, emotions). STT serializes at ~130 ms per request.

| when | do | cost |
|---|---|---|
| LLM queueing (`/metrics`: `vllm:num_requests_waiting`) | `PROVIDER=fireworks` for the loop, local STT/TTS stay; frees 13.5 GB | per token, +200–400 ms to first token |
| STT or TTS queueing | faster-whisper and Qwen3-TTS on Modal behind the same OpenAI-shaped paths, `min_containers=1`; swap `VOICE_API` | ~$0.80/h per warm L4 |
| more than ~20 sessions per box | launcher on `modal.Sandbox` (image from this Dockerfile, stdin/stdout streamed) and the gateway as a Modal web server; nothing above the launcher changes | per-second CPU/RAM, no idle cost |
