# Per-user settings and credentials

The launching client obtains a 15-minute container grant, saves the response to a private JSON file, and supplies that file to `agent-loop`. Trusted runtime code uses the grant to fetch the user's settings and credentials through the `container-access` backend. OAuth setup and provider-token refresh remain out of scope.

```text
Authenticated client -> container-access: create session, receive grant
Client -> launcher -> protected runtime file: session ID + token
Trusted provider/tool -> container-access -> ops: fetch one credential
```

**Storage and authentication.** The ops Supabase project (`ftlvnvrzsukhsjgeqebl`) owns `public.ivon_users` and `public.ivon_container_grants`. The [migration](../supabase/migrations/202609100001_container_credentials.sql) adds `container_settings` and `container_connections` JSON objects to the existing user row. Grants contain `session_id`, `user_id`, a SHA-256 hash of a random 256-bit token, expiry, revocation, and creation time. Each session has one current grant, so renewal invalidates its previous token without affecting other sessions.

The [backend](../supabase/functions/container-access/service.ts) verifies an existing Supabase user JWT or Ivon application API token when issuing, renewing, or resuming grants. It derives ownership from that authentication. Container requests instead carry their opaque grant and `X-Session-ID`; they cannot issue grants, renew themselves, or modify connections. Supabase service-role access stays in the backend, which checks ownership explicitly. Database grants/RLS block direct client access to user secrets and container grants. See [Supabase function authentication](https://supabase.com/docs/guides/functions/auth) and [RLS](https://supabase.com/docs/guides/database/postgres/row-level-security).

New credentials use AES-256-GCM encryption with the user and connection IDs authenticated alongside the ciphertext. The encryption key stays in the backend's `IVON_CREDENTIAL_KEY` secret. The existing Facebook connection is available as `facebook`, using the current Ivon columns and expiry. Existing Facebook and application API tokens retain their current storage format for compatibility with Ivon; this migration does not convert those legacy values.

**API.** Paths below are relative to `/functions/v1/container-access`. All requests require `Authorization: Bearer <credential>`. Container requests also require `X-Session-ID`. Responses disable caching.

| Method and path | Caller | Result |
|---|---|---|
| `POST /sessions` | Application | `{session_id, user_id, token, expires_at}` |
| `POST /sessions/{id}/renew` | Application owner | Replacement grant for an unrevoked session |
| `POST /sessions/{id}/resume` | Application owner | Fresh grant, including for a previously ended session |
| `DELETE /sessions/{id}` | Application owner or that session's grant | Revoke the grant |
| `GET /settings` | Application or container | User ID and nonsecret settings |
| `PUT /settings` | Application | Replace settings: `timezone`, `provider`, `model`, `web_search_provider` |
| `GET /connections` | Application or container | Connection IDs, providers, account labels, and expiry |
| `PUT /connections/{id}` | Application | Store `{provider, account_label, credential, expires_at?}` |
| `DELETE /connections/{id}` | Application | Remove a stored connection |
| `GET /connections/{id}/credential` | Container | One credential, grant expiry, and provider expiry |

Use connection IDs `openai`, `fireworks`, `together`, `qwen`, `brave`, and `parallel` for the built-in clients. Other IDs are available to trusted integrations through `runtime.credentials.credential(id)`. Manage the existing `facebook` connection through Ivon's current UI. Settings and connection writes are for an authenticated application client; raw credentials are never registered as model tools.

**Launch and lifetime.**

```bash
./agent-loop --grant-file /private/session-grant.json \
  --backend-url https://ftlvnvrzsukhsjgeqebl.supabase.co/functions/v1/container-access
```

The client renews before expiry and atomically replaces its grant file with the response. Before each lookup, the launcher reads the latest file, checks its session ID, and updates the protected runtime file. Each outbound model request and search performs a new credential lookup. Missing, expired, or revoked credentials fail closed; an authenticated client must renew before retrying. Grant expiry cannot invalidate a provider credential already sent to a provider.

The launcher revokes the grant at session end. Resume requires the application's `/resume` response for the same session ID. Credentials live outside `/work`, in a protected mount excluded from Docker snapshots. The model's shell uses a separate user and an environment allowlist. Provider keys stay out of SDK configuration, model tools, logs, prompts, conversation state, and saved container configuration.

**Deployment.** Apply the migration to ops, configure a base64-encoded 32-byte `IVON_CREDENTIAL_KEY` as an Edge Function secret, then deploy `container-access` with the supplied [function configuration](../supabase/config.toml). The function uses Supabase's injected `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. Preserve the encryption key across deployments. The repository contains deployment artifacts; no remote migration or function deployment is performed by the launcher.

**Validation.** Run `./test.sh credentials` for the handler, PostgreSQL migration/permissions, and runtime grant delivery tests. [SESSIONS.md](SESSIONS.md) covers container persistence and its real Docker tests. `./run.sh` and explicit `./agent-loop --local` retain `.env` credentials for local development.
