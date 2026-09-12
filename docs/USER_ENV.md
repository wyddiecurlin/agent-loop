# Per-user settings and credentials

For v1, the client launching an agent-loop container must hold a short-lived token for that user's session. The container uses it to fetch the user's settings, API keys, and existing connections through a backend backed by Supabase. OAuth setup and provider-token refresh are out of scope; use credentials already stored for the user.

**Flow.**

1. The authenticated client requests a session. The backend derives the user from existing application authentication, creates a session ID, and returns a random opaque bearer token bound to that user and session. Proposed lifetime: 15 minutes.
2. The client supplies the token before launching the container. The launcher delivers it to trusted runtime code through a protected temporary mounted file outside `/work`, excluded from snapshots. Keep it out of command arguments and saved container configuration.
3. The runtime calls the backend over HTTPS to read settings, list connections, or fetch a credential by connection ID. On every request, the backend checks the token hash, expiry, revocation, session binding, and connection ownership. The token determines the user; callers cannot supply another `user_id`. Return only the requested data, never the full user row.

```text
Authenticated client -> backend: create session, receive token
Client -> launcher -> container runtime: session ID + token
Container runtime -> backend -> Supabase: fetch user's credential
```

**Storage.** Use `public.ivon_users` in the **ops Supabase project** for per-user settings, API keys, provider tokens, and connection metadata. A connection needs an ID, provider, account label, credential, and optional expiry. Reuse existing columns where possible; encrypt credential values with a backend-managed key outside the database.

Store container grants in `public.ivon_container_grants` in ops with `user_id`, `session_id`, `token_hash`, `expires_at`, and `revoked_at`. Store only token hashes. Separate grants support concurrent sessions and independent revocation without replacing the user's application API token. Exact schema additions require inspection of the existing ops table.

Only the backend accesses secret columns and grants. Keep Supabase secret/service-role keys there, with explicit ownership checks because they bypass RLS. Use database grants and RLS to block direct client access to secrets. See [Supabase API keys](https://supabase.com/docs/guides/getting-started/api-keys) and [RLS](https://supabase.com/docs/guides/database/postgres/row-level-security).

**Lifetime.** The authenticated client obtains and delivers replacement grants for longer sessions; the container token cannot renew itself. Expired or revoked grants deny further lookups. Pause credential-dependent work if renewal is unavailable, revoke grants at session end, and require a fresh grant on resume. Grant expiry does not invalidate provider credentials already fetched.

**Runtime boundary.** Trusted tools keep fetched credentials in memory, dropping them when the grant expires or revocation is detected. Preserve the shell environment allowlist and sandbox-user separation. Keep secrets out of tool results, prompts, logs, memory records, and snapshots. [SESSIONS.md](SESSIONS.md) owns container persistence; this plan owns credential storage and access.

**Delivery.** Implement backend issuance/lookup, launcher token delivery, then runtime credential loading. Verify cross-user/session denial, expiry/revocation, concurrent sessions, renewal/resume, unavailable provider credentials, and secret isolation from shell access and snapshots. This is proposed work; the current launcher still loads a shared `.env`.
