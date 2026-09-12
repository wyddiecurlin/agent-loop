# User context and memory

- Separate session history (messages/tools), compacted working context (a shorter continuation), and durable memory (user-requested preferences/facts reused across sessions).
- Add memory records to the host SQLite store in [SESSIONS.md](SESSIONS.md), sharing verified ownership with [USER_ENV.md](USER_ENV.md). Start with scoped rows and text search; no separate service or vector database.
- Support remember, list, correct, and forget through Runtime-backed operations. Retrieve a small relevant set at each turn; never promote summaries or workspace files into trusted profile or authorization data.
- Expose the same operations through CLI, Mimo, and a future browser voice client; make saved facts and their source visible and editable.

**Today.** [CLI](../agent_loop/__main__.py) and [voice bridge](../voice/bridge.py) hold conversation history in memory. [ContextBudget](../agent_loop/context.py) can clear previous turns; its compaction mode is currently a placeholder. There is no durable user-memory store. [COMPACTION.md](COMPACTION.md) owns shortening active context; it must not silently create permanent memories.

**Smallest implementation.** Store an ID, verified owner, optional project scope, fact/preference, source message/session, timestamps, and revision/status. Require an explicit user request to persist a fact initially. Model suggestions remain unconfirmed; retrieved text is contextual data, not instructions. Runtime mediates container access to the trusted launcher store; model-written files cannot set identity, change permissions, or overwrite trusted records. Credentials stay in USER_ENV's protected store. Keep authoritative settings such as timezone there too; memory may describe a preference without changing those settings.

Filter by owner/project before keyword search, then apply relevance, recency, and a token cap. A correction supersedes its earlier revision; unresolved conflicting claims retain their sources and trigger clarification instead of a silent overwrite. Forget removes the saved content from retrieval and indexes, invalidates derived context, and prevents old transcripts/checkpoints from re-importing it. Define transcript retention separately; deleting a memory alone does not erase the original conversation.

**Sessions and clients.** Personal memory remains current across resume, rewind, and fork; branch-specific task notes stay in session state. Recheck memory revisions before continuing restored sessions. Mimo currently saves no durable device transcript and its gateway uses a shared bearer token: bind an authenticated owner before enabling personal memory. Add a memory sheet to Mimo, terminal commands for standalone CLI, and equivalent controls in a new browser voice UI.

**Acceptance.** Remember a preference, reconnect, resume, and fork: it remains available. Correct or forget it, then restore an older checkpoint: stale versions do not return. Verify branch notes remain separate, project filtering works, and another user cannot read or modify the record.
