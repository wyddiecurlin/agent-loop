# Working-context compaction

- Replace older conversation/tool exchanges with a bounded continuation summary before the model runs out of context; retain current instructions, the active task, and recent complete exchanges.
- Implement through `ContextBudget` and one summary call using the existing model adapter. Check before every main-model request, including inside one long-running task.
- Save the summary and transcript position atomically through [SESSIONS.md](SESSIONS.md); resume without re-executing recorded tool actions.
- Share the behavior across CLI, Mimo, and browser voice. Keep durable user facts in [MEMORY.md](MEMORY.md), separate from this working transcript.

**Today.** [context.py](../agent_loop/context.py) accepts `compaction` but does nothing; `auto-clear` drops earlier user turns. [loop.py](../agent_loop/loop.py) already checks the budget before each model request, but ordinary [CLI](../agent_loop/__main__.py) does not supply a budget. The [voice bridge](../voice/bridge.py) also checks before its preamble. Preserving the entire current run would still overflow during a 24-hour task.

**Smallest implementation.** Add an injectable summarizer to the budget path; call the existing adapter without tools and with bounded input/output. Preserve the original active request separately. Summarize the previous summary plus older completed exchanges, retaining goals, constraints, decisions, completed effects, artifact/job/immutable-attachment references, uncertainties, and next steps. Keep fresh system instructions and recent complete tool-call/result groups verbatim. Never split a group or summarize pending calls. Treat summaries as historical data, not new instructions.

**Images.** Image bytes are no longer part of conversation history after compaction,
including otherwise retained recent exchanges. The summary may describe image content
using observations already made by the model, or a bounded vision summary before the
payload is released. Preserve relevant findings, uncertainty, source references, and
PDF page numbers as text. Remove raw bytes, base64/data URLs, and image-bearing content
parts from the replacement history, including payloads embedded in tool arguments or
results; preserve call IDs and result pairing. Do not copy remote image URLs back as
image parts or automatically reload attachments on resume. A reference may support a
later explicit `image_view` call, but is not itself retained visual context. Validate
that compacted history and its persisted form contain no image payloads.

Use the existing conservative estimate and provider limits, including schemas, attachment costs, and output reserve. Trigger early enough for the summary request itself to fit; chunk oversized older history. Reset the usage anchor after replacement and verify the result fits. Bound retries; on failure retain original context and report a recoverable error rather than clearing it silently. Commit validated context with its covered transcript position and generation; cancellation or restart must expose either the old state or the complete replacement. Reuse session operation records to prevent replay.

**Clients and acceptance.** Mimo's [configuration](../../pet-moment/mobile-app/Mimo/Agent/AgentClient.swift) already accepts both modes but defaults to auto-clear. Wire CLI budgets. Distinguish compaction from clearing: the bridge currently discards history when preparation returns true. Preserve the replacement, use stable turn references for preamble insertion, and add an allowlisted `context_compacted` event through Mimo’s gateway. Enable compaction after validation; a new browser voice UI shares these events. Extend [context tests](../tests/test_context.py) for repeated compaction inside one task, preserved call pairs/task constraints, oversized outputs, summary failure/cancellation, resume, completed writes never replayed, and image content summarized without retaining or restoring image bytes.

**Requested reference.** The [pinned Codex compact templates](https://github.com/openai/codex/tree/cfd5d77d63f53303aad63ef0886d3b159996a1af/codex-rs/prompts/templates/compact) remain unverified: GitHub fetches failed during planning. Verify their exact handoff prompt/prefix before adapting wording; this plan does not require importing the Codex runtime or changing model APIs.
