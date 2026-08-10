"""astra.context

Concept: the context engine. A loop that appends every message forever
eventually sends a transcript larger than the model can read, and long
before that it is paying to re-send older file dumps on every turn.
Compaction trades the exact transcript for a summary of it: the old
messages become one paragraph, the recent ones are kept verbatim, and the
run continues in a fraction of the tokens. This plugs into the before_turn
seam loop.py has carried unused -- the loop does not change.

Design rules this file embodies:
  - Compaction is lossy, so the boundary is drawn by recency, not by
    importance: the last KEEP_RECENT messages survive untouched because
    that is the work in progress, and the model is asked to carry the
    durable facts (task, files touched, decisions, open errors) out of
    everything older. Summarize the recent slice too and the agent forgets
    what it was doing mid-edit.
  - The kept slice may not start with a tool result. Claude pairs each
    tool_result with the tool_use that produced it, and provider._to_wire
    only knows ids handed out by an assistant message it has already seen
    -- a slice opening on an orphaned tool message sends tool_use_id null
    and the API rejects the whole request. Dropping those leading messages
    is what makes the seam safe to call on any turn boundary.
  - estimate_tokens counts characters and divides. It is wrong by a good
    margin on any particular message and close enough in aggregate to
    decide "is this transcript getting long", which is the only question
    asked of it. A real tokenizer here would cost an import and a network
    round trip to buy precision nothing downstream uses.
  - Compaction itself costs a model call, so it happens only when over
    budget. Under budget the messages come back untouched and the caller
    cannot tell the seam is wired in at all.
"""

from . import provider

CHARS_PER_TOKEN = 4
KEEP_RECENT = 6

# Per-message clip for the rendered transcript. The summarizer needs to see
# what a message was about, not all of it -- an un-clipped render would feed
# the whole oversized transcript back in, which is the problem, not the fix.
CLIP_CHARS = 800

COMPACT_SYSTEM = (
    "You compress agent transcripts. Preserve: the original task, every file "
    "created or edited and its purpose, key decisions, unresolved errors, and "
    "what remains to be done. Be dense and factual."
)


def estimate_tokens(messages):
    """Roughly how many tokens this message list occupies."""
    return sum(len(str(m)) for m in messages) // CHARS_PER_TOKEN


def _render(messages):
    """Flatten messages into a plain transcript for the summarizer.

    Each line names its role, the tool it came from when there is one, and
    the tools any assistant turn asked for -- a summary that loses "it
    already ran the tests" invites the agent to run them again.
    """
    lines = []
    for m in messages:
        label = m["role"]
        if m.get("name"):
            label = f"{label}({m['name']})"

        text = (m.get("text") or "").strip()
        if len(text) > CLIP_CHARS:
            text = text[:CLIP_CHARS] + " ...[clipped]"

        calls = m.get("tool_calls") or []
        if calls:
            named = ", ".join(c["name"] for c in calls)
            text = f"{text}\n[called: {named}]" if text else f"[called: {named}]"

        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _drop_leading_tool_results(messages):
    """Return messages with any leading tool results removed.

    A tool result is only meaningful next to the tool_use it answers. When
    the split lands mid-turn those results are orphans, and the fix is to
    drop them rather than to move the boundary -- their content is already
    represented in the summary.
    """
    start = 0
    while start < len(messages) and messages[start]["role"] == "tool":
        start += 1
    return messages[start:]


def compact(model, messages, budget_tokens):
    """Summarize all but the last KEEP_RECENT messages when over budget.

    Returns a new list -- a single user message holding the summary,
    followed by the recent slice -- or the original list untouched when it
    is small enough or too short to be worth splitting.
    """
    if estimate_tokens(messages) <= budget_tokens or len(messages) <= KEEP_RECENT + 1:
        return messages

    old = messages[:-KEEP_RECENT]
    recent = _drop_leading_tool_results(messages[-KEEP_RECENT:])

    reply = provider.complete(
        model,
        COMPACT_SYSTEM,
        [{"role": "user", "text": _render(old)}],
        [],
    )
    summary = {"role": "user", "text": f"[Conversation so far, compacted]\n{reply['text']}"}
    print("----->[summary] ", reply['text'])
    return [summary] + recent
