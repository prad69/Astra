"""astra.loop

Concept: the turn cycle. An agent is just this: ask the model, and if it
asked for tools, run them and ask again. Later additions -- memory,
security, subagents -- plug into this loop through its existing seams
(on_event, before_tool, before_turn) rather than by changing the cycle.

Design rules: the loop never crashes because a tool did -- a bad call
becomes an "ERROR: ..." string fed back to the model, not a Python
exception that kills the run. before_tool is a gate, not a filter: it
returns a reason to block or None to allow, and a blocked call still gets
a normal tool-result message so the transcript stays coherent.

The default fills the before_turn seam: it now defaults to context.compact
rather than to doing nothing, so a long run stays inside a token budget
without every caller remembering to ask. Passing before_turn explicitly
still overrides that -- the seam did not change shape, only its default.
"""

from . import context, provider

# Roughly half of a 200k-token context window. Compaction is what keeps a
# run affordable, so the default leaves room for the reply and the tool
# specs rather than compacting only once the window is nearly full.
DEFAULT_BUDGET_TOKENS = 100_000


def run_loop(model, system, messages, tools, on_event, before_tool,
             max_turns=80, before_turn=None, budget_tokens=DEFAULT_BUDGET_TOKENS):
    """Run the model/tool cycle until the model stops calling tools.

    tools is a dict of name -> Tool, where a Tool has .spec (a {"schema":
    ...} dict) and .run(**kwargs). Returns the model's final text answer.

    before_turn defaults to compacting against budget_tokens; pass a
    callable to replace that policy, or `lambda msgs: msgs` to switch
    compaction off entirely.
    """
    if before_turn is None:
        print(f"Running compaction with budget {budget_tokens} tokens")
        before_turn = lambda msgs: context.compact(model, msgs, budget_tokens)

    for _ in range(max_turns):
        # Slice-assign rather than rebind: compact returns a new list, and a
        # plain `messages = ...` would leave the caller holding the stale
        # pre-compaction one while the loop worked from another.
        messages[:] = before_turn(messages)

        specs = [t.spec for t in tools.values()]
        reply = provider.complete(model, system, messages, specs)
        assistant_msg = {
            "role": "assistant", "text": reply["text"], "tool_calls": reply["tool_calls"]
        }
        messages.append(assistant_msg)
        on_event("assistant", assistant_msg)

        if not reply["tool_calls"]:
            # A reply cut off at the output limit looks exactly like a finished
            # one -- no tool calls, often no text -- so without this check the
            # loop returns truncated work as the final answer and the caller
            # records a success that produced nothing.
            if reply.get("stop_reason") == "max_tokens":
                messages.append({"role": "user", "text":
                                 "Your last reply hit the output limit and was cut off. "
                                 "Continue from where you stopped. Write large files in "
                                 "several smaller pieces rather than one call."})
                continue
            return reply["text"]

        for call in reply["tool_calls"]:
            on_event("tool_start", call)
            result = _run_one(call, tools, before_tool)
            tool_msg = {"role": "tool", "name": call["name"], "text": str(result)}
            messages.append(tool_msg)
            on_event("tool_end", tool_msg)

    messages.append({"role": "user", "text": "Turn limit reached; wrap up now."})
    reply = provider.complete(model, system, messages, [])
    messages.append({"role": "assistant", "text": reply["text"], "tool_calls": []})
    return reply["text"]


def _run_one(call, tools, before_tool):
    """Dispatch a single tool call, converting blocks/errors into result text."""
    reason = before_tool(call)
    if reason is not None:
        return f"BLOCKED: {reason}"

    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"

    try:
        return tool.run(**call["args"])
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
