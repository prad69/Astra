"""astra.provider

Concept: the provider boundary. Everything above this module (loop, tools,
harness) speaks a small neutral message format -- plain dicts with "role"
and "text" -- and never touches the Messages API's wire shapes directly.
This is the only file that knows what the API looks like, so swapping
providers later means rewriting this file and nothing else.

Design rules: stdlib-only HTTP via urllib; transient errors (429, 500, 502,
503, dropped connections) retry with exponential backoff, everything else
is a hard failure. Claude's Messages API pairs each tool_use block with a
tool_result block by id, not by name or position -- send the wrong id (or
none) and the call is rejected outright. So the neutral message format's
"signature" field carries that id: _to_wire stamps it onto the replayed
tool_use block and threads the same id into the matching tool_result,
which is the round-trip this provider's calls depend on.
"""

import json
import os
import time
import urllib.error
import urllib.request

API_ROOT = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5"

# A single-file landing page -- design system, nine sections, inline SVGs -- is
# tens of thousands of output tokens in one write_file call. At 8192 the reply
# was cut off inside the tool_use block, so the call arrived missing its
# "content" argument and the write silently produced nothing.
MAX_OUTPUT_TOKENS = 32_000


def api_key():
    """Return the configured API key, preferring ASTRA_API_KEY over ANTHROPIC_API_KEY."""
    key = os.environ.get("ASTRA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key set. Export ASTRA_API_KEY or ANTHROPIC_API_KEY before running."
        )
    return key


def _to_wire(messages):
    """Translate neutral messages into Claude `messages` entries.

    user -> role "user" text block. assistant -> role "assistant" text block
    (if any) plus one tool_use block per tool call, id'd by its stored
    signature. tool -> role "user" tool_result block, tool_use_id matched
    positionally to the ids the preceding assistant turn handed out; runs of
    tool results for one turn are merged into a single user message, which
    is what the API expects rather than one message per result.
    """
    wire = []
    pending_ids = []
    for m in messages:
        role = m["role"]
        if role == "user":
            wire.append({"role": "user", "content": [{"type": "text", "text": m["text"]}]})
        elif role == "assistant":
            content = []
            if m.get("text"):
                content.append({"type": "text", "text": m["text"]})
            pending_ids = []
            for call in m.get("tool_calls", []):
                call_id = call.get("signature") or f"toolu_{len(pending_ids)}"
                content.append({"type": "tool_use", "id": call_id,
                                 "name": call["name"], "input": call["args"]})
                pending_ids.append(call_id)
            wire.append({"role": "assistant", "content": content})
        elif role == "tool":
            block = {"type": "tool_result",
                     "tool_use_id": pending_ids.pop(0) if pending_ids else None,
                     "content": m["text"]}
            if wire and wire[-1]["role"] == "user" and wire[-1]["content"][0]["type"] == "tool_result":
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
    return wire


def complete(model, system, messages, tools=None):
    """Call the Messages API once and return {"text", "tool_calls", "usage", "stop_reason"}."""
    body = {
        "model": model,
        "system": system,
        "messages": _to_wire(messages),
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    if tools:
        body["tools"] = [
            {"name": t["schema"]["name"], "description": t["schema"]["description"],
             "input_schema": t["schema"]["parameters"]}
            for t in tools
        ]

    headers = {
        "content-type": "application/json",
        "x-api-key": api_key(),
        "anthropic-version": "2023-06-01",
    }
    data = _post(API_ROOT, body, headers)

    text = "".join(b["text"] for b in data.get("content", []) if b["type"] == "text")
    tool_calls = [
        {"name": b["name"], "args": b["input"], "signature": b["id"]}
        for b in data.get("content", []) if b["type"] == "tool_use"
    ]
    usage_meta = data.get("usage", {})
    usage = {
        "input": usage_meta.get("input_tokens", 0),
        "output": usage_meta.get("output_tokens", 0),
    }
    # stop_reason travels with the reply because "max_tokens" makes a truncated
    # turn otherwise indistinguishable from a finished one -- same empty text,
    # same absent tool calls. The loop needs it to tell those apart.
    return {"text": text, "tool_calls": tool_calls, "usage": usage,
            "stop_reason": data.get("stop_reason")}


def _post(url, body, headers, retries=5):
    """POST JSON with retry-on-transient-failure, backing off 2**attempt * 2 seconds."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"Claude API error {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise
