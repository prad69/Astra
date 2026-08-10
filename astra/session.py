"""astra.session

Concept: durability. A run that dies -- crash, Ctrl-C, closed laptop --
should be resumable rather than repeated, so every message is appended to
a JSONL file the moment it exists. JSONL because the interesting failure
is a process killed mid-write: one object per line means a torn file loses
its last line and nothing else, where a single JSON array would lose the
whole document.

Design rules this file embodies:
  - load stops at the first unparseable line instead of skipping it. A
    broken line in the middle would mean corruption we do not understand;
    a broken line at the end is the expected shape of a killed process.
    Treating them the same way -- stop, keep the good prefix -- means the
    recovery path never has to guess which one it is looking at.
  - Repair is not optional cleanup, it is what makes the transcript
    legal. Claude rejects any assistant turn whose tool_use blocks lack
    matching tool_result blocks, so a process that died between calling a
    tool and recording its result leaves a file that cannot be replayed.
    Synthesising "Interrupted before this ran" results restores the
    pairing rule and tells the model the truth about what happened, which
    is enough for it to decide whether to run the call again.
  - Sessions are named, not numbered: a timestamp for ordering and a slug
    for recognising the run later in a directory listing.
"""

import json
import os
import re
import time

SESSION_DIR = ".astra/sessions"

INTERRUPTED = "Interrupted before this ran (process restarted)."


def _slugify(label):
    """Reduce a label to alphanumerics and dashes, clipped to 40 characters."""
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-")
    return slug[:40]


def new_session(workdir, label="session"):
    """Create the session directory and return the path for a new transcript."""
    root = os.path.join(os.path.realpath(workdir), SESSION_DIR)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{int(time.time())}-{_slugify(label)}.jsonl")


def append(path, message):
    """Append one message to the session file as a single JSON line.

    ensure_ascii=False so a transcript containing non-English text or a
    stack trace full of box-drawing characters stays readable in an editor
    rather than turning into escape sequences.
    """
    with open(path, "a") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def repair(messages):
    """Give every unanswered tool call of the last assistant turn a result.

    Counts the tool messages that already follow the final assistant turn
    and synthesises one interrupted result per tool_call beyond that count,
    which is exactly the set of calls that never reported back.
    """
    last = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last = i
            break
    if last is None:
        return messages

    calls = messages[last].get("tool_calls") or []
    answered = sum(1 for m in messages[last + 1:] if m["role"] == "tool")
    for call in calls[answered:]:
        messages.append({"role": "tool", "name": call["name"], "text": INTERRUPTED})
    return messages


def load(path):
    """Read a session file, discard any torn tail, and repair the pairing."""
    messages = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            try:
                messages.append(json.loads(line))
            except ValueError:
                # A half-written final line: everything before it is intact.
                break
    return repair(messages)


def latest(workdir):
    """Return the most recently modified session file, or None if there are none."""
    root = os.path.join(os.path.realpath(workdir), SESSION_DIR)
    if not os.path.isdir(root):
        return None

    sessions = [os.path.join(root, n) for n in os.listdir(root) if n.endswith(".jsonl")]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)
