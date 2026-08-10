"""astra.security

Concept: the policy gate that sits at before_tool. Every call the loop is
about to run passes through Policy.check first; a reason string blocks it,
None lets it through. This file has no idea what a tool does -- it only
knows names and bash command text -- which keeps it small enough to audit
by eye, the property a security gate most needs.

Design rules this file embodies:
  - Deny patterns are checked before mode, so a matching bash command is
    blocked even under "yolo" -- there is no mode that bypasses them,
    because a mode is a policy choice and these are treated as bugs, not
    policy.
  - The default approver refuses. A caller who wants "safe" mode to ever
    approve anything must supply a real approver; wiring nothing in is
    fail-closed by construction, not by accident.
"""

import re

READ_TOOLS = {"read_file", "list_files", "grep"}

# Each pattern targets one class of irreversible or scope-breaking command:
# recursive force-delete of a root-ish path, privilege escalation, disk
# formatting/raw writes, piping a remote script into a shell, force-pushing
# history, and writing straight onto a block device.
DENY_PATTERNS = [
    # \b fails after a bare "/" or "~" at end-of-string (neither side is a
    # word character, so there is no boundary) -- an explicit lookahead for
    # whitespace/end/slash/star is what actually catches "rm -rf /".
    r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+(/|~|\$HOME)(?=\s|$|/|\*)",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"curl\b[^|\n]*\|\s*(sudo\s+)?(sh|bash)\b",
    r"git\s+push\b[^\n]*(--force\b|-f\b)",
    r">\s*/dev/sd[a-z0-9]*\b",
]


class Policy:
    """Gate tool calls by mode: read-only, safe (ask), or yolo (allow)."""

    def __init__(self, mode="safe", approver=None):
        self.mode = mode
        self.approver = approver or (lambda call, reason: False)

    def check(self, call):
        """Return None to allow the call, or a reason string to block it."""
        if call["name"] == "bash":
            command = call.get("args", {}).get("command", "")
            for pattern in DENY_PATTERNS:
                if re.search(pattern, command):
                    return f"command matches denied pattern: {pattern}"

        if call["name"] in READ_TOOLS or self.mode == "yolo":
            return None

        if self.mode == "read-only":
            return f"read-only mode blocks {call['name']}"

        reason = f"approve {call['name']}({call.get('args', {})})?"
        if self.approver(call, reason):
            return None
        return f"not approved: {reason}"
