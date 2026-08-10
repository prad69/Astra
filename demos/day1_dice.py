"""demos.day1_dice

Concept: the smallest possible end-to-end run of the loop -- one
hand-written tool, a task that should trigger exactly one call to it, and
enough printing to see the whole transcript (user prompt, assistant tool
call, tool result, assistant answer) go by. Everything here is throwaway
demo wiring; the reusable pieces (provider, loop) live in astra/.

Design rule this file embodies: before_tool here always allows, because
the loop has no policy yet -- security.py is what gives this
parameter teeth.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra import provider
from astra.loop import run_loop


class RollDice:
    """A hand-written tool: roll `count` six-sided dice and report the results."""

    spec = {
        "schema": {
            "name": "roll_dice",
            "description": "Roll count six-sided dice",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "string", "description": "How many dice"},
                },
                "required": ["count"],
            },
        }
    }

    def run(self, count):
        """Roll `count` dice and return the individual results and their total."""
        rolls = [random.randint(1, 6) for _ in range(int(count))]
        return f"rolls={rolls} total={sum(rolls)}"


def on_event(kind, payload):
    """Print each loop event so the transcript is visible as it happens."""
    if kind == "assistant":
        if payload["tool_calls"]:
            for call in payload["tool_calls"]:
                print(f"[assistant] calls {call['name']}({call['args']})")
        if payload["text"]:
            print(f"[assistant] {payload['text']}")
    elif kind == "tool_start":
        print(f"[tool_start] {payload['name']}({payload['args']})")
    elif kind == "tool_end":
        print(f"[tool_result] {payload['text']}")


def before_tool(call):
    """Default to allow every call."""
    return None


def main():
    """Run the dice task through the loop and print the final answer."""
    task = "Roll 3 dice and tell me whether the total beats 10"
    print(f"[user] {task}")
    messages = [{"role": "user", "text": task}]
    tools = {"roll_dice": RollDice()}
    answer = run_loop(
        model=provider.DEFAULT_MODEL,
        system="You are a concise assistant with access to a dice-rolling tool.",
        messages=messages,
        tools=tools,
        on_event=on_event,
        before_tool=before_tool,
    )
    print(f"[final] {answer}")


if __name__ == "__main__":
    main()
