"""demos.day2_build

Concept: a loop with real tools and a policy gate in
front of them. core_tools sandboxes every path to a scratch directory;
Policy("yolo") allows everything except the always-denied bash patterns,
so this demo exercises the tool surface without also exercising approval
prompts -- a stricter mode can also be configured.

The default task now points fetch_url at TARGET_URL and asks for a
summary, which splits the work the way the harness intends: the tool
retrieves text, the model is what summarizes it.

Design rule this file embodies: the task is a command-line argument with a
sensible default, so the same script drives all verification
scenarios (build-and-run, a destructive request, a path escape, a fetch)
without editing code between runs.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra import provider
from astra.loop import run_loop
from astra.security import Policy
from astra.tools import core_tools

TARGET_URL = "https://www.geekslopout.com"

DEFAULT_TASK = (
    f"Fetch {TARGET_URL} with the fetch_url tool and summarize what the page "
    "says. If the tool returns an error, report the error text verbatim "
    "instead of guessing at the content."
)


def on_event(kind, payload):
    """Print each loop event so the transcript is visible as it happens."""
    if kind == "assistant":
        for call in payload["tool_calls"]:
            print(f"[assistant] calls {call['name']}({call['args']})")
        if payload["text"]:
            print(f"[assistant] {payload['text']}")
    elif kind == "tool_start":
        print(f"[tool_start] {payload['name']}({payload['args']})")
    elif kind == "tool_end":
        print(f"[tool_result] {payload['text']}")


def create_onboarding_document(repo_path):
    """Run the agent loop to create an onboarding document for the specified repository."""
    print(f"\n--- Creating Onboarding Document for {repo_path} ---")
    tools = {t.name: t for t in core_tools(repo_path)}
    task = (
        "Inspect the repository structure, key files, and architecture of the codebase in the current directory. "
        "Create a comprehensive onboarding document (in Markdown) that explains the codebase structure, how "
        "to run the main entry points, the overall architecture, and key features. "
        "Write this document directly to a file named ONBOARDING.md in the root of the repository. "
        "Verify that the file has been written correctly."
    )
    print(f"[user] {task}")
    policy = Policy("yolo")
    messages = [{"role": "user", "text": task}]
    answer = run_loop(
        model=provider.DEFAULT_MODEL,
        system="You are a senior software architect creating repository onboarding documentation. "
               "Analyze the repository structure, read key files, and write a thorough, helpful ONBOARDING.md file.",
        messages=messages,
        tools=tools,
        on_event=on_event,
        before_tool=policy.check,
    )
    print(f"[final onboarding answer] {answer}")


def main():
    """Run a task through the loop with real tools and a yolo-mode policy."""
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    scratch = tempfile.mkdtemp(prefix="astra_")
    print(f"[scratch] {scratch}")

    tools = {t.name: t for t in core_tools(scratch)}

    # Call fetch_url directly first, so the raw tool output is visible next
    # to what the model later makes of it -- a tool is an ordinary function
    # and does not need a loop around it to be exercised.
    print(f"[direct] fetch_url({TARGET_URL})")
    print(f"[direct result] {tools['fetch_url'].run(url=TARGET_URL)}")

    print(f"[user] {task}")
    policy = Policy("yolo")
    messages = [{"role": "user", "text": task}]
    answer = run_loop(
        model=provider.DEFAULT_MODEL,
        system="You are a careful coding agent with filesystem and shell tools. "
               "Verify your work (e.g. by running code) before answering.",
        messages=messages,
        tools=tools,
        on_event=on_event,
        before_tool=policy.check,
    )
    print(f"[final] {answer}")

    # Create onboarding document for EcommerceAgents repository
    create_onboarding_document("/Users/pradeep/Documents/work/Algopractice")


if __name__ == "__main__":
    main()
