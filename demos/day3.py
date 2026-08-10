"""demos.day3

Demo/Test script to run the context compaction process.
It constructs a long mock conversation history, sets a low token budget,
and calls `compact` to show how the context engine compresses old messages
while keeping recent ones verbatim.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra import provider
from astra.context import compact, estimate_tokens


def main():
    # 1. Create a simulated long conversation history (11 messages)
    # The history contains a user request, followed by multiple tool calls,
    # results, and assistant replies.
    messages = [
        {"role": "user", "text": "Analyze the log files and find the database error."},
        {"role": "assistant", "text": "I will list the files in the directory first.", "tool_calls": [{"name": "list_files", "args": {}, "signature": "t1"}]},
        {"role": "tool", "name": "list_files", "text": "app.log\nmain.py\nconfig.json"},
        {"role": "assistant", "text": "Now I will read the app.log file.", "tool_calls": [{"name": "read_file", "args": {"path": "app.log"}, "signature": "t2"}]},
        {"role": "tool", "name": "read_file", "text": "INFO: system start\nERROR: database connection timed out after 30s\nINFO: retrying..."},
        {"role": "assistant", "text": "I found the error: database connection timed out. I should inspect config.json.", "tool_calls": [{"name": "read_file", "args": {"path": "config.json"}, "signature": "t3"}]},
        {"role": "tool", "name": "read_file", "text": '{"db_host": "localhost", "db_port": 5432, "timeout": 30}'},
        {"role": "assistant", "text": "The configuration timeout is 30 seconds. I will explain this to the user."},
        {"role": "user", "text": "Can you double check if there are other error logs?"},
        {"role": "assistant", "text": "Let me search for 'ERROR' in app.log.", "tool_calls": [{"name": "grep", "args": {"regex": "ERROR"}, "signature": "t4"}]},
        {"role": "tool", "name": "grep", "text": "ERROR: database connection timed out after 30s"},
    ]

    print("=== Original Conversation History ===")
    print(f"Total Messages: {len(messages)}")
    initial_tokens = estimate_tokens(messages)
    print(f"Estimated Tokens: {initial_tokens}")
    for i, msg in enumerate(messages):
        text = msg.get("text", "")
        # truncate for clean printing
        snippet = text[:60] + "..." if len(text) > 60 else text
        print(f"  [{i}] {msg['role']}: {snippet}")

    # Set budget_tokens lower than initial_tokens to trigger compaction
    budget_tokens = initial_tokens - 10
    print(f"\nSetting token budget to: {budget_tokens} (compaction will trigger)")

    # 2. Attempt compaction
    # To be resilient to API credit issues, we try to run with the real provider.
    # If the provider fails (e.g. credit/auth error), we run a fallback test with a mocked provider.
    print("\nRunning compaction...")
    try:
        compacted = compact(provider.DEFAULT_MODEL, messages, budget_tokens)
    except Exception as e:
        print(f"\n[Warning] Compaction via real Claude API failed: {e}")
        print("Falling back to simulated/mocked model completion for testing...")
        
        # Monkeypatch provider.complete to simulate Claude's summarization
        original_complete = provider.complete
        def mock_complete(model, system, messages, tools=None):
            return {
                "text": (
                    "The user asked to find database errors in the logs. "
                    "The assistant listed files, read app.log, found a database connection timeout error, "
                    "inspected config.json (which showed a 30s timeout), and is currently checking "
                    "for other errors."
                ),
                "tool_calls": [],
                "usage": {"input": 100, "output": 50}
            }
        provider.complete = mock_complete
        compacted = compact(provider.DEFAULT_MODEL, messages, budget_tokens)
        provider.complete = original_complete  # restore

    # 3. Print the compacted result
    print("\n=== Compacted Conversation History ===")
    print(f"Total Messages: {len(compacted)}")
    print(f"Estimated Tokens: {estimate_tokens(compacted)}")
    for i, msg in enumerate(compacted):
        text = msg.get("text", "")
        snippet = text[:150] + "..." if len(text) > 150 else text
        print(f"  [{i}] {msg['role']}:\n      {snippet}")


if __name__ == "__main__":
    main()
