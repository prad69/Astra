"""astra.subagent

Concept: delegation as a tool. A sub-agent is an ordinary Harness given
its own task and its own empty transcript; spawning one is how the parent
explores something expensive -- reading a large subsystem, trying an
approach that may not work -- without that exploration landing in its own
context. The child reports back one string. Everything it read to produce
that string is discarded with it.

Design rules this file embodies:
  - The harness is injected, not imported. subagent_tool takes a
    make_harness callable, so this module never imports Harness and
    harness.py is free to import this one. That keeps the composition in
    one direction and this file testable with a stub.
  - Depth is carried by the tool, not by the agent. Each child is built
    with depth+1 baked into its own spawn_agent, so recursion is bounded
    by construction rather than by asking a model to keep count.
  - The limit is a tool result, not an exception. An agent that hits the
    ceiling is told to do the work itself and carries on; raising here
    would end a run over what is a routine planning correction.
"""

from .tools import tool

DEPTH_LIMIT_ERROR = "ERROR: sub-agent depth limit reached; do this task yourself"


def subagent_tool(make_harness, depth=0, max_depth=2):
    """Build the spawn_agent Tool for a harness at this depth.

    make_harness(child_depth) must return an object with .run(task) -> str.
    """

    @tool("Delegate a self-contained task to a fresh sub-agent with its own "
          "clean context. The sub-agent cannot see this conversation, so the "
          "task must describe everything it needs. Returns the sub-agent's "
          "final report.",
          task="the complete, self-contained task for the sub-agent")
    def spawn_agent(task):
        if depth >= max_depth:
            return DEPTH_LIMIT_ERROR
        child = make_harness(depth + 1)
        return child.run(task)

    return spawn_agent
