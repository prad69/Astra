"""astra.harness

Concept: the composition. Every module built here does one job and
knows nothing about the others; Harness is the single place that wires
them into an agent -- tools from tools.py behind the gate from
security.py, a prompt from memory.py carrying the catalog from skills.py,
the cycle from loop.py compacted by context.py and recorded by
session.py, with subagent.py handing the whole arrangement a way to build
more of itself. Nothing new happens here. It is assembly.

Design rules this file embodies:
  - Children are ephemeral. A sub-agent gets the same workdir and the same
    policy but persist=False, because --resume picks the newest session
    file and a child that logged its own would win that race and resume
    the wrong conversation. The parent's log is the run; the child's
    contribution to it is the one string it returns.
  - The session is written as the run happens, not at the end. A crash is
    exactly the case durability exists for, so there is no "flush on exit"
    -- every message lands on disk as it is created, and a process killed
    between two messages loses nothing before the last one.
  - Recording tracks an index, not a diff. The loop hands the same list
    object throughout (loop.py slice-assigns), so "what is new" is
    everything past _recorded -- except after a compaction, when the list
    got shorter and the index has to be clamped back inside it. That
    clamp is why the session log keeps the full history while the model
    sees the compacted one: the log is the truth, the context is the
    working set.
"""

import os

from . import context, loop, memory, provider, session, skills
from .security import Policy
from .subagent import subagent_tool
from .tools import core_tools, tool

SESSION_LABEL_CHARS = 32


class Harness:
    """A composed agent: tools, policy, prompt, memory, skills, and a session."""

    def __init__(self, workdir=".", model=None, policy=None, extra_tools=None,
                 system_extra="", on_event=None, budget_tokens=600_000,
                 max_turns=120, session_path=None, enable_subagents=True,
                 persist=True, _depth=0):
        self.workdir = os.path.realpath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

        self.model = model or os.environ.get("ASTRA_MODEL") or provider.DEFAULT_MODEL
        self.policy = policy or Policy("yolo")
        self.on_event = on_event or (lambda kind, payload: None)
        self.budget_tokens = budget_tokens
        self.max_turns = max_turns
        self.session_path = session_path
        self.persist = persist
        self.depth = _depth

        self.messages = []
        self._recorded = 0

        self.tools = self._build_tools(extra_tools, enable_subagents)
        self.system = memory.build_system_prompt(
            self.workdir, self._prompt_extra(system_extra))

    def _build_tools(self, extra_tools, enable_subagents):
        """Assemble the tool set: filesystem, memory, skills, delegation, extras."""
        tools = {t.name: t for t in core_tools(self.workdir)}

        @tool("Save a durable fact about this project to ASTRA.md, so future "
              "runs know it without being told again",
              note="the fact to remember, in one line")
        def remember(note):
            return memory.remember(self.workdir, note)

        tools[remember.name] = remember

        # Only advertised when there is something to load -- an agent offered a
        # loader for an empty catalog will eventually call it to find out.
        if skills.catalog(self.workdir):
            @tool("Load the full instructions of a named skill before doing work "
                  "it applies to", name="the skill name from the catalog")
            def use_skill(name):
                return skills.read_skill(self.workdir, name)

            tools[use_skill.name] = use_skill

        if enable_subagents:
            tools["spawn_agent"] = subagent_tool(self._make_child, depth=self.depth)

        if extra_tools:
            tools.update({t.name: t for t in extra_tools})
        return tools

    def _make_child(self, child_depth):
        """Build a sub-agent: same directory and policy, own context, no session."""
        return Harness(
            workdir=self.workdir,
            model=self.model,
            policy=self.policy,
            on_event=self.on_event,
            budget_tokens=self.budget_tokens,
            max_turns=self.max_turns,
            enable_subagents=True,
            persist=False,
            _depth=child_depth,
        )

    def _prompt_extra(self, system_extra):
        """Join the skills catalog and any caller additions, skipping the empty."""
        parts = [p for p in (skills.catalog_prompt(self.workdir), system_extra) if p]
        return "\n\n".join(parts)

    def resume(self, path=None):
        """Adopt a previous session's messages. True when there were any.

        session.load repairs the transcript on the way in, so a run killed
        mid-tool comes back with an explicit interrupted result rather than
        an assistant turn the API would refuse to accept.
        """
        path = path or session.latest(self.workdir)
        if path is None or not os.path.exists(path):
            return False

        self.messages = session.load(path)
        self.session_path = path
        self._recorded = len(self.messages)
        return bool(self.messages)

    def _record(self):
        """Append every message not yet written to the session file."""
        if not self.persist or self.session_path is None:
            return

        # Compaction replaces the list contents with a shorter summary plus the
        # recent slice; without this clamp the index would point past the end
        # and every message after a compaction would go unrecorded.
        if self._recorded > len(self.messages):
            self._recorded = len(self.messages)

        for message in self.messages[self._recorded:]:
            session.append(self.session_path, message)
        self._recorded = len(self.messages)

    def flush(self):
        """Write any messages the session file does not have yet.

        run() records as it goes, so this exists for the callers that end
        a run from outside the loop -- an interrupt, a stop button -- and
        then repair the transcript. Without it those repairs would live
        only in memory and the session on disk would stay unreplayable.
        """
        self._record()

    def run(self, task):
        """Run one task to completion and return the agent's final answer."""
        if self.persist and self.session_path is None:
            self.session_path = session.new_session(
                self.workdir, task[:SESSION_LABEL_CHARS])

        self.messages.append({"role": "user", "text": task})
        self._record()

        def on_event(kind, payload):
            # Every event follows a message landing in the list, so recording
            # here writes each one through at the moment it exists.
            self._record()
            self.on_event(kind, payload)

        answer = loop.run_loop(
            model=self.model,
            system=self.system,
            messages=self.messages,
            tools=self.tools,
            on_event=on_event,
            before_tool=self.policy.check,
            max_turns=self.max_turns,
            before_turn=lambda msgs: context.compact(
                self.model, msgs, self.budget_tokens),
        )
        self._record()
        return answer
