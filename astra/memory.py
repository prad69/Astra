"""astra.memory

Concept: durable memory, and the prompt that carries it. Compaction (see
context.py) keeps a single run affordable but forgets everything the
moment the process exits; ASTRA.md is the other half -- a plain file in
the working directory that the agent writes to and that every later run
reads back through its system prompt. Memory that survives a restart has
to live on disk, and the cheapest disk format a coding agent already knows
how to edit is Markdown.

Design rules this file embodies:
  - The system prompt is assembled, not stored. Nothing here caches a
    string: build_system_prompt reads ASTRA.md on every call, so a note
    written during a run is visible to the next run without anyone
    invalidating anything.
  - remember appends and never rewrites. An agent that can restructure its
    own memory file can also erase it by mistake half way through a run;
    append-only means the worst case is a cluttered file, not a lost one.
    Curating it is a job for edit_file, deliberately, like any other file.
  - The base prompt is behavioural, not encyclopaedic. It says how to act
    -- inspect first, prefer small edits, verify, stop when done -- and
    leaves what to do to the task. Facts about this project belong in
    ASTRA.md, which is the whole point of splitting them.
"""

import os
import platform

MEMORY_FILE = "ASTRA.md"

BASE_PROMPT = """You are Astra, a small sharp coding agent. You work inside one \
directory using the tools provided.

Act, don't narrate: make the tool call rather than describing the tool call you \
would make.
Inspect before assuming: read the file, list the directory, run the command -- do \
not guess at what is there.
Prefer edit_file for small changes; write_file replaces a whole file and loses \
anything you did not repeat.
Verify after building: run the code or re-read the file you wrote, and say what \
you observed.
Never repeat a failing call unchanged. Read the error, change something, then try \
again.
When the task is complete, reply with a short summary and stop calling tools."""


def build_system_prompt(workdir, extra=""):
    """Assemble the system prompt: base rules, environment, memory, extra.

    Sections are joined by blank lines and every one after the base is
    conditional, so a fresh directory with no ASTRA.md and no caller
    additions yields exactly the base prompt plus its environment line.
    """
    root = os.path.realpath(workdir)
    sections = [
        BASE_PROMPT,
        f"Platform: {platform.system()}. Working directory: {root}",
    ]

    memory_path = os.path.join(root, MEMORY_FILE)
    if os.path.exists(memory_path):
        with open(memory_path, "r", errors="replace") as f:
            sections.append(f"Project memory ({MEMORY_FILE}):\n{f.read().strip()}")

    if extra:
        sections.append(extra)
    return "\n\n".join(sections)


def remember(workdir, note):
    """Append one bullet to ASTRA.md and report it back to the model."""
    path = os.path.join(os.path.realpath(workdir), MEMORY_FILE)
    with open(path, "a") as f:
        f.write(f"- {note}\n")
    return f"Remembered in {MEMORY_FILE}"
