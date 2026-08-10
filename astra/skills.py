"""astra.skills

Concept: teaching the agent something new without touching its code. A
skill is a directory holding a SKILL.md -- prose the agent reads when a
task calls for it. The catalog goes in the system prompt so the agent
knows what exists; the body is loaded only on demand, because the point of
a catalog is that ten skills cost ten lines of context rather than ten
documents.

Design rules this file embodies:
  - Progressive disclosure. catalog_prompt advertises names and one-line
    descriptions; read_skill fetches the full text. Inlining every skill
    would work at three skills and quietly ruin the context budget at
    thirty.
  - The front matter parse stops at the first "description:" and never
    imports a YAML library. A skill whose front matter is malformed still
    appears in the catalog with an empty description rather than breaking
    the listing -- the file is documentation, and documentation with a
    typo in it is still worth offering.
  - A missed lookup returns the available names. The model picked a wrong
    name from a list it can no longer see, so the error carries the list
    back; without it the only recovery is to guess again.
"""

import os

SKILLS_DIR = "skills"


def _description(path):
    """Read the front matter "description:" line, or "" when there is none.

    Front matter is the leading "---" fenced block. Scanning it line by
    line -- rather than parsing the document -- keeps this to a few lines
    and means a skill body containing the word "description:" cannot be
    mistaken for metadata.
    """
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.read().split("\n")
    except OSError:
        return ""

    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.lower().startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def catalog(workdir):
    """Map skill name -> {"description", "path"} for every skills/*/SKILL.md."""
    root = os.path.join(os.path.realpath(workdir), SKILLS_DIR)
    if not os.path.isdir(root):
        return {}

    found = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "SKILL.md")
        if os.path.isfile(path):
            found[name] = {"description": _description(path), "path": path}
    return found


def catalog_prompt(workdir):
    """Render the catalog as a prompt section, or "" when there are no skills."""
    found = catalog(workdir)
    if not found:
        return ""

    lines = ["Skills available (load one with the use_skill tool when relevant):"]
    lines += [f"- {name}: {meta['description']}" for name, meta in found.items()]
    return "\n".join(lines)


def read_skill(workdir, name):
    """Return the full SKILL.md text, or an ERROR naming what is available."""
    found = catalog(workdir)
    if name not in found:
        available = ", ".join(found) or "(none)"
        return f"ERROR: no skill named {name}. Available: {available}"

    with open(found[name]["path"], "r", errors="replace") as f:
        return f.read()
