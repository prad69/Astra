"""astra.cli

Concept: the front door. Everything built is reachable from
Python already; this file is what makes it reachable from a terminal, in
the two shapes people actually use -- one task and exit, or a session you
sit in. It owns argument parsing, the printing, and the approval prompt,
and it owns nothing else: no tool, no policy decision, no message handling
lives here.

Design rules this file embodies:
  - The default mode depends on how you invoked it. A person watching the
    output can answer "approve this?", so interactive defaults to safe. A
    script piping -p into a build cannot, and a prompt nobody answers is a
    hang, so headless defaults to yolo. The mode is still one flag away in
    both directions.
  - Ctrl-C interrupts the run, not the process. The session file is
    already on disk (harness records as it goes), so the useful thing to
    do with an interrupt is say so and return to the prompt, leaving
    --resume to pick the work back up.
  - Output is clipped, not summarized. A tool call shows its name and
    shortened arguments, a result shows its first line; anyone wanting the
    whole thing has the session file. A terminal that scrolls a 4000-line
    read_file result past you is not showing you anything.
"""

import argparse
import os
import sys
from pathlib import Path

# Enable running this file directly as a script
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "astra"

from .harness import Harness
from .security import Policy

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

ARG_CLIP = 70
RESULT_CLIP = 160

# The wordmark, cut one letter at a time so the serifs stay under control.
# Five rows of light rule: letters chiselled into stone rather than lit on
# an arcade cabinet. Each glyph is a fixed 8 columns, so the row strings
# are just the glyphs run together.
_SERIF = {
    "A": ["   ╱╲   ", "  ╱  ╲  ", " ╱────╲ ", " │    │ ", " ┴    ┴ "],
    "S": ["╭─────┐ ", "│       ", "╰─────╮ ", "      │ ", "└─────╯ "],
    "T": ["┌──┬───┐", "   │    ", "   │    ", "   │    ", "  ╶┴╴   "],
    "R": [" ┌────╮ ", " │    │ ", " ├────╯ ", " │  ╲   ", " ┴   ╲  "],
}

_SERIF_ASCII = {
    "A": ["   /\\   ", "  /  \\  ", " /----\\ ", " |    | ", " '    ' "],
    "S": [",-----. ", "|       ", "'-----. ", "      | ", "'-----' "],
    "T": [".--,---.", "   |    ", "   |    ", "   |    ", "  -'-   "],
    "R": [" ,----. ", " |    | ", " |----' ", " |  \\   ", " '   \\  "],
}


def _cut(glyphs):
    """Set the five rows of ASTRA from a glyph table."""
    return ["".join(glyphs[letter][row] for letter in "ASTRA") for row in range(5)]


WORDMARK = _cut(_SERIF)

# ASCII stand-in for terminals that cannot encode the rule glyphs.
WORDMARK_ASCII = _cut(_SERIF_ASCII)

# 256-color ramp: lit from above, pale gold down into bronze.
WORDMARK_COLORS = [230, 222, 221, 178, 172]

# An astra is a divine weapon, loosed from a drawn bow with a mantra. The
# emblem is that instant: limbs bent, string hauled back to a point, the
# arrow nocked and waiting on the word. Column layout, left to right --
# fletching at 5, nock at 6, the string converging from the limb tips,
# the grip at 17, the head just past it.
EMBLEM = [
    "                      ╲",
    "                  ╱╱╱╱ ╲╲",
    "              ╱╱╱╱       ╲",
    "          ╱╱╱╱            ┃",
    "       ≪──────────────────╂───▶",
    "          ╲╲╲╲            ┃",
    "              ╲╲╲╲       ╱",
    "                  ╲╲╲╲ ╱╱",
    "                      ╱",
]

EMBLEM_ASCII = [
    "                      \\",
    "                  //// \\\\",
    "              ////       \\",
    "          ////            |",
    "       <------------------+--->",
    "          \\\\\\\\            |",
    "              \\\\\\\\       /",
    "                  \\\\\\\\ //",
    "                      /",
]

# Row shades for the emblem: dark bronze at the limb tips, glowing toward
# the nocked arrow at the center.
EMBLEM_COLORS = [94, 130, 137, 172, 179, 172, 137, 130, 94]

# Index of the arrow row, and the ramp its shaft walks nock-to-tip: the
# head is the hottest thing on the screen, which is where a drawn bow
# puts your eye too.
SHAFT_ROW = 4
SHAFT_COLORS = [137, 172, 214, 220, 229]

TAGLINE = "a small agent harness"

BANNER_INFO = """  {dim}model{reset} {model}
  {dim}mode {reset} {mode}
  {dim}jail {reset} {workdir}
{dim}Ctrl-D to exit, Ctrl-C to interrupt a run.{reset}"""


def _unicode_ok(stream=None):
    """True when the output encoding can carry the block-glyph wordmark."""
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or ""
    if "utf" not in encoding.lower():
        return False
    try:
        "".join(WORDMARK + EMBLEM).encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _gradient(text, colors):
    """Color a line left-to-right by walking a 256-color ramp across it."""
    out = []
    for i, char in enumerate(text):
        shade = colors[min(i * len(colors) // max(len(text), 1), len(colors) - 1)]
        out.append(f"\033[1;38;5;{shade}m{char}")
    return "".join(out) + RESET


def render_banner(model, mode, workdir, color=None):
    """Build the startup splash: the astra, the wordmark, then the settings."""
    color = _color() if color is None else color
    unicode_ok = _unicode_ok()
    art = WORDMARK if unicode_ok else WORDMARK_ASCII
    emblem = EMBLEM if unicode_ok else EMBLEM_ASCII
    emblem_width = max(len(row) for row in emblem)
    width = max(len(art[0]), len(TAGLINE), emblem_width)

    def centered(row):
        return " " * ((width - len(row)) // 2) + row

    # The emblem is one drawing, so every row shifts by the same amount --
    # centering rows individually would pull the bow apart.
    emblem_pad = " " * ((width - emblem_width) // 2)

    lines = []
    for i, row in enumerate(emblem):
        if not color:
            lines.append((emblem_pad + row).rstrip())
        elif i == SHAFT_ROW:
            # Only the drawn arrow walks the ramp -- indent it, don't paint it.
            indent = len(row) - len(row.lstrip())
            lines.append(emblem_pad + " " * indent
                         + _gradient(row.lstrip(), SHAFT_COLORS))
        else:
            lines.append(f"{emblem_pad}\033[1;38;5;{EMBLEM_COLORS[i]}m{row}{RESET}")
    lines.append("")

    for i, row in enumerate(art):
        if color:
            shade = WORDMARK_COLORS[i * len(WORDMARK_COLORS) // len(art)]
            lines.append(f"\033[1;38;5;{shade}m{centered(row)}{RESET}")
        else:
            lines.append(centered(row))

    lines.append(f"{DIM if color else ''}{centered(TAGLINE)}{RESET if color else ''}")
    lines.append("")
    lines.append(BANNER_INFO.format(
        dim=DIM if color else "", reset=RESET if color else "",
        model=model, mode=mode, workdir=workdir))
    return "\n".join(lines)


def _color(stream=None):
    """True when the output is a terminal that will render escape codes."""
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


def _clip(value, limit):
    """Render a value as a single short line."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def make_printer(color=None):
    """Build the on_event callback: assistant text, tool calls, dimmed results."""
    color = _color() if color is None else color
    dim, reset = (DIM, RESET) if color else ("", "")

    def on_event(kind, payload):
        if kind == "assistant":
            if payload["text"]:
                print(payload["text"])
        elif kind == "tool_start":
            args = ", ".join(f"{k}={_clip(v, ARG_CLIP)}" for k, v in payload["args"].items())
            print(f"{dim}> {payload['name']}({args}){reset}")
        elif kind == "tool_end":
            first = payload["text"].strip().split("\n")[0] if payload["text"] else ""
            print(f"{dim}  {_clip(first, RESULT_CLIP)}{reset}")

    return on_event


def make_approver(color=None):
    """Build the safe-mode approver: show the call, ask, default to no."""
    color = _color() if color is None else color
    bold, reset = (BOLD, RESET) if color else ("", "")

    def approve(call, reason):
        args = ", ".join(f"{k}={_clip(v, ARG_CLIP)}" for k, v in call.get("args", {}).items())
        print(f"\n{bold}approve {call['name']}({args})?{reset} [y/N] ", end="", flush=True)
        try:
            return input().strip().lower() in ("y", "yes")
        except EOFError:
            # No one is there to answer; a silent yes would be the wrong guess.
            print()
            return False

    return approve


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="astra", description="A small agent harness.")
    parser.add_argument("-p", "--prompt", help="run one task headlessly and exit")
    parser.add_argument("-d", "--workdir", default=".",
                        help="directory the agent is confined to (default: .)")
    parser.add_argument("-m", "--model", help="model id (default: $ASTRA_MODEL)")
    parser.add_argument("--mode", choices=["safe", "yolo", "read-only"],
                        help="tool policy (default: safe interactively, yolo with -p)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the most recent session in the workdir")
    parser.add_argument("--max-turns", type=int, default=120,
                        help="turn limit for a single task (default: 120)")
    parser.add_argument("--web", action="store_true",
                        help="serve the browser studio instead of a prompt loop")
    parser.add_argument("--projects", default="./projects",
                        help="directory holding studio projects (default: ./projects)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address the studio binds to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7777,
                        help="port the studio listens on (default: 7777)")
    return parser


def main(argv=None):
    """Entry point for `python3 -m astra`."""
    args = build_parser().parse_args(argv)
    headless = args.prompt is not None
    mode = args.mode or ("yolo" if headless else "safe")

    if args.web:
        # A browser can answer "approve this?" through the event stream, but
        # a studio is for building, so the default matches -p rather than
        # the interactive prompt loop.
        from . import web
        return web.serve(root=args.projects, host=args.host, port=args.port,
                         model=args.model, mode=args.mode or "yolo",
                         max_turns=args.max_turns)

    agent = Harness(
        workdir=args.workdir,
        model=args.model,
        policy=Policy(mode, approver=make_approver()),
        on_event=make_printer(),
        max_turns=args.max_turns,
    )

    if args.resume and agent.resume():
        print(f"{DIM if _color() else ''}resumed {os.path.basename(agent.session_path)} "
              f"({len(agent.messages)} messages){RESET if _color() else ''}")

    if headless:
        agent.run(args.prompt)
        return 0

    return interactive(agent, mode)


def interactive(agent, mode):
    """Run the prompt loop until Ctrl-D."""
    print(render_banner(agent.model, mode, agent.workdir))

    while True:
        try:
            task = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        if not task:
            continue

        try:
            agent.run(task)
        except KeyboardInterrupt:
            # The transcript is already on disk; say where it is and carry on.
            name = os.path.basename(agent.session_path) if agent.session_path else "(none)"
            print(f"\n\ninterrupted. session log is safe: {name}")
            print("continue it with:  python3 -m astra --resume")
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
