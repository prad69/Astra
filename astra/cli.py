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

from .harness import Harness
from .security import Policy

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

ARG_CLIP = 70
RESULT_CLIP = 160

BANNER = """{bold}Astra{reset}
  model {model}
  mode  {mode}
  jail  {workdir}
{dim}Ctrl-D to exit, Ctrl-C to interrupt a run.{reset}"""


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
    return parser


def main(argv=None):
    """Entry point for `python3 -m astra`."""
    args = build_parser().parse_args(argv)
    headless = args.prompt is not None
    mode = args.mode or ("yolo" if headless else "safe")

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
    color = _color()
    print(BANNER.format(
        bold=BOLD if color else "", reset=RESET if color else "",
        dim=DIM if color else "",
        model=agent.model, mode=mode, workdir=agent.workdir))

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
