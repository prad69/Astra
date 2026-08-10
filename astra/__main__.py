"""Package entry point, so `python3 -m astra` reaches the CLI.

It defers rather than implements: everything lives in cli.py, and this
file exists so the module has a front door.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
