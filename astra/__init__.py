"""Astra -- a small agent harness.

The public surface is deliberately four names: build a Harness, gate it
with a Policy, and extend it with tools written using the tool decorator.
Everything else in the package is a module you import directly when you
want to reach past the composition.
"""

from .fleet import run_fleet
from .harness import Harness
from .security import Policy
from .tools import Tool, tool

__all__ = ["Harness", "Policy", "Tool", "tool", "run_fleet"]
