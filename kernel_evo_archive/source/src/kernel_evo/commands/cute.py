"""Top-level adapter for ``kernel-evo cute``."""

from kernel_evo.cute_harness.cli import command as cute_command
from kernel_evo.cute_harness.cli import setup_parser

__all__ = ["cute_command", "setup_parser"]

