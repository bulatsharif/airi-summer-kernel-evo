"""Stable paths for the packaged CuTe DSL laboratory."""

from __future__ import annotations

from pathlib import Path


HARNESS_ROOT = Path(__file__).resolve().parent


def harness_root() -> Path:
    return HARNESS_ROOT


def resource_path(relative: str | Path) -> Path:
    candidate = (HARNESS_ROOT / relative).resolve()
    if not candidate.is_relative_to(HARNESS_ROOT):
        raise ValueError(f"CuTe harness resource escapes its root: {relative}")
    return candidate

