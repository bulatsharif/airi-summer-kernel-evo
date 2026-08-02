"""Atomic JSON state and append-only event storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from kernel_evo.agent.errors import ConfigurationError, RunNotFoundError


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir).expanduser().resolve()

    def run_dir(self, run_id: str, *, must_exist: bool = True) -> Path:
        if not RUN_ID_PATTERN.fullmatch(str(run_id)):
            raise ConfigurationError(
                "run_id must contain only letters, numbers, '.', '_' or '-' (maximum 64 characters)"
            )
        path = self.runs_dir / str(run_id)
        if must_exist and not (path / "state.json").exists():
            raise RunNotFoundError(f"KernelEvo run not found: {run_id} under {self.runs_dir}")
        return path

    def create(self, run_id: str) -> Path:
        path = self.run_dir(run_id, must_exist=False)
        if path.exists() and any(path.iterdir()):
            raise ConfigurationError(f"Run already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_state(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "state.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunNotFoundError(f"Cannot read run state at {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RunNotFoundError(f"Invalid run state at {path}")
        return value

    def write_state(self, run_id: str, state: dict[str, Any]) -> None:
        run_dir = self.run_dir(run_id, must_exist=False)
        state["updated_at"] = utc_now()
        self.write_json(run_dir / "state.json", state)

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)

    @staticmethod
    def append_jsonl(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()

    @contextmanager
    def locked(self, run_id: str) -> Iterator[None]:
        """Serialize short state transitions; evaluations deliberately run outside this lock."""
        import fcntl

        lock_path = self.run_dir(run_id) / ".state.lock"
        with lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
