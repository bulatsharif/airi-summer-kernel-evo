"""Local idea/event tracking with optional best-effort HTTP mirroring."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from kernel_evo.agent.store import RunStore, utc_now


class EventTracker:
    def __init__(self, run_dir: Path, remote_url: str = "") -> None:
        self.path = run_dir / "tracker.jsonl"
        self.remote_url = str(remote_url or "").strip()

    def emit(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = {"timestamp": utc_now(), "event": event, **dict(payload)}
        RunStore.append_jsonl(self.path, record)
        if self.remote_url:
            self._mirror(record)
        return record

    def _mirror(self, record: Mapping[str, Any]) -> None:
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.remote_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0):  # noqa: S310 - user configured endpoint
                pass
        except Exception as exc:
            RunStore.append_jsonl(
                self.path,
                {
                    "timestamp": utc_now(),
                    "event": "tracker_sync_failed",
                    "remote_url": self.remote_url,
                    "error": str(exc),
                },
            )
