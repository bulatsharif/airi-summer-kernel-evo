from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Change this before exposing the service. It is intentionally a single constant,
# as requested, while the environment override makes deployment without a source
# edit possible.
DEFAULT_API_KEY = "cute-harness-change-me-7fcb9a3d"


@dataclass(frozen=True)
class Settings:
    api_key: str
    artifact_dir: Path
    timeout_seconds: int
    max_source_bytes: int
    max_log_bytes: int
    python_executable: str
    nsys_executable: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("CUTE_HARNESS_API_KEY", DEFAULT_API_KEY),
            artifact_dir=Path(
                os.getenv("CUTE_HARNESS_ARTIFACT_DIR", "/tmp/cute-harness-profiles")
            ),
            timeout_seconds=int(os.getenv("CUTE_HARNESS_TIMEOUT_SECONDS", "300")),
            max_source_bytes=int(
                os.getenv("CUTE_HARNESS_MAX_SOURCE_BYTES", str(1024 * 1024))
            ),
            max_log_bytes=int(
                os.getenv("CUTE_HARNESS_MAX_LOG_BYTES", str(10 * 1024 * 1024))
            ),
            python_executable=os.getenv("CUTE_HARNESS_PYTHON", "python3"),
            nsys_executable=os.getenv("CUTE_HARNESS_NSYS", "nsys"),
        )
