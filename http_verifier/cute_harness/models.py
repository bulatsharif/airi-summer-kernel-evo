from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Profiler(str, Enum):
    pytorch = "pytorch"
    nsys = "nsys"


class RunRequest(BaseModel):
    code: str = Field(description="Complete Python source including its main section")
    filename: str = "submission.py"
    profiler: Profiler | None = None


class RunResponse(BaseModel):
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    device_time_ms: float | None = None
    profile_id: str | None = None
    profile_error: str | None = None
    timed_out: bool = False
