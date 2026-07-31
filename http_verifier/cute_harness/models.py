from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


MAX_ITERATIONS = 100


class Profiler(str, Enum):
    pytorch = "pytorch"
    nsys = "nsys"


class RunRequest(BaseModel):
    code: str = Field(description="Complete Python source including its main section")
    filename: str = "submission.py"
    profiler: Profiler | None = None
    iterations: int = Field(default=1, ge=1, le=MAX_ITERATIONS)
    exclusive: bool = False


class RunResponse(BaseModel):
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    device_time_ms: float | None = None
    device_times_ms: list[float] = Field(default_factory=list)
    profile_id: str | None = None
    profile_error: str | None = None
    timed_out: bool = False
