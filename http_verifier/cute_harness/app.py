from __future__ import annotations

import asyncio
import hmac
from functools import lru_cache

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from .config import Settings
from .models import MAX_ITERATIONS, Profiler, RunRequest, RunResponse
from .policy import UnsafeSourceError, validate_source
from .runner import HarnessRunner


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache
def get_runner() -> HarnessRunner:
    return HarnessRunner(get_settings())


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = get_settings().api_key
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
        )


app = FastAPI(
    title="Cute Harness",
    version="0.1.0",
    description="Run and profile a single PyTorch/CuTe Python source file.",
)
_gpu_slots = asyncio.Semaphore(1)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/runs",
    response_model=RunResponse,
    dependencies=[Depends(require_api_key)],
)
async def create_run(request: RunRequest) -> RunResponse:
    return await _validated_run(
        request.code,
        request.filename,
        request.profiler,
        request.iterations,
        request.exclusive,
    )


@app.post(
    "/v1/runs/file",
    response_model=RunResponse,
    dependencies=[Depends(require_api_key)],
)
async def create_run_from_file(
    file: UploadFile = File(...),
    profiler: str | None = Form(default=None),
    iterations: int = Form(default=1, ge=1, le=MAX_ITERATIONS),
    exclusive: bool = Form(default=False),
) -> RunResponse:
    content = await file.read(get_settings().max_source_bytes + 1)
    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="source must be UTF-8") from exc
    try:
        selected_profiler = None if profiler in (None, "") else Profiler(profiler)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="profiler must be 'pytorch' or 'nsys'",
        ) from exc
    return await _validated_run(
        code,
        file.filename or "submission.py",
        selected_profiler,
        iterations,
        exclusive,
    )


async def _validated_run(
    code: str,
    filename: str,
    profiler: Profiler | None,
    iterations: int,
    exclusive: bool,
) -> RunResponse:
    encoded = code.encode("utf-8")
    if len(encoded) > get_settings().max_source_bytes:
        raise HTTPException(status_code=413, detail="source file is too large")
    if not filename.endswith(".py"):
        raise HTTPException(status_code=422, detail="filename must end in .py")
    try:
        validate_source(code)
    except UnsafeSourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _gpu_slots:
        return await asyncio.to_thread(
            get_runner().run,
            code,
            profiler,
            iterations,
            exclusive,
        )


@app.get(
    "/v1/profiles/{profile_id}",
    response_class=FileResponse,
    dependencies=[Depends(require_api_key)],
)
async def download_profile(profile_id: str) -> Response:
    path = get_runner().artifact_path(profile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="profile not found")
    media_type = (
        "application/json"
        if path.suffix == ".json"
        else "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type, filename=path.name)


def main() -> None:
    uvicorn.run("cute_harness.app:app", host="0.0.0.0", port=18080)


if __name__ == "__main__":
    main()
