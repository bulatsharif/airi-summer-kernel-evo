from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid


class RemoteHarnessError(RuntimeError):
    pass


def _validate_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RemoteHarnessError(
            f"invalid harness URL {base_url!r}; expected http(s)://host[:port]"
        )
    return normalized


def build_multipart(
    submission_path: Path,
    profiler: str,
) -> tuple[bytes, str]:
    boundary = f"cute-harness-{uuid.uuid4().hex}"
    newline = b"\r\n"
    file_bytes = submission_path.read_bytes()
    filename = submission_path.name.replace('"', "_")

    chunks = [
        f"--{boundary}".encode(),
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"'
        ).encode(),
        b"Content-Type: text/x-python",
        b"",
        file_bytes,
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="profiler"',
        b"",
        profiler.encode(),
        f"--{boundary}--".encode(),
        b"",
    ]
    return newline.join(chunks), boundary


class HarnessClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 360.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        if not api_key:
            raise RemoteHarnessError("API key is empty")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _open(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RemoteHarnessError(
                f"harness HTTP {error.code}: {body[:1000]}"
            ) from error
        except URLError as error:
            raise RemoteHarnessError(
                f"cannot reach harness at {self.base_url}: {error.reason}"
            ) from error

    def run_file(
        self,
        submission_path: Path,
        profiler: str = "pytorch",
    ) -> dict[str, Any]:
        body, boundary = build_multipart(submission_path, profiler)
        request = Request(
            f"{self.base_url}/v1/runs/file",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "User-Agent": "cute-agent-harness/0.1",
                "X-API-Key": self.api_key,
            },
        )
        payload = self._open(request)
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteHarnessError(
                "harness returned a non-JSON run response"
            ) from error
        if not isinstance(result, dict):
            raise RemoteHarnessError("harness run response is not an object")
        return result

    def download_profile(self, profile_id: str) -> bytes:
        if not profile_id or "/" in profile_id or "\\" in profile_id:
            raise RemoteHarnessError(f"invalid profile id: {profile_id!r}")
        request = Request(
            f"{self.base_url}/v1/profiles/{profile_id}",
            method="GET",
            headers={
                "User-Agent": "cute-agent-harness/0.1",
                "X-API-Key": self.api_key,
            },
        )
        return self._open(request)
