from pathlib import Path

from fastapi.testclient import TestClient

from cute_harness import app as app_module
from cute_harness.config import Settings
from cute_harness.models import RunResponse


class FakeRunner:
    def run(self, code, profiler, iterations, exclusive):
        assert iterations == 3
        assert exclusive is True
        return RunResponse(
            success=True,
            exit_code=0,
            stdout=code,
            stderr="",
            device_time_ms=2.5,
            device_times_ms=[2.4, 2.5, 2.6],
        )

    def artifact_path(self, profile_id):
        return None


def test_api_key_is_required(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: Settings(
            api_key="secret",
            artifact_dir=tmp_path,
            timeout_seconds=1,
            max_source_bytes=1024,
            max_log_bytes=1024,
            python_executable="python3",
            nsys_executable="nsys",
        ),
    )
    monkeypatch.setattr(app_module, "get_runner", lambda: FakeRunner())
    client = TestClient(app_module.app)

    assert client.post("/v1/runs", json={"code": "print(1)"}).status_code == 401
    response = client.post(
        "/v1/runs",
        headers={"X-API-Key": "secret"},
        json={
            "code": "print(1)",
            "iterations": 3,
            "exclusive": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["device_time_ms"] == 2.5
    assert response.json()["device_times_ms"] == [2.4, 2.5, 2.6]


def test_file_upload_and_policy_rejection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: Settings(
            api_key="secret",
            artifact_dir=tmp_path,
            timeout_seconds=1,
            max_source_bytes=1024,
            max_log_bytes=1024,
            python_executable="python3",
            nsys_executable="nsys",
        ),
    )
    monkeypatch.setattr(app_module, "get_runner", lambda: FakeRunner())
    client = TestClient(app_module.app)
    headers = {"X-API-Key": "secret"}

    response = client.post(
        "/v1/runs/file",
        headers=headers,
        files={"file": ("kernel.py", b"print('uploaded')", "text/x-python")},
        data={
            "profiler": "pytorch",
            "iterations": "3",
            "exclusive": "true",
        },
    )
    assert response.status_code == 200
    assert response.json()["stdout"] == "print('uploaded')"

    rejected = client.post(
        "/v1/runs",
        headers=headers,
        json={"code": "import os\nos.system('id')"},
    )
    assert rejected.status_code == 422
    assert "os.system" in rejected.json()["detail"]
