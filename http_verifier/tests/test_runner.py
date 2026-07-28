from pathlib import Path
import json
import subprocess
import sys

from cute_harness.config import Settings
from cute_harness.models import Profiler
from cute_harness.runner import HarnessRunner


def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test",
        artifact_dir=tmp_path / "artifacts",
        timeout_seconds=5,
        max_source_bytes=1024,
        max_log_bytes=1024,
        python_executable="python3",
        nsys_executable="definitely-not-installed-nsys",
    )


def test_returns_full_worker_logs(tmp_path: Path) -> None:
    fake_worker = tmp_path / "worker.py"
    fake_worker.write_text(
        "import sys\nprint('normal log')\nprint('full failure log', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    runner = HarnessRunner(settings(tmp_path))
    runner.worker = fake_worker
    result = runner.run("print('not reached')", None)
    assert not result.success
    assert result.exit_code == 7
    assert result.stdout == "normal log\n"
    assert result.stderr == "full failure log\n"


def test_nsys_error_does_not_hide_successful_run(monkeypatch, tmp_path: Path) -> None:
    runner = HarnessRunner(settings(tmp_path))
    monkeypatch.setattr(
        runner,
        "_execute",
        lambda command, cwd: __import__(
            "cute_harness.runner", fromlist=["ProcessResult"]
        ).ProcessResult(0, "hello\n", "", False),
    )
    monkeypatch.setattr(
        runner,
        "_load_metadata",
        lambda path: {"success": True, "device_time_ms": 1.25},
    )
    result = runner.run("print('hello')", Profiler.nsys)
    assert result.success
    assert result.device_time_ms == 1.25
    assert result.profile_id is None
    assert "not installed" in (result.profile_error or "")


def test_worker_does_not_leak_wrapper_arguments_to_submission(tmp_path: Path) -> None:
    source = tmp_path / "submission.py"
    result_path = tmp_path / "result.json"
    source.write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\nprint('parsed')\n",
        encoding="utf-8",
    )
    worker = Path(__file__).parents[1] / "cute_harness" / "worker.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--source",
            str(source),
            "--result",
            str(result_path),
            "--no-torch-profile",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == "parsed\n"
    assert json.loads(result_path.read_text(encoding="utf-8"))["success"] is True
