from pathlib import Path
import json
import subprocess
import sys
import textwrap
import types

from cute_harness.config import Settings
from cute_harness.models import Profiler
from cute_harness.runner import HarnessRunner
from cute_harness.worker import _run_job


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
    result = runner.run("print('not reached')", None, exclusive=True)
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
    result = runner.run("print('hello')", Profiler.nsys, exclusive=True)
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


def test_default_runs_reuse_persistent_worker(tmp_path: Path) -> None:
    fake_worker = tmp_path / "persistent_worker.py"
    fake_worker.write_text(
        textwrap.dedent(
            """\
            import json
            import pathlib
            import sys

            count = 0
            for line in sys.stdin:
                request = json.loads(line)
                count += 1
                pathlib.Path(request["stdout"]).write_text(
                    f"worker run {count}\\n", encoding="utf-8"
                )
                pathlib.Path(request["stderr"]).write_text("", encoding="utf-8")
                pathlib.Path(request["result"]).write_text(
                    json.dumps({
                        "success": True,
                        "exit_code": 0,
                        "device_times_ms": [float(count)],
                    }),
                    encoding="utf-8",
                )
            """
        ),
        encoding="utf-8",
    )
    runner = HarnessRunner(settings(tmp_path))
    runner.worker = fake_worker
    try:
        first = runner.run("print('first')", None)
        process = runner._worker_process
        second = runner.run("print('second')", None)
    finally:
        runner.close()

    assert first.device_times_ms == [1.0]
    assert second.device_times_ms == [2.0]
    assert first.stdout == "worker run 1\n"
    assert second.stdout == "worker run 2\n"
    assert process is not None
    assert process.pid > 0


def test_worker_excludes_warmup_and_returns_every_iteration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def key_averages(self):
            return [types.SimpleNamespace(self_device_time_total=1250.0)]

    fake_torch = types.SimpleNamespace(
        profiler=types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=lambda activities: FakeProfile(),
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source = tmp_path / "submission.py"
    result_path = tmp_path / "result.json"
    counter_name = "_cute_harness_worker_test_counter"
    source.write_text(
        "import builtins\n"
        f"builtins.{counter_name} = getattr(builtins, "
        f"'{counter_name}', 0) + 1\n",
        encoding="utf-8",
    )
    try:
        assert _run_job(str(source), str(result_path), 3, None) == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert getattr(__import__("builtins"), counter_name) == 4
    finally:
        delattr(__import__("builtins"), counter_name)

    assert result["device_times_ms"] == [1.25, 1.25, 1.25]
    assert result["device_time_ms"] == 1.25
