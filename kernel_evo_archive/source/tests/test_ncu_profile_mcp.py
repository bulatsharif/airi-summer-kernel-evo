from __future__ import annotations

from pathlib import Path

import pytest

from kernel_evo.tools import ncu_profile_mcp
from kernel_evo.tools.ncu_gpu_guard import GuardedProcessResult, NcuGpuGuard
from kernel_evo.tools.ncu_profile_mcp import (
    RestrictedNcuService, _compact_csv,
)


def test_guard_selects_lowest_memory_gpu_after_consecutive_idle_samples(monkeypatch) -> None:
    guard = NcuGpuGuard({4, 5}, idle_samples=2, poll_seconds=0)
    states = {
        4: {"utilization": 0, "memory_used": 1000},
        5: {"utilization": 0, "memory_used": 10},
    }
    monkeypatch.setattr(guard, "_gpu_states", lambda: states)

    assert guard.select_idle_gpu(timeout=1) == 5


def test_compact_csv_keeps_only_actionable_hardware_counters() -> None:
    text = "\n".join(
        [
            '"ID","Kernel Name","gpu__time_duration.sum","dram__bytes.sum.per_second",'
            '"sm__throughput.avg.pct_of_peak_sustained_elapsed"',
            '"","","us","Tbyte/s","%"',
            '"0","fp8_gemm","12.5","1.5","25.0"',
        ]
    )

    compact = _compact_csv(text)

    assert compact["status"] == "completed"
    assert compact["kernel_count"] == 1
    assert compact["kernels"][0]["metrics"] == {
        "duration_us": {"value": 12.5, "unit": "us"},
        "dram_throughput": {"value": 1.5, "unit": "Tbyte/s"},
        "sm_peak_pct": {"value": 25.0, "unit": "%"},
    }


def test_service_rejects_candidate_outside_experiment(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "profile_fp8_baseline.py").write_text("", encoding="utf-8")
    outside = tmp_path / "candidate.py"
    outside.write_text("", encoding="utf-8")
    service = RestrictedNcuService(
        experiment_root=root,
        allowed_devices={5},
        ncu_path=Path("/usr/local/cuda/bin/ncu"),
        python_path=Path("/usr/bin/python3"),
        timeout=60,
    )

    with pytest.raises(ValueError, match="inside the experiment"):
        service.validate_candidate(str(outside))


def test_guarded_job_invalidates_and_retries_external_gpu_activity(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    driver = root / "profile_fp8_baseline.py"
    driver.write_text("", encoding="utf-8")
    candidate = root / "production_fp8.py"
    candidate.write_text("", encoding="utf-8")
    service = RestrictedNcuService(
        experiment_root=root,
        allowed_devices={4, 5},
        ncu_path=Path("/usr/local/cuda/bin/ncu"),
        python_path=Path("/usr/bin/python3"),
        timeout=60,
    )

    class AlwaysContaminated:
        selections = 0

        def select_idle_gpu(self, *, timeout: float) -> int:
            self.selections += 1
            return 4 if self.selections % 2 else 5

        def run_monitored(self, *args, **kwargs) -> GuardedProcessResult:
            return GuardedProcessResult(
                returncode=-15,
                stdout="",
                stderr="",
                contaminated=True,
                contaminating_pids=(999,),
                timed_out=False,
            )

    service.gpu_guard = AlwaysContaminated()
    job_dir = root / "external_ncu_reports" / "job"
    job_dir.mkdir()
    service.jobs["job"] = {
        "job_id": "job",
        "status": "queued",
        "candidate": str(candidate),
        "job_dir": str(job_dir),
    }

    service._run_guarded("job", candidate, "hot", job_dir)

    result = service.status("job")
    assert result["status"] == "failed"
    assert [attempt["device"] for attempt in result["attempts"]] == [4, 5, 4]
    assert all(attempt["contaminated"] for attempt in result["attempts"])


def test_profile_command_runs_ncu_and_target_as_current_identity(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "profile_fp8_baseline.py").write_text("", encoding="utf-8")
    candidate = root / "production_fp8.py"
    candidate.write_text("", encoding="utf-8")
    service = RestrictedNcuService(
        experiment_root=root,
        allowed_devices={5},
        ncu_path=Path("/usr/local/cuda/bin/ncu"),
        python_path=Path("/usr/bin/python3"),
        timeout=60,
    )

    command = service._profile_command(
        candidate=candidate,
        device=5,
        scope="full",
        report_base=root / "report",
    )

    assert Path(command[0]) == Path("/usr/local/cuda/bin/ncu").resolve()
    assert str(Path("/usr/bin/python3").resolve()) in command
    assert "--drop-uid" not in command
    assert "--drop-gid" not in command
    assert "sudo" not in command
    assert "setpriv" not in command


def test_profile_environment_pins_system_cuda_toolkit(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "profile_fp8_baseline.py").write_text("", encoding="utf-8")
    service = RestrictedNcuService(
        experiment_root=root,
        allowed_devices={5},
        ncu_path=Path("/usr/local/cuda/bin/ncu"),
        python_path=Path("/usr/bin/python3"),
        timeout=60,
    )

    env = service._profile_env()

    assert Path(env["CUDA_HOME"]) == Path("/usr/local/cuda").resolve()
    assert Path(env["CUDA_PATH"]) == Path("/usr/local/cuda").resolve()
    assert Path(env["PATH"].split(":")[0]) == Path("/usr/local/cuda/bin").resolve()


def test_non_root_rejects_admin_only_counter_policy(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "profile_fp8_baseline.py").write_text("", encoding="utf-8")
    candidate = root / "production_fp8.py"
    candidate.write_text("", encoding="utf-8")
    service = RestrictedNcuService(
        experiment_root=root,
        allowed_devices={5},
        ncu_path=Path("/usr/local/cuda/bin/ncu"),
        python_path=Path("/usr/bin/python3"),
        timeout=60,
    )
    monkeypatch.setattr(ncu_profile_mcp.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        ncu_profile_mcp, "_unprivileged_counters_enabled", lambda: False
    )

    with pytest.raises(RuntimeError, match="admin-only"):
        service.start(str(candidate), "full")
