from __future__ import annotations

import subprocess
from pathlib import Path

from kernel_evo.tools import profile_ncu


def test_ncu_subprocess_env_uses_per_user_cache_instead_of_system_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("KERNELEVO_NCU_TMPDIR", raising=False)

    env, lock_dir = profile_ncu._ncu_subprocess_env({})

    assert lock_dir == (
        tmp_path / ".cache" / "kernel-evo" / "ncu-tmp" / f"uid-{profile_ncu.os.geteuid()}"
    )
    assert env["TMPDIR"] == str(lock_dir)
    assert lock_dir.is_dir()


def test_ncu_subprocess_env_honors_configured_tmpdir(tmp_path: Path) -> None:
    configured = tmp_path / "custom-ncu-locks"
    env, lock_dir = profile_ncu._ncu_subprocess_env(
        {"profile_ncu_tmpdir": str(configured)}
    )
    assert lock_dir == configured
    assert env["TMPDIR"] == str(configured)


def test_effective_target_device_uses_run_config_device_by_default() -> None:
    run_config = {
        "device": "cuda:7",
    }

    assert profile_ncu._effective_target_device(run_config) == "cuda:7"


def test_resolve_ncu_options_uses_device_index_by_default() -> None:
    run_config = {
        "device": "cuda:7",
    }

    devices, section_set, kernel_name, extra_args = profile_ncu._resolve_ncu_options(run_config=run_config)

    assert devices == "7"
    assert section_set == "full"
    assert kernel_name == ""
    assert extra_args == ""


def test_effective_target_device_ignores_removed_profile_ncu_devices_field() -> None:
    run_config = {
        "device": "cuda:7",
        "profile_ncu_devices": "2",
    }

    assert profile_ncu._effective_target_device(run_config) == "cuda:7"


def test_retry_with_stable_sections_for_speed_of_light_without_explicit_sections() -> None:
    should_retry = profile_ncu._should_retry_with_stable_sections(
        section_set="speedOfLight",
        extra_args="",
        no_kernels_profiled=True,
        report_exists=False,
    )

    assert should_retry is True


def test_no_retry_when_sections_are_already_explicit() -> None:
    should_retry = profile_ncu._should_retry_with_stable_sections(
        section_set="",
        extra_args="--section LaunchStats --section Occupancy",
        no_kernels_profiled=True,
        report_exists=False,
    )

    assert should_retry is False


def test_ncu_target_steps_do_not_inherit_torch_profile_steps() -> None:
    run_config = {
        "profile_torch_warmup_steps": 4,
        "profile_torch_active_steps": 8,
    }

    assert profile_ncu._profile_target_steps(run_config) == 1


def test_ncu_attempt_captures_one_prepared_forward(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(profile_ncu, "run_profile_subprocess", fake_run)
    run_config = {
        "device": "cuda:0",
        "profile_torch_warmup_steps": 4,
        "profile_torch_active_steps": 8,
        "profile_ncu_target_steps": 1,
        "profile_ncu_warmup_steps": 1,
        "profile_ncu_launch_count": 128,
        "profile_ncu_tmpdir": str(tmp_path / "ncu-tmp"),
    }

    result = profile_ncu._run_ncu_attempt(
        resolved_ncu="/fake/ncu",
        report_base=tmp_path / "report",
        run_config_path=tmp_path / "run_config.json",
        candidate_file=tmp_path / "candidate.py",
        reference_file=tmp_path / "reference.py",
        target_run_work=tmp_path / "target-run",
        run_config=run_config,
        set_override=None,
        kernel_name_override=None,
        extra_args_override=None,
        label="requested",
    )

    command = commands[0]
    assert command[command.index("--profile-from-start") + 1] == "off"
    assert command[command.index("--launch-count") + 1] == "128"
    assert command[command.index("--target-steps") + 1] == "1"
    assert command[command.index("--target-warmup-steps") + 1] == "1"
    assert "--cuda-profiler-range" in command
    assert result["target_steps"] == 1
    assert result["warmup_steps"] == 1
    assert result["capture_scope"] == "cuda_profiler_range"


def test_preflight_cache_is_invalidated_when_device_changes(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "artifact" / "ncu"
    out_dir.mkdir(parents=True)
    cache_file = tmp_path / "artifact" / "ncu_host_preflight.json"
    cache_file.write_text('{"available": true, "devices": "7"}', encoding="utf-8")

    run_config = {
        "profile_artifacts_dir": str(tmp_path / "artifact"),
        "device": "cuda:2",
    }

    calls: list[str] = []

    def fake_run_preflight(
        resolved_ncu: str,
        *,
        run_config: dict[str, object],
    ) -> dict[str, object]:
        devices, _, _, _ = profile_ncu._resolve_ncu_options(run_config=run_config)
        calls.append(devices)
        return {"available": False, "devices": "2", "reason": "rerun"}

    monkeypatch.setattr(profile_ncu, "_run_preflight", fake_run_preflight)

    result = profile_ncu._load_or_run_preflight(
        run_config=run_config,
        resolved_ncu="/fake/ncu",
        out_dir=out_dir,
    )

    assert result["devices"] == "2"
    assert calls == ["2"]
