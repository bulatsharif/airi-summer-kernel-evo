from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from kernel_evo.core.profile.contracts import run_profile_subprocess


def _resolve_nsys(configured: str) -> str | None:
    value = str(configured or "nsys").strip() or "nsys"
    if "/" in value:
        path = Path(value).expanduser().resolve()
        return str(path) if path.is_file() else None
    if value == "nsys":
        installed = sorted(
            Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True
        )
        if installed:
            return str(installed[0].resolve())
    return shutil.which(value)


def _preflight_cache(run_config: dict[str, Any], out_dir: Path) -> Path:
    experiment_dir = str(run_config.get("experiment_dir", "") or "").strip()
    if experiment_dir:
        return Path(experiment_dir).expanduser().resolve() / "nsys_host_preflight.json"
    artifacts_dir = str(run_config.get("profile_artifacts_dir", "") or "").strip()
    if artifacts_dir:
        return Path(artifacts_dir).expanduser().resolve() / "nsys_host_preflight.json"
    return out_dir.parent / "nsys_host_preflight.json"


def _load_or_run_preflight(
    *, nsys: str, run_config: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    cache = _preflight_cache(run_config, out_dir)
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("nsys") == nsys:
                return cached
        except Exception:
            pass

    with tempfile.TemporaryDirectory(prefix="kernel_evo_nsys_probe_") as tmpdir:
        report_base = Path(tmpdir) / "probe"
        command = [
            nsys,
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output",
            str(report_base),
            "/bin/true",
        ]
        completed = run_profile_subprocess(command, timeout=75, text=True, capture_output=True)
        report = report_base.with_suffix(".nsys-rep")
        preflight = {
            "available": completed.returncode == 0 and report.is_file(),
            "nsys": nsys,
            "returncode": completed.returncode,
            "reason": (
                "nsys host preflight succeeded"
                if completed.returncode == 0 and report.is_file()
                else "nsys agent preflight failed; timeline profiling disabled"
            ),
            "stdout_excerpt": (completed.stdout or "")[-1_000:],
            "stderr_excerpt": (completed.stderr or "")[-1_000:],
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    return preflight


def _profile_command(
    *,
    nsys: str,
    report_base: Path,
    run_config_path: Path,
    candidate_file: Path,
    reference_file: Path,
    target_out_dir: Path,
) -> list[str]:
    return [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--force-overwrite=true",
        "--output",
        str(report_base),
        sys.executable,
        "-m",
        "kernel_evo.tools.profile_target",
        "--run-config",
        str(run_config_path),
        "--candidate-file",
        str(candidate_file),
        "--reference-file",
        str(reference_file),
        "--out-dir",
        str(target_out_dir),
    ]


def run_nsys_profile(
    *,
    run_config_path: Path,
    candidate_file: Path,
    reference_file: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    configured = str(run_config.get("profile_nsys_path", "nsys") or "nsys")
    nsys = _resolve_nsys(configured)
    if nsys is None:
        summary = {"status": "unavailable", "reason": f"nsys executable not found: {configured}"}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    host_preflight = _load_or_run_preflight(nsys=nsys, run_config=run_config, out_dir=out_dir)
    if not bool(host_preflight.get("available")):
        summary = {
            "status": "unavailable",
            "reason": str(host_preflight.get("reason", "nsys host preflight failed")),
            "host_preflight": host_preflight,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    report_base = out_dir / "timeline"
    target_out_dir = out_dir / "target_run"
    command = _profile_command(
        nsys=nsys,
        report_base=report_base,
        run_config_path=run_config_path,
        candidate_file=candidate_file,
        reference_file=reference_file,
        target_out_dir=target_out_dir,
    )
    profiled = run_profile_subprocess(command, timeout=240, text=True, capture_output=True)
    report_file = report_base.with_suffix(".nsys-rep")

    stats_file = out_dir / "stats.txt"
    stats_command = [
        nsys,
        "stats",
        "--report",
        "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
        "--format",
        "table",
        "--force-export=true",
        str(report_file),
    ]
    stats = None
    if report_file.is_file():
        stats = run_profile_subprocess(stats_command, timeout=120, text=True, capture_output=True)
        stats_file.write_text((stats.stdout or "") + (stats.stderr or ""), encoding="utf-8")

    sqlite_file = report_base.with_suffix(".sqlite")
    summary: dict[str, Any] = {
        "status": "completed" if profiled.returncode == 0 and report_file.is_file() else "failed",
        "returncode": profiled.returncode,
        "report_file": str(report_file) if report_file.is_file() else "",
        "sqlite_file": str(sqlite_file) if sqlite_file.is_file() else "",
        "stats_file": str(stats_file) if stats_file.is_file() else "",
        "stats_returncode": stats.returncode if stats is not None else None,
        "command": command,
        "stdout_excerpt": (profiled.stdout or "")[-4_000:],
        "stderr_excerpt": (profiled.stderr or "")[-4_000:],
        "stats_excerpt": (stats_file.read_text(encoding="utf-8")[-20_000:] if stats_file.is_file() else ""),
        "host_preflight": host_preflight,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture an Nsight Systems timeline for a candidate.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = run_nsys_profile(
        run_config_path=Path(args.run_config).expanduser().resolve(),
        candidate_file=Path(args.candidate_file).expanduser().resolve(),
        reference_file=Path(args.reference_file).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
