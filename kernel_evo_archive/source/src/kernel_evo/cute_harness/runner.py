"""Structured subprocess runners for compile, correctness, sanitizer, and profiling probes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in value.split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _last_json(stdout: str) -> Mapping[str, Any] | None:
    for candidate in reversed([stdout.strip(), *[line.strip() for line in stdout.splitlines() if line.strip()]]):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _phase(output: str) -> str:
    lowered = output.lower()
    if "dslruntimeerror" in lowered or "preprocess" in lowered or "traceback" in lowered:
        return "python_or_dsl_lowering"
    if "nvvm" in lowered or "compilationerror" in lowered or "ptxas" in lowered:
        return "nvvm_or_ptxas"
    if "cuda error" in lowered or "illegal memory" in lowered or "misaligned" in lowered:
        return "cuda_runtime"
    return "command"


def _configure_artifacts(env: dict[str, str], artifact_dir: Path, *, debug: bool) -> str:
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        version = "0"
    if _version_tuple(version) >= (4, 4):
        env["CUTE_DSL_KEEP"] = "ir,ptx,cubin,sass"
        env["CUTE_DSL_DUMP_DIR"] = str(artifact_dir)
        if debug:
            env["CUTE_DSL_DEBUG"] = "1"
    else:
        env["CUTE_DSL_KEEP_IR"] = "1"
        if debug:
            env["CUTE_DSL_FILTER_STACKTRACE"] = "0"
            env["CUTE_DSL_ENABLE_OPTIMIZATION_WARNINGS"] = "1"
    return version


def run_command(
    command: Sequence[str],
    *,
    kind: str,
    arch: str,
    cwd: str | Path,
    artifact_dir: str | Path,
    timeout: float = 300.0,
    debug: bool = False,
    extra_env: Mapping[str, str] | None = None,
    expectations: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    workdir = Path(cwd).expanduser().resolve()
    artifacts = Path(artifact_dir).expanduser().resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for pattern in ("*.mlir", "*.ptx", "*.cubin", "*.sass") for path in workdir.glob(pattern)}
    env = os.environ.copy()
    env["CUTE_DSL_ARCH"] = arch
    env["CUTE_HARNESS_ARTIFACT_DIR"] = str(artifacts)
    env.update({str(key): str(value) for key, value in (extra_env or {}).items()})
    version = _configure_artifacts(env, artifacts, debug=debug)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return {
            "kind": kind,
            "success": False,
            "timed_out": True,
            "elapsed_s": elapsed,
            "phase": "timeout",
            "primary_error": f"Command exceeded {timeout}s",
            "stdout": (exc.stdout or "")[-8_000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8_000:] if isinstance(exc.stderr, str) else "",
            "artifacts": [],
        }
    elapsed = time.monotonic() - started
    generated = [
        path.resolve()
        for pattern in ("*.mlir", "*.ptx", "*.cubin", "*.sass")
        for path in workdir.glob(pattern)
        if path.resolve() not in before
    ]
    retained: list[str] = []
    for source in generated:
        destination = artifacts / source.name
        if source != destination:
            destination = Path(shutil.move(str(source), str(destination)))
        retained.append(str(destination.resolve()))
    retained.extend(
        str(path.resolve())
        for path in artifacts.iterdir()
        if path.is_file() and str(path.resolve()) not in retained
    )
    combined = f"{completed.stderr}\n{completed.stdout}".strip()
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    payload = _last_json(completed.stdout)
    codegen: list[dict[str, Any]] = []
    if completed.returncode == 0:
        from kernel_evo.cute_harness.codegen import inspect_artifact

        for retained_path in retained:
            if Path(retained_path).suffix.lower() in {".cubin", ".ptx", ".sass"}:
                try:
                    codegen.append(inspect_artifact(retained_path))
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    codegen.append({"path": retained_path, "error": str(exc)})
    codegen_gate: dict[str, Any] = {}
    if completed.returncode == 0 and expectations:
        from kernel_evo.cute_harness.codegen import verify_codegen_reports

        candidates = [item for item in codegen if not item.get("error")]
        codegen_gate = verify_codegen_reports(candidates, expectations)
    command_success = completed.returncode == 0
    if codegen_gate:
        command_success = command_success and bool(codegen_gate.get("passed"))
    gate_error = ""
    if codegen_gate and not codegen_gate.get("passed"):
        failures = codegen_gate.get("failures", [])
        gate_error = "; ".join(
            str(item.get("message", "code-generation contract failed"))
            for item in failures
            if isinstance(item, Mapping)
        )[:2_000]
    return {
        "kind": kind,
        "success": command_success,
        "timed_out": timed_out,
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "phase": (
            "complete"
            if command_success
            else ("codegen_contract" if completed.returncode == 0 else _phase(combined))
        ),
        "primary_error": (
            ""
            if command_success
            else gate_error or (lines[-1] if lines else "command failed")[:2_000]
        ),
        "stdout": completed.stdout[-8_000:],
        "stderr": completed.stderr[-8_000:],
        "metrics": dict(payload) if payload else {},
        "artifacts": sorted(set(retained)),
        "codegen": codegen,
        "codegen_gate": codegen_gate,
        "environment": {"dialect": "cute_dsl_python", "cutlass_version": version, "arch": arch},
    }


def sanitizer_command(
    command: Sequence[str],
    *,
    tool: str,
    arch: str,
    cwd: str | Path,
    artifact_dir: str | Path,
    timeout: float = 600.0,
    expectations: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tool not in {"memcheck", "racecheck", "initcheck", "synccheck"}:
        raise ValueError(f"Unsupported sanitizer tool: {tool}")
    executable = shutil.which("compute-sanitizer")
    if not executable:
        raise RuntimeError("compute-sanitizer is not available")
    wrapped = [executable, "--tool", tool, "--error-exitcode", "86", *command]
    result = run_command(
        wrapped,
        kind=f"sanitizer:{tool}",
        arch=arch,
        cwd=cwd,
        artifact_dir=artifact_dir,
        timeout=timeout,
        expectations=expectations,
    )
    result["sanitizer_errors"] = result.get("returncode") == 86
    return result


def profile_command(
    command: Sequence[str],
    *,
    arch: str,
    cwd: str | Path,
    artifact_dir: str | Path,
    section_set: str = "basic",
    timeout: float = 900.0,
    expectations: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executable = shutil.which("ncu")
    if not executable:
        raise RuntimeError("Nsight Compute (ncu) is not available")
    report = Path(artifact_dir).expanduser().resolve() / "profile"
    wrapped = [
        executable,
        "--set",
        section_set,
        "--target-processes",
        "all",
        "--export",
        str(report),
        "--force-overwrite",
        *command,
    ]
    return run_command(
        wrapped,
        kind="profile:ncu",
        arch=arch,
        cwd=cwd,
        artifact_dir=artifact_dir,
        timeout=timeout,
        expectations=expectations,
    )
