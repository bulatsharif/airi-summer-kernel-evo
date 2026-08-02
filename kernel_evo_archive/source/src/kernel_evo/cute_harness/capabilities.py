"""Discover the exact Python CuTe DSL, CUDA, and GPU capability contract."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_ARCH_RE = re.compile(r"^(?:sm_|compute_)?(?P<major>\d+)(?:[._]?(?P<minor>\d))?(?P<a>a)?$")


def arch_from_compute_capability(major: int, minor: int) -> str:
    """Return the architecture spelling required for architecture-accelerated features."""
    number = f"{major}{minor}"
    if major in {9, 10}:
        return f"sm_{number}a"
    return f"sm_{number}"


def normalize_cute_arch(value: str, *, torch_style: bool = False) -> str:
    """Normalize common CUDA arch spellings without hiding a missing ``a`` suffix.

    Explicit ``sm_90`` remains ``sm_90`` so callers can diagnose that it cannot
    legally target Hopper WGMMA. Torch-style ``9.0`` becomes ``sm_90a``.
    """
    text = str(value or "").strip().lower().replace("+ptx", "")
    if not text:
        return ""
    text = re.split(r"[;,\s]+", text, maxsplit=1)[0]
    explicit_sm = text.startswith("sm_")
    match = _ARCH_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Unsupported CUDA architecture spelling: {value!r}")
    major_digits = match.group("major")
    minor_text = match.group("minor")
    if "." in text or "_" not in text and len(major_digits) == 1:
        major = int(major_digits)
        minor = int(minor_text or 0)
    elif len(major_digits) >= 2:
        major = int(major_digits[:-1])
        minor = int(major_digits[-1])
    else:
        major = int(major_digits)
        minor = int(minor_text or 0)
    suffix = "a" if match.group("a") else ""
    if torch_style and not suffix:
        return arch_from_compute_capability(major, minor)
    if explicit_sm:
        return f"sm_{major}{minor}{suffix}"
    return arch_from_compute_capability(major, minor) if not suffix else f"sm_{major}{minor}a"


def resolve_target_arch(
    *,
    explicit_arch: str = "",
    arch_list: str = "",
    device: int = 0,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = environ if environ is not None else os.environ
    if explicit_arch:
        return normalize_cute_arch(explicit_arch)
    if arch_list:
        return normalize_cute_arch(arch_list, torch_style=True)
    if env.get("CUTE_DSL_ARCH"):
        return normalize_cute_arch(str(env["CUTE_DSL_ARCH"]))
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            return arch_from_compute_capability(major, minor)
    except Exception:
        pass
    return ""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _has_symbol(module_name: str, symbol: str) -> bool:
    try:
        value: Any = importlib.import_module(module_name)
        for part in symbol.split("."):
            value = getattr(value, part)
        return value is not None
    except Exception:
        return False


def _driver_version() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ""
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return completed.stdout.splitlines()[0].strip() if completed.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _cuda_toolkit_version() -> str:
    executable = shutil.which("nvcc")
    if not executable:
        return ""
    try:
        completed = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        match = re.search(r"release\s+([0-9.]+)", completed.stdout)
        return match.group(1) if completed.returncode == 0 and match else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _tool_contract() -> dict[str, dict[str, Any]]:
    return {
        name: {"available": bool(path := shutil.which(name)), "path": path or ""}
        for name in ("nvcc", "cuobjdump", "nvdisasm", "compute-sanitizer", "ncu")
    }


@lru_cache(maxsize=16)
def _probe_cached(device: int, explicit_arch: str, arch_list: str) -> dict[str, Any]:
    cutlass_version = _package_version("nvidia-cutlass-dsl")
    target_arch = resolve_target_arch(
        explicit_arch=explicit_arch,
        arch_list=arch_list,
        device=device,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "dialect_scope": {
            "included": ["nvidia-cutlass-dsl", "cutlass.cute"],
            "excluded": ["CuTe C++", "legacy CUTLASS Python operation API"],
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {
            "nvidia-cutlass-dsl": cutlass_version,
            "torch": _package_version("torch"),
            "cuda-python": _package_version("cuda-python"),
        },
        "target_arch": target_arch,
        "gpu": {"available": False, "device": device},
        "cuda": {
            "driver": _driver_version(),
            "torch_runtime": "",
            "toolkit": _cuda_toolkit_version(),
        },
        "types": {},
        "features": {},
        "tools": _tool_contract(),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUTE_DSL_ARCH",
                "CUDA_TOOLKIT_PATH",
                "CUTE_DSL_KEEP_IR",
                "CUTE_DSL_ENABLE_OPTIMIZATION_WARNINGS",
            )
        },
        "issues": [],
    }

    try:
        import cutlass

        report["packages"]["cutlass_module"] = str(Path(cutlass.__file__).resolve())
        report["types"] = {
            name: bool(getattr(cutlass, name, None))
            for name in (
                "BFloat16",
                "Float16",
                "Float32",
                "Float8E4M3FN",
                "Float8E5M2",
            )
        }
    except Exception as exc:
        report["issues"].append(f"Python CuTe DSL import failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        report["cuda"]["torch_runtime"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            major, minor = torch.cuda.get_device_capability(device)
            report["gpu"] = {
                "available": True,
                "device": device,
                "name": torch.cuda.get_device_name(device),
                "compute_capability": f"{major}.{minor}",
                "native_arch": arch_from_compute_capability(major, minor),
                "total_memory_bytes": int(props.total_memory),
                "multiprocessors": int(props.multi_processor_count),
                "max_threads_per_block": int(getattr(props, "max_threads_per_block", 1024)),
                "shared_memory_per_block_bytes": int(props.shared_memory_per_block),
                "shared_memory_per_block_optin_bytes": int(
                    getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
                ),
                "bf16": bool(torch.cuda.is_bf16_supported()),
            }
            if not target_arch:
                report["target_arch"] = report["gpu"]["native_arch"]
    except Exception as exc:
        report["issues"].append(f"CUDA device probe failed: {type(exc).__name__}: {exc}")

    arch = str(report["target_arch"])
    sm90a = arch == "sm_90a"
    report["features"] = {
        "layout_algebra": _has_symbol("cutlass.cute", "make_layout"),
        "tiled_copy": _has_symbol("cutlass.cute", "make_tiled_copy_tv"),
        "predication": _has_symbol("cutlass.cute", "make_identity_tensor"),
        "cp_async": _has_symbol("cutlass.cute", "nvgpu.cpasync"),
        "tma": sm90a and _has_symbol("cutlass.cute", "nvgpu.cpasync.make_tiled_tma_atom"),
        "mbarrier": sm90a and _has_symbol("cutlass.cute", "arch.mbarrier_init"),
        "clusters": sm90a and _has_symbol("cutlass.cute", "arch.cluster_arrive"),
        "wgmma_bf16": sm90a and _has_symbol("cutlass.cute", "nvgpu.warpgroup.MmaF16BF16Op"),
        "wgmma_fp8": sm90a and _has_symbol("cutlass.cute", "nvgpu.warpgroup.MmaF8Op"),
        "warpgroup_register_control": sm90a
        and _has_symbol("cutlass.cute", "arch.warpgroup_reg_alloc"),
        "jit_compile": _has_symbol("cutlass.cute", "compile"),
        "ir_retention": cutlass_version != "not-installed",
        "cubin_disassembly": bool(report["tools"]["nvdisasm"]["available"]),
        "compute_sanitizer": bool(report["tools"]["compute-sanitizer"]["available"]),
        "nsight_compute": bool(report["tools"]["ncu"]["available"]),
    }
    if arch == "sm_90":
        report["issues"].append("Hopper WGMMA/TMA requires sm_90a; sm_90 omits architecture features.")
    if arch and arch != "sm_90a":
        report["issues"].append(
            "This first harness corpus targets Hopper sm_90a; architecture-specific cards are filtered."
        )
    if not report["gpu"]["available"]:
        report["issues"].append("No CUDA GPU is visible; API lookup works but device validation is unavailable.")
    return report


def probe_capabilities(
    *, device: int = 0, explicit_arch: str = "", arch_list: str = ""
) -> dict[str, Any]:
    """Return a copy so callers cannot mutate the cached environment contract."""
    report = copy.deepcopy(_probe_cached(int(device), str(explicit_arch), str(arch_list)))
    report["fingerprint"] = capability_fingerprint(report)
    return report


def capability_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a full probe to the immutable facts needed beside an evaluation result."""
    packages = report.get("packages", {})
    gpu = report.get("gpu", {})
    cuda = report.get("cuda", {})
    features = report.get("features", {})
    tools = report.get("tools", {})
    python = report.get("python", {})
    identity = {
        "schema_version": 1,
        "dialect": str(report.get("dialect", "cute_dsl_python")),
        "python": str(python.get("version", "")) if isinstance(python, Mapping) else "",
        "nvidia_cutlass_dsl": str(packages.get("nvidia-cutlass-dsl", ""))
        if isinstance(packages, Mapping)
        else "",
        "torch": str(packages.get("torch", "")) if isinstance(packages, Mapping) else "",
        "target_arch": str(report.get("target_arch", "")),
        "gpu": {
            "name": str(gpu.get("name", "")),
            "compute_capability": str(gpu.get("compute_capability", "")),
            "native_arch": str(gpu.get("native_arch", "")),
            "shared_memory_per_block_optin_bytes": int(
                gpu.get("shared_memory_per_block_optin_bytes", 0) or 0
            ),
        }
        if isinstance(gpu, Mapping)
        else {},
        "cuda": {
            "driver": str(cuda.get("driver", "")),
            "torch_runtime": str(cuda.get("torch_runtime", "")),
            "toolkit": str(cuda.get("toolkit", "")),
        }
        if isinstance(cuda, Mapping)
        else {},
        "features": {
            str(name): bool(value) for name, value in features.items()
        }
        if isinstance(features, Mapping)
        else {},
        "tools": {
            str(name): bool(value.get("available", False))
            for name, value in tools.items()
            if isinstance(value, Mapping)
        }
        if isinstance(tools, Mapping)
        else {},
    }
    identity["fingerprint"] = capability_fingerprint(identity)
    return identity


def capability_fingerprint(report: Mapping[str, Any]) -> str:
    """Return a stable short digest for environment comparison and archive records."""
    payload = dict(report)
    payload.pop("fingerprint", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def capability_issues(
    identity: Mapping[str, Any],
    *,
    precision: str,
    required_arch: str,
    required_version: str = "",
) -> list[dict[str, str]]:
    """Describe evaluator capability mismatches without guessing a fallback target."""
    issues: list[dict[str, str]] = []
    target_arch = str(identity.get("target_arch", ""))
    gpu = identity.get("gpu", {})
    native_arch = str(gpu.get("native_arch", "")) if isinstance(gpu, Mapping) else ""
    features = identity.get("features", {})
    installed_version = str(identity.get("nvidia_cutlass_dsl", ""))
    if required_version and installed_version != required_version:
        issues.append(
            {
                "severity": "error",
                "code": "dsl-version-mismatch",
                "message": (
                    f"Evaluator has nvidia-cutlass-dsl {installed_version or 'unknown'}, "
                    f"authoring contract requires {required_version}."
                ),
            }
        )
    if required_arch and target_arch != required_arch:
        issues.append(
            {
                "severity": "error",
                "code": "target-arch-mismatch",
                "message": f"Evaluator targets {target_arch or 'unknown'}, required {required_arch}.",
            }
        )
    if required_arch == "sm_90a" and native_arch and native_arch != "sm_90a":
        issues.append(
            {
                "severity": "error",
                "code": "gpu-arch-mismatch",
                "message": f"Evaluator GPU reports {native_arch}, required Hopper sm_90a.",
            }
        )
    required_feature = "wgmma_fp8" if str(precision).lower() == "fp8" else "wgmma_bf16"
    if required_arch == "sm_90a" and isinstance(features, Mapping) and not bool(features.get(required_feature)):
        issues.append(
            {
                "severity": "error",
                "code": "missing-wgmma-capability",
                "message": f"Evaluator does not expose `{required_feature}` in the installed Python DSL.",
            }
        )
    if installed_version in {"", "not-installed"}:
        issues.append(
            {
                "severity": "error",
                "code": "missing-cute-dsl",
                "message": "Evaluator has no discoverable `nvidia-cutlass-dsl` package.",
            }
        )
    return issues
