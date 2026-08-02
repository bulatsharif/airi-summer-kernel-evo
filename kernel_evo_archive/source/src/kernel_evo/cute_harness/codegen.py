"""Compact, question-oriented inspection of PTX, SASS, MLIR, and CUBIN artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


_FAMILIES: dict[str, tuple[str, ...]] = {
    # PTX spells the family WGMMA; Hopper SASS uses QGMMA/HGMMA mnemonics.
    "wgmma": (r"\bWGMMA\.MMA", r"\bwgmma\.mma", r"\b[QH]?GMMA\."),
    "mma_sync": (
        r"(?<![A-Za-z0-9_.])(?:HMMA|IMMA|BMMA|MMA)(?:\.SYNC)?\.",
        r"\bmma\.sync",
    ),
    "tma": (r"CP_ASYNC_BULK_TENSOR", r"cp\.async\.bulk\.tensor", r"UTMALDG", r"UTMASTG"),
    "cp_async": (r"\bCP_ASYNC\b", r"\bcp\.async\b"),
    "mbarrier": (r"MBARRIER", r"mbarrier"),
    "cluster": (r"CLUSTER", r"cluster"),
    "vector_global_load_128": (r"\bLDG(?:\.E)?\.128\b", r"ld\.global[^\n]*\.v4\."),
    "vector_global_store_128": (r"\bSTG(?:\.E)?\.128\b", r"st\.global[^\n]*\.v4\."),
    "scalar_global_load": (r"\bLDG(?:\.E)?(?:\.SYS)?\b(?![^\n]*\.128)", r"\bld\.global\b(?![^\n]*\.v[248])"),
    "scalar_global_load_32": (r"\bLDG(?:\.E)?(?:\.SYS)?\.(?:U32|S32)\b", r"ld\.global[^\n]*\.b32"),
    "local_load": (r"\bLDL\b", r"ld\.local"),
    "local_store": (r"\bSTL\b", r"st\.local"),
    "shared_load": (r"\bLDS\b", r"ld\.shared"),
    "shared_store": (r"\bSTS\b", r"st\.shared"),
}


def inspect_text(text: str, *, expected: Iterable[str] = ()) -> dict[str, Any]:
    counts = {
        family: sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
        for family, patterns in _FAMILIES.items()
    }
    required = tuple(dict.fromkeys(str(item).strip().lower() for item in expected if str(item).strip()))
    warnings = [f"Expected instruction family `{name}` was not found." for name in required if counts.get(name, 0) == 0]
    if counts["local_load"] or counts["local_store"]:
        warnings.append("Local-memory instructions were found; inspect register spilling.")
    return {
        "instruction_families": counts,
        "expectations": {name: counts.get(name, 0) > 0 for name in required},
        "warnings": warnings,
    }


def _run_tool(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def inspect_artifact(
    path: str | Path,
    *,
    expected: Iterable[str] = (),
    timeout: float = 30.0,
) -> dict[str, Any]:
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    suffix = artifact.suffix.lower()
    resource_output = ""
    if suffix in {".cubin", ".fatbin"}:
        nvdisasm = shutil.which("nvdisasm")
        if not nvdisasm:
            raise RuntimeError("nvdisasm is required to inspect CUBIN/SASS")
        completed = _run_tool([nvdisasm, str(artifact)], timeout=timeout)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4_000:])
        text = completed.stdout
        cuobjdump = shutil.which("cuobjdump")
        if cuobjdump:
            resources = _run_tool([cuobjdump, "--dump-resource-usage", str(artifact)], timeout=timeout)
            resource_output = (resources.stdout or resources.stderr).strip()[-8_000:]
        kind = "cubin"
    else:
        text = artifact.read_text(encoding="utf-8", errors="replace")
        kind = suffix.lstrip(".") or "text"
    result = inspect_text(text, expected=expected)
    resources: dict[str, int] = {}
    for key, pattern in {
        "registers": r"\bREG:(\d+)",
        "stack_bytes": r"\bSTACK:(\d+)",
        "static_shared_bytes": r"\bSHARED:(\d+)",
        "local_bytes": r"\bLOCAL:(\d+)",
    }.items():
        if match := re.search(pattern, resource_output):
            resources[key] = int(match.group(1))
    result.update(
        {
            "path": str(artifact),
            "kind": kind,
            "bytes": artifact.stat().st_size,
            "resource_usage": resource_output,
            "resources": resources,
        }
    )
    return result


def load_codegen_expectations(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load an example's machine-readable code-generation contract."""
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a YAML/JSON object in {path}")
    if isinstance(payload.get("expected_codegen"), Mapping):
        payload = payload["expected_codegen"]
    return dict(payload)


def verify_codegen(
    report: Mapping[str, Any],
    expectations: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Turn advisory instruction/resource metadata into an explicit evidence gate."""
    contract = load_codegen_expectations(expectations)
    families = report.get("instruction_families", {})
    resources = report.get("resources", {})
    if not isinstance(families, Mapping):
        families = {}
    if not isinstance(resources, Mapping):
        resources = {}

    required = tuple(str(value) for value in contract.get("required_instruction_families", ()))
    forbidden = tuple(str(value) for value in contract.get("forbidden_instruction_families", ()))
    checks: list[dict[str, Any]] = []
    for family in required:
        count = int(families.get(family, 0) or 0)
        checks.append(
            {
                "kind": "required_instruction_family",
                "name": family,
                "observed": count,
                "passed": count > 0,
                "message": f"Required `{family}` {'found' if count > 0 else 'missing'}.",
            }
        )
    for family in forbidden:
        count = int(families.get(family, 0) or 0)
        checks.append(
            {
                "kind": "forbidden_instruction_family",
                "name": family,
                "observed": count,
                "passed": count == 0,
                "message": f"Forbidden `{family}` count is {count}.",
            }
        )

    aliases = {
        "local_load": "local_load",
        "local_store": "local_store",
        "registers_per_thread": "registers",
        "shared_memory_bytes": "static_shared_bytes",
    }
    expected_resources = contract.get("resource_expectations", {})
    if isinstance(expected_resources, Mapping):
        for name, expected in expected_resources.items():
            family_name = aliases.get(str(name), str(name))
            source = families if family_name in families else resources
            observed = source.get(family_name)
            passed = observed is not None and int(observed) == int(expected)
            checks.append(
                {
                    "kind": "resource_expectation",
                    "name": str(name),
                    "observed": observed,
                    "expected": expected,
                    "passed": passed,
                    "message": f"Expected `{name}`={expected}; observed {observed!r}.",
                }
            )

    resource_limits = contract.get("resource_limits", {})
    if isinstance(resource_limits, Mapping):
        for name, maximum in resource_limits.items():
            resource_name = aliases.get(str(name), str(name))
            observed = resources.get(resource_name)
            passed = observed is not None and int(observed) <= int(maximum)
            checks.append(
                {
                    "kind": "resource_limit",
                    "name": str(name),
                    "observed": observed,
                    "maximum": maximum,
                    "passed": passed,
                    "message": f"Expected `{name}` <= {maximum}; observed {observed!r}.",
                }
            )

    requested_resources = tuple(str(value) for value in contract.get("resource_checks", ()))
    observations: list[str] = []
    for name in requested_resources:
        resource_name = aliases.get(name, name)
        if resource_name in families or resource_name in resources:
            continue
        observations.append(f"Requested resource `{name}` was not available in this artifact report.")

    failures = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "observations": observations,
    }


def verify_codegen_reports(
    reports: Iterable[Mapping[str, Any]],
    expectations: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Select evidence from the artifact that best satisfies one contract."""
    rank = {"cubin": 4, "sass": 3, "ptx": 2, "mlir": 1}
    evaluated: list[tuple[int, int, dict[str, Any], Mapping[str, Any]]] = []
    for report in reports:
        if report.get("error"):
            continue
        verification = verify_codegen(report, expectations)
        failures = verification.get("failures", [])
        failure_count = len(failures) if isinstance(failures, list) else 999
        evaluated.append(
            (
                0 if verification.get("passed") else 1,
                failure_count,
                verification,
                report,
            )
        )
    if not evaluated:
        return {
            "schema_version": 1,
            "dialect": "cute_dsl_python",
            "passed": False,
            "checks": [],
            "failures": [{"message": "No inspectable code-generation artifact was retained."}],
            "observations": [],
            "artifact": {},
        }
    evaluated.sort(
        key=lambda item: (
            item[0],
            item[1],
            -rank.get(str(item[3].get("kind", "")), 0),
            str(item[3].get("path", "")),
        )
    )
    _, _, selected, report = evaluated[0]
    selected = dict(selected)
    selected["artifact"] = {
        "path": str(report.get("path", "")),
        "kind": str(report.get("kind", "")),
    }
    selected["artifacts_checked"] = len(evaluated)
    return selected


def inspect_and_verify_artifact(
    path: str | Path,
    expectations: str | Path | Mapping[str, Any],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Inspect once, then attach a contract result suitable for evaluator evidence."""
    contract = load_codegen_expectations(expectations)
    report = inspect_artifact(
        path,
        expected=contract.get("required_instruction_families", ()),
        timeout=timeout,
    )
    report["verification"] = verify_codegen(report, contract)
    return report
