"""Structured empirical memory for CuTe DSL experiments."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_REQUIRED = ("task", "hypothesis", "change", "decision")


def record_experiment(database: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        raise ValueError(f"Experiment record is missing: {', '.join(missing)}")
    payload = dict(record)
    payload.setdefault("schema_version", 1)
    payload.setdefault("dialect", "cute_dsl_python")
    if payload["dialect"] != "cute_dsl_python":
        raise ValueError("Experiment database only accepts the Python CuTe DSL dialect")
    payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    path = Path(database).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(encoded + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def query_experiments(
    database: str | Path,
    *,
    task: str = "",
    tag: str = "",
    decision: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    path = Path(database).expanduser().resolve()
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("dialect") != "cute_dsl_python":
            continue
        if task and task.lower() not in str(item.get("task", "")).lower():
            continue
        if decision and decision != str(item.get("decision", "")):
            continue
        if tag and tag not in {str(value) for value in item.get("lesson_tags", [])}:
            continue
        results.append(item)
    return results[-max(1, int(limit)) :]


def record_archive_evaluation(
    database: str | Path,
    *,
    entry: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one barrier-owned archive result as reusable CuTe evidence."""
    result = entry.get("result", {})
    if not isinstance(result, Mapping):
        result = {}
    idea = entry.get("idea", {})
    if not isinstance(idea, Mapping):
        idea = {}
    submission = entry.get("submission", {})
    if not isinstance(submission, Mapping):
        submission = {}
    evidence = entry.get("cute_evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}

    valid = bool(result.get("valid"))
    promoted = bool(entry.get("promoted"))
    decision = "accept" if promoted else ("reject" if valid else "invalid")
    operation = str(context.get("operation", "unknown"))
    precision = str(context.get("precision", "unknown"))
    arch = str(context.get("arch", "unknown"))
    idea_id = str(idea.get("id", "candidate"))
    concepts_value = context.get("concepts", [])
    concepts = (
        [str(value) for value in concepts_value]
        if isinstance(concepts_value, (list, tuple, set))
        else []
    )
    lesson_tags = [
        value
        for value in (arch, precision, operation, idea_id, *concepts)
        if str(value).strip()
    ]

    payload = {
        "task": f"{operation}:{precision}",
        "hypothesis": str(idea.get("summary", "candidate optimization")),
        "change": {
            "idea_id": idea_id,
            "author_summary": str(submission.get("idea_summary", "")),
            "mechanism": str(submission.get("expected_perf_mechanism", "")),
            "candidate_sha256": str(entry.get("sha256", "")),
            "parent_id": str(entry.get("parent_id", "")),
        },
        "correctness": {
            "compiled": bool(result.get("compiled")),
            "passed": bool(result.get("correctness")),
            "valid": valid,
            "error": str(result.get("error", ""))[:1_000],
        },
        "performance": {
            "runtime_us": result.get("runtime_us"),
            "reference_runtime_us": result.get("ref_runtime_us"),
            "speedup": float(result.get("speedup", 0.0) or 0.0),
        },
        "profile": {
            "summary": str(entry.get("profile_summary", ""))[:2_000],
        },
        "evidence": _compact_evidence(evidence),
        "decision": decision,
        "lesson_tags": list(dict.fromkeys(str(value) for value in lesson_tags)),
        "archive": {
            "entry_id": str(entry.get("id", "")),
            "iteration": entry.get("iteration"),
            "island": entry.get("island"),
        },
    }
    return record_experiment(database, payload)


def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    source_lint = evidence.get("source_lint", {})
    if isinstance(source_lint, Mapping):
        issues = source_lint.get("issues", [])
        compact["source_lint"] = {
            "counts": source_lint.get("counts", {}),
            "issues": [
                {
                    key: item.get(key)
                    for key in ("code", "severity", "message", "line")
                    if item.get(key) not in (None, "")
                }
                for item in issues[:8]
                if isinstance(item, Mapping)
            ]
            if isinstance(issues, list)
            else [],
        }
    environment = evidence.get("evaluator_environment", {})
    if isinstance(environment, Mapping) and environment:
        compact["evaluator_environment"] = {
            key: environment.get(key)
            for key in ("fingerprint", "nvidia_cutlass_dsl", "target_arch", "gpu", "cuda")
            if environment.get(key) not in (None, "", {})
        }
    capability = evidence.get("capability_issues", [])
    if isinstance(capability, list) and capability:
        compact["capability_issues"] = capability[:8]
    codegen = evidence.get("codegen", [])
    if isinstance(codegen, list) and codegen:
        compact["codegen"] = [
            {
                key: item.get(key)
                for key in ("kind", "instruction_families", "resources", "warnings", "error")
                if item.get(key) not in (None, "", {}, [])
            }
            for item in codegen[:4]
            if isinstance(item, Mapping)
        ]
    gate = evidence.get("codegen_gate", {})
    if isinstance(gate, Mapping) and gate:
        compact["codegen_gate"] = {
            "passed": bool(gate.get("passed")),
            "failures": gate.get("failures", [])[:8]
            if isinstance(gate.get("failures"), list)
            else [],
        }
    return compact


def compact_experiment_lessons(records: list[Mapping[str, Any]], *, limit: int = 3) -> list[str]:
    """Render bounded causal reminders rather than replaying experiment transcripts."""
    lessons: list[str] = []
    for item in reversed(records):
        hypothesis = " ".join(str(item.get("hypothesis", "candidate")).split())
        decision = str(item.get("decision", "unknown"))
        performance = item.get("performance", {})
        speedup = (
            float(performance.get("speedup", 0.0) or 0.0)
            if isinstance(performance, Mapping)
            else 0.0
        )
        correctness = item.get("correctness", {})
        error = str(correctness.get("error", "")) if isinstance(correctness, Mapping) else ""
        profile = item.get("profile", {})
        profile_summary = str(profile.get("summary", "")) if isinstance(profile, Mapping) else ""
        observation = " ".join((profile_summary or error).split())[:180]
        suffix = f" — {observation}" if observation else ""
        lessons.append(f"- `{decision}` at {speedup:.3f}x: {hypothesis[:180]}{suffix}")
        if len(lessons) >= max(0, int(limit)):
            break
    return lessons
