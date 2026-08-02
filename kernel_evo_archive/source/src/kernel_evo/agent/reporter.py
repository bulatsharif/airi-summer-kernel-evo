"""Deterministic iteration reports; coordinators never need raw evaluator logs."""

from __future__ import annotations

from typing import Any, Mapping

from kernel_evo.agent.idea_store import find_entry
from kernel_evo.agent.models import is_repairable_result


def iteration_report(state: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    iterations = state.get("iterations", {})
    record = iterations.get(str(iteration), {}) if isinstance(iterations, Mapping) else {}
    islands = record.get("islands", {}) if isinstance(record, Mapping) else {}
    rows: list[dict[str, Any]] = []
    if isinstance(islands, Mapping):
        for island_key in sorted(islands, key=lambda value: int(value)):
            island = islands[island_key]
            if not isinstance(island, Mapping):
                continue
            result = island.get("result", {})
            if not isinstance(result, Mapping):
                result = {}
            idea = island.get("idea", {})
            metadata = result.get("metadata", {})
            progress = (
                metadata.get("development_progress", {})
                if isinstance(metadata, Mapping)
                else {}
            )
            milestones = progress.get("milestones", {}) if isinstance(progress, Mapping) else {}
            repair_count = int(island.get("repair_count", 0) or 0)
            repair_limit = int(
                state.get("config", {}).get("max_repairs_per_island", 1) or 0
            )
            repairable = is_repairable_result(result, repair_count, repair_limit)
            profile = island.get("profile", {})
            torch_profile = profile.get("torch", {}) if isinstance(profile, Mapping) else {}
            inner = (
                torch_profile.get("inner_kernel", {})
                if isinstance(torch_profile, Mapping)
                else {}
            )
            eager = (
                torch_profile.get("eager_complete_layer", {})
                if isinstance(torch_profile, Mapping)
                else {}
            )
            graph = (
                torch_profile.get("cuda_graph_complete_layer", {})
                if isinstance(torch_profile, Mapping)
                else {}
            )
            rows.append(
                {
                    "island": int(island_key),
                    "status": str(island.get("status", "unknown")),
                    "idea": str(idea.get("summary", "")) if isinstance(idea, Mapping) else "",
                    "compiled": bool(result.get("compiled", False)),
                    "correctness": bool(result.get("correctness", False)),
                    "valid": bool(result.get("valid", False)),
                    "runtime_us": result.get("runtime_us"),
                    "speedup": float(result.get("speedup", 0.0) or 0.0),
                    "promoted": bool(island.get("promoted", False)),
                    "error": str(result.get("error", "") or ""),
                    "repairable": repairable,
                    "repair_count": repair_count,
                    "development_score": float(progress.get("score", 0.0) or 0.0)
                    if isinstance(progress, Mapping)
                    else 0.0,
                    "development_milestones": [
                        name for name, passed in milestones.items() if passed
                    ]
                    if isinstance(milestones, Mapping)
                    else [],
                    "profile_status": str(island.get("profile_status", "not_selected")),
                    "profile_reason": str(island.get("profile_reason", "")),
                    "active_device_us": inner.get("active_device_time_us")
                    if isinstance(inner, Mapping)
                    else None,
                    "kernel_count": inner.get("kernel_count") if isinstance(inner, Mapping) else None,
                    "memcpy_count": inner.get("memcpy_count") if isinstance(inner, Mapping) else None,
                    "eager_layer_us": eager.get("end_to_end_us") if isinstance(eager, Mapping) else None,
                    "dispatch_gaps_us": eager.get("inferred_dispatch_gaps_us")
                    if isinstance(eager, Mapping)
                    else None,
                    "graph_capturable": graph.get("capturable") if isinstance(graph, Mapping) else None,
                    "graph_replay_us": graph.get("replay_us") if isinstance(graph, Mapping) else None,
                    "profile_reviewed": bool(island.get("profile_review")),
                }
            )
    archive = state.get("archive", {})
    best = find_entry(archive, str(archive.get("global_best_id", ""))) if isinstance(archive, Mapping) else None
    review_required = bool(
        state.get("config", {}).get("profile_review_required", True)
    )
    return {
        "run_id": state.get("run_id"),
        "iteration": iteration,
        "status": record.get("status", "not_prepared") if isinstance(record, Mapping) else "not_prepared",
        "islands": rows,
        "valid_candidates": sum(1 for row in rows if row["valid"]),
        "promoted_candidates": sum(1 for row in rows if row["promoted"]),
        "global_best": best,
        "repairable_islands": [row["island"] for row in rows if row["repairable"]],
        "pending_profile_reviews": [
            row["island"]
            for row in rows
            if review_required
            and row["profile_status"] == "completed"
            and not row["profile_reviewed"]
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# KernelEvo iteration {report['iteration']}",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Barrier status: **{report['status']}**  ",
        f"Valid: {report['valid_candidates']} · Promoted: {report['promoted_candidates']}",
        "",
        "| Island | Result | Progress | Runtime | Speedup | Promoted | Idea |",
        "|---:|---|---:|---:|---:|:---:|---|",
    ]
    for row in report.get("islands", []):
        result = "valid" if row["valid"] else row["status"]
        if row["error"]:
            result = f"failed: {_single_line(row['error'], 80)}"
        runtime = f"{row['runtime_us']:.3f} µs" if isinstance(row["runtime_us"], (int, float)) else "—"
        progress = (
            f"{row['development_score']:.2f} "
            f"({_single_line(', '.join(row['development_milestones']), 52)})"
            if row.get("development_score", 0.0) > 0
            else "—"
        )
        lines.append(
            f"| {row['island']} | {result} | {progress} | {runtime} | {row['speedup']:.3f}x | "
            f"{'yes' if row['promoted'] else 'no'} | {_single_line(row['idea'], 100)} |"
        )
    best = report.get("global_best")
    if isinstance(best, Mapping):
        result = best.get("result", {})
        speedup = float(result.get("speedup", 0.0)) if isinstance(result, Mapping) else 0.0
        lines.extend(
            [
                "",
                f"Global best: `{best.get('id')}` at **{speedup:.3f}x** speedup.",
            ]
        )
    profiled = [row for row in report.get("islands", []) if row.get("profile_reason")]
    if profiled:
        lines.extend(["", "## Compact profile objectives", ""])
        for row in profiled:
            lines.append(
                f"- Island {row['island']} ({row['profile_reason']}): "
                f"inner device {_metric(row.get('active_device_us'))} µs, "
                f"{_metric(row.get('kernel_count'))} kernels, "
                f"{_metric(row.get('memcpy_count'))} memcpys; "
                f"eager layer {_metric(row.get('eager_layer_us'))} µs, "
                f"dispatch gaps {_metric(row.get('dispatch_gaps_us'))} µs; "
                f"graph {'yes' if row.get('graph_capturable') else 'no'}, "
                f"replay {_metric(row.get('graph_replay_us'))} µs."
            )
    repairable = report.get("repairable_islands", [])
    if repairable:
        lines.extend(
            [
                "",
                "Bounded repair available for island(s): "
                + ", ".join(str(value) for value in repairable)
                + ". Use `kernel-evo island repair` before advancing.",
            ]
        )
    pending_reviews = report.get("pending_profile_reviews", [])
    if pending_reviews:
        lines.extend(
            [
                "",
                "Compact profile review required for island(s): "
                + ", ".join(str(value) for value in pending_reviews)
                + ". Use `kernel-evo iter review-profiles`, then `kernel-evo island review-submit`.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _single_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _metric(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "—"
