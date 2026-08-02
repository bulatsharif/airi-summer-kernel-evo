"""Archive updates and elite promotion for barrier-evaluated candidates."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping

from kernel_evo.agent.store import utc_now


class IdeaStore:
    """Mutate the archive only after every island reaches the evaluation barrier."""

    def archive_iteration(
        self,
        state: dict[str, Any],
        run_dir: Path,
        iteration: int,
    ) -> list[dict[str, Any]]:
        archive = state["archive"]
        record = state["iterations"][str(iteration)]
        added: list[dict[str, Any]] = []
        for island in range(int(state["island_count"])):
            island_record = record["islands"][str(island)]
            if island_record.get("archive_entry_id"):
                continue
            repair_count = int(island_record.get("repair_count", 0) or 0)
            entry_id = f"iter-{iteration:03d}-island-{island}"
            if repair_count:
                entry_id += f"-repair-{repair_count}"
            candidate_path = _run_path(run_dir, str(island_record["candidate_path"]))
            snapshot_path = run_dir / "archive" / entry_id / candidate_path.name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, snapshot_path)
            entry: dict[str, Any] = {
                "id": entry_id,
                "iteration": iteration,
                "island": island,
                "path": str(snapshot_path.relative_to(run_dir)),
                "sha256": _sha256(snapshot_path),
                "parent_id": island_record["baseline_entry_id"],
                "idea": island_record["idea"],
                "submission": island_record.get("submission", {}),
                "result": island_record["result"],
                "profile_summary": island_record.get("profile_summary", ""),
                "profile": island_record.get("profile", {}),
                "profile_status": island_record.get("profile_status", "not_selected"),
                "profile_reason": island_record.get("profile_reason", ""),
                "cute_context": island_record.get("cute_context", {}),
                "cute_evidence": island_record.get("cute_evidence", {}),
                "created_at": utc_now(),
                "promoted": False,
            }
            incumbent_id = str(archive["island_elites"].get(str(island), "") or "")
            incumbent = find_entry(archive, incumbent_id)
            if beats(entry, incumbent):
                archive["island_elites"][str(island)] = entry_id
                entry["promoted"] = True
                island_record["promoted"] = True
            archive["entries"].append(entry)
            island_record["archive_entry_id"] = entry_id
            added.append(entry)

            progress = _development_progress(entry)
            if progress is not None:
                development = archive.setdefault("development_elites", {})
                incumbent_progress_id = str(development.get(str(island), "") or "")
                incumbent_progress = find_entry(archive, incumbent_progress_id)
                if _development_score(entry) > _development_score(incumbent_progress):
                    development[str(island)] = entry_id

            performance = archive.setdefault("performance_development_elites", {})
            performance_incumbent_id = str(performance.get(str(island), "") or "")
            performance_incumbent = find_entry(archive, performance_incumbent_id)
            if performance_incumbent is None:
                performance_incumbent = find_entry(
                    archive, str(archive.get("global_best_id", "") or "")
                )
            if developmental_beats(entry, performance_incumbent):
                performance[str(island)] = entry_id

            global_best = find_entry(archive, str(archive.get("global_best_id", "")))
            if beats(entry, global_best):
                archive["global_best_id"] = entry_id
        return added


def find_entry(archive: Mapping[str, Any], entry_id: str) -> Mapping[str, Any] | None:
    if not entry_id:
        return None
    seed = archive.get("seed")
    if isinstance(seed, Mapping) and seed.get("id") == entry_id:
        return seed
    entries = archive.get("entries", [])
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("id") == entry_id:
                return entry
    return None


def compact_entry(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "id": entry.get("id"),
        "iteration": entry.get("iteration"),
        "island": entry.get("island"),
        "path": entry.get("path"),
        "result": entry.get("result"),
    }


def beats(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    result = candidate.get("result", {})
    if not isinstance(result, Mapping) or not bool(result.get("valid")):
        return False
    if incumbent is None:
        return True
    if int(incumbent.get("parent_profile_failures", 0) or 0) >= 2:
        return True
    incumbent_result = incumbent.get("result")
    if not isinstance(incumbent_result, Mapping) or not bool(incumbent_result.get("valid")):
        return True
    candidate_fitness = float(result.get("fitness", result.get("speedup", 0.0)) or 0.0)
    incumbent_fitness = float(
        incumbent_result.get("fitness", incumbent_result.get("speedup", 0.0)) or 0.0
    )
    return candidate_fitness > incumbent_fitness


def developmental_beats(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None
) -> bool:
    """Rank valid incremental progress without weakening production promotion."""
    result = candidate.get("result", {})
    if not (
        isinstance(result, Mapping)
        and bool(result.get("compiled"))
        and bool(result.get("correctness"))
        and bool(result.get("valid"))
    ):
        return False
    if incumbent is None:
        return True
    incumbent_result = incumbent.get("result", {})
    if not isinstance(incumbent_result, Mapping) or not bool(incumbent_result.get("valid")):
        return True
    candidate_speedup = float(result.get("speedup", 0.0) or 0.0)
    incumbent_speedup = float(incumbent_result.get("speedup", 0.0) or 0.0)
    if candidate_speedup != incumbent_speedup:
        return candidate_speedup > incumbent_speedup
    candidate_runtime = float(result.get("runtime_us", 0.0) or 0.0)
    incumbent_runtime = float(incumbent_result.get("runtime_us", 0.0) or 0.0)
    return candidate_runtime > 0 and (
        incumbent_runtime <= 0 or candidate_runtime < incumbent_runtime
    )


def _development_progress(entry: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    result = entry.get("result", {})
    if not isinstance(result, Mapping):
        return None
    metadata = result.get("metadata", {})
    progress = metadata.get("development_progress", {}) if isinstance(metadata, Mapping) else {}
    milestones = progress.get("milestones", {}) if isinstance(progress, Mapping) else {}
    if (
        isinstance(progress, Mapping)
        and isinstance(milestones, Mapping)
        and bool(result.get("compiled"))
        and bool(result.get("correctness"))
        and bool(milestones.get("executor_executed"))
    ):
        return progress
    return None


def _development_score(entry: Mapping[str, Any] | None) -> float:
    progress = _development_progress(entry)
    return float(progress.get("score", 0.0) or 0.0) if progress else 0.0


def _run_path(run_dir: Path, value: str) -> Path:
    path = (run_dir / value).resolve()
    path.relative_to(run_dir.resolve())
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
