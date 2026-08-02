"""Deterministic island scheduling, idea selection, and compressed feedback."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_IDEAS: dict[str, tuple[dict[str, str], ...]] = {
    "triton": (
        {
            "id": "triton-fuse",
            "summary": "Fuse adjacent pointwise/reduction work to remove launches and intermediates.",
        },
        {"id": "triton-tiles", "summary": "Retune block size, warps, and tile shape for the observed tensor geometry."},
        {
            "id": "triton-memory",
            "summary": "Improve coalescing and reuse while reducing redundant global-memory traffic.",
        },
        {
            "id": "triton-specialize",
            "summary": "Specialize a safe fast path for dominant shapes while preserving a fallback.",
        },
    ),
    "cuda_inline": (
        {"id": "cuda-fuse", "summary": "Fuse operations to reduce launches and global-memory round trips."},
        {"id": "cuda-layout", "summary": "Retile thread/block layout for coalesced access and useful reuse."},
        {"id": "cuda-vectorize", "summary": "Use safe aligned vectorized loads/stores with a correct tail path."},
        {"id": "cuda-occupancy", "summary": "Reduce register/shared-memory pressure or tune launch geometry."},
    ),
    "cute": (
        {"id": "cute-fuse", "summary": "Fuse a Torch epilogue into register-resident CuTe fragment compute."},
        {
            "id": "cute-tv-layout",
            "summary": "Retune thread-value layout for vectorization and coalescing.",
            "codegen_contract": "vector",
        },
        {
            "id": "cute-copy",
            "summary": "Improve copy atoms, alignment, and predicated boundary handling.",
            "codegen_contract": "vector",
        },
        {"id": "cute-specialize", "summary": "Add a dominant-shape CuTe fast path without losing general correctness."},
    ),
}


def _cute_mma_ideas(precision: str) -> tuple[dict[str, str], ...]:
    if precision == "fp8":
        mma_summary = (
            "Build or improve a genuine sm_90a FP8 `MmaF8Op`/WGMMA path with Float32 "
            "accumulation; include quantization/prepacking cost and do not substitute scalar casts."
        )
    else:
        mma_summary = (
            "Map GEMM-like work to sm_90a BF16 `MmaF16BF16Op`/WGMMA with Float32 "
            "accumulation and a fused BF16 epilogue."
        )
    return (
        {"id": "cute-wgmma", "summary": mma_summary, "codegen_contract": "hopper_wgmma"},
        {
            "id": "cute-tma-pipeline",
            "summary": "Retune a correct TMA-to-SMEM/WGMMA pipeline, including stage count and barrier bytes.",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
        {
            "id": "cute-smem-layout",
            "summary": "Improve WGMMA-compatible shared-memory swizzle, tile shape, and bank-conflict behavior.",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
        {
            "id": "cute-resource-balance",
            "summary": "Balance warp groups, registers, shared memory, cluster shape, and occupancy without spills.",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
        {
            "id": "cute-fuse",
            "summary": "Fuse the output epilogue into register-resident accumulator fragments.",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
        {
            "id": "cute-specialize",
            "summary": "Add an aligned dominant-shape TMA/WGMMA fast path while preserving a correct fallback.",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
    )


class IslandScheduler:
    def select_profile_idea(
        self,
        *,
        parent_entry: Mapping[str, Any],
        iteration: int,
        island: int,
    ) -> dict[str, str]:
        profile = parent_entry.get("profile", {})
        torch_profile = profile.get("torch", {}) if isinstance(profile, Mapping) else {}
        raw_ideas = (
            torch_profile.get("optimization_ideas", [])
            if isinstance(torch_profile, Mapping)
            else []
        )
        ideas = [str(value).strip() for value in raw_ideas if str(value).strip()]
        if not ideas:
            return {}
        index = (max(1, iteration) - 1 + max(0, island)) % len(ideas)
        return {
            "id": f"profile-guided-{index + 1}",
            "summary": ideas[index],
            "mechanism": "Selected from the compact profile of the actual parent.",
        }

    def select_idea(
        self,
        *,
        backend: str,
        configured_ideas: Sequence[Mapping[str, str]],
        iteration: int,
        island: int,
        islands: int,
        operation: str = "",
        precision: str = "",
        archive: Mapping[str, Any] | None = None,
        allow_agent_ideas: bool = True,
        agent_idea_limit: int = 3,
    ) -> dict[str, str]:
        if configured_ideas:
            ideas = tuple(dict(item) for item in configured_ideas)
        elif backend == "cute" and operation in {"gemm", "attention", "convolution"}:
            ideas = _cute_mma_ideas(precision)
        else:
            ideas = DEFAULT_IDEAS.get(backend, DEFAULT_IDEAS["triton"])
        learned: tuple[dict[str, str], ...] = ()
        manually_steered = any(
            str(idea.get("source", "")) == "manual_steering" for idea in ideas
        )
        if allow_agent_ideas and not manually_steered:
            learned = _agent_authored_ideas(archive or {}, limit=agent_idea_limit)
            if learned:
                ideas = (*learned, *ideas)
        index = ((iteration - 1) * islands + island) % len(ideas)
        capabilities = archive_capabilities(archive or {})
        if (
            backend == "cute"
            and "hopper_wgmma" in capabilities
            and "wgmma_production_output" not in capabilities
        ):
            return {
                "id": "cute-integrate-wgmma-output",
                "summary": (
                    "Replace the measured batch-64 out_proj numerical fixture with the archived "
                    "working FP8 WGMMA executor using the real quantized activation and packed "
                    "production weight. The WGMMA result must directly produce out_proj output; "
                    "preserve a correct fallback for other batches. Expose set_wgmma_enabled(bool) "
                    "and declare production_output in fp8_contract_evidence for evaluator ablation."
                ),
                "codegen_contract": "hopper_wgmma",
                "requires_capability": "hopper_wgmma",
                "produces_capability": "wgmma_production_output",
            }
        production_entry_id = capabilities.get("wgmma_production_output", "")
        production_entry = _archive_entry(archive or {}, production_entry_id)
        production_result = (
            production_entry.get("result", {})
            if isinstance(production_entry, Mapping)
            else {}
        )
        if (
            backend == "cute"
            and production_entry_id
            and isinstance(production_result, Mapping)
            and float(production_result.get("speedup", 0.0) or 0.0) < 0.5
        ):
            return {
                "id": "cute-consolidate-production-wgmma",
                "summary": (
                    "Replace the many per-128x128 block launches and external FP32 accumulation "
                    "with one tiled batch-64 out_proj GEMM over M=64, N=7168, K=8192. Reuse the "
                    "working production-output WGMMA lineage, preserve contribution ablation, FP8 "
                    "scales, correctness fallbacks, and retained WGMMA/TMA/mbarrier artifacts."
                ),
                "codegen_contract": "hopper_wgmma",
                "requires_capability": "wgmma_production_output",
                "produces_capability": "wgmma_production_output",
            }
        seed = (archive or {}).get("seed", {})
        seed_result = seed.get("result", {}) if isinstance(seed, Mapping) else {}
        seed_runtime = (
            float(seed_result.get("runtime_us", 0.0) or 0.0)
            if isinstance(seed_result, Mapping)
            else 0.0
        )
        production_runtime = (
            float(production_result.get("runtime_us", 0.0) or 0.0)
            if isinstance(production_result, Mapping)
            else 0.0
        )
        if (
            backend == "cute"
            and production_entry_id
            and seed_runtime > 0
            and production_runtime > seed_runtime * 1.05
        ):
            return {
                "id": "cute-reduce-quantization-overhead",
                "summary": (
                    "Reduce measured batch-64 preprocessing around the working production WGMMA: "
                    "cache immutable scaled/packed weights in prepare_for_evaluation, eliminate "
                    "redundant activation quantization and temporary module substitution, and keep "
                    "only the dynamic per-token group scaling required by the contract. Preserve "
                    "the exact fallbacks, contribution ablation, and codegen evidence."
                ),
                "codegen_contract": "hopper_wgmma",
                "requires_capability": "wgmma_production_output",
                "produces_capability": "wgmma_production_output",
            }
        if learned and iteration > 1 and iteration % 2 == 0:
            for offset in range(len(learned)):
                learned_idea = learned[(index + offset) % len(learned)]
                required = str(learned_idea.get("requires_capability", ""))
                if not required or required in capabilities:
                    return dict(learned_idea)
        for offset in range(len(ideas)):
            idea = ideas[(index + offset) % len(ideas)]
            required = str(idea.get("requires_capability", ""))
            if not required or required in capabilities:
                return dict(idea)
        return dict(ideas[index])

    def select_baseline_entry(
        self,
        *,
        archive: Mapping[str, Any],
        iteration: int,
        island: int,
        migration_interval: int,
        required_capability: str = "",
    ) -> str:
        def parent_eligible(entry_id: str) -> bool:
            entry = _archive_entry(archive, entry_id)
            return not (
                isinstance(entry, Mapping)
                and int(entry.get("parent_profile_failures", 0) or 0) >= 2
            )

        if required_capability:
            capability_entry = archive_capabilities(archive).get(required_capability, "")
            if capability_entry and parent_eligible(capability_entry):
                return capability_entry
        performance_elites = archive.get("performance_development_elites", {})
        performance = (
            str(performance_elites.get(str(island), "") or "")
            if isinstance(performance_elites, Mapping)
            else ""
        )
        if performance and parent_eligible(performance):
            return performance
        best_development = _best_development_entry(archive, island)
        if best_development and parent_eligible(best_development):
            return best_development
        development_elites = archive.get("development_elites", {})
        development = (
            str(development_elites.get(str(island), "") or "")
            if isinstance(development_elites, Mapping)
            else ""
        )
        if development and parent_eligible(development):
            return development
        island_elites = archive.get("island_elites", {})
        own = str(island_elites.get(str(island), "") or "") if isinstance(island_elites, Mapping) else ""
        global_best = str(archive.get("global_best_id", "") or "")
        migration = migration_interval > 0 and iteration > 1 and iteration % migration_interval == 0
        if migration and global_best and parent_eligible(global_best):
            return global_best
        if own and parent_eligible(own):
            return own
        if global_best and parent_eligible(global_best):
            return global_best
        return "seed"

    def compact_feedback(
        self,
        archive: Mapping[str, Any],
        *,
        island: int,
        parent_entry: Mapping[str, Any] | None = None,
        limit: int = 5,
    ) -> list[str]:
        entries = archive.get("entries", [])
        if not isinstance(entries, list):
            return []
        repaired_entries = {
            str(entry.get("id", "")).split("-repair-", 1)[0]
            for entry in entries
            if isinstance(entry, Mapping)
            and "-repair-" in str(entry.get("id", ""))
            and isinstance(entry.get("result"), Mapping)
            and bool(entry["result"].get("valid"))
        }
        feedback: list[str] = []
        if isinstance(parent_entry, Mapping):
            parent_id = str(parent_entry.get("id", "parent"))
            profile = str(parent_entry.get("profile_summary", "") or "")
            status = str(parent_entry.get("profile_status", "") or "unknown")
            if profile:
                feedback.append(
                    f"Selected parent `{parent_id}` profile ({status}): "
                    + " ".join(profile.split())[:900]
                )
            else:
                feedback.append(
                    f"Selected parent `{parent_id}` has no usable profile; do not assume its bottleneck."
                )
        for entry in reversed(entries):
            if not isinstance(entry, Mapping):
                continue
            result = entry.get("result", {})
            if not isinstance(result, Mapping):
                continue
            if str(entry.get("id", "")) in repaired_entries and not bool(result.get("valid")):
                continue
            promoted = bool(entry.get("promoted"))
            same_island = int(entry.get("island", -1)) == island
            if promoted and not same_island:
                continue
            idea = entry.get("idea", {})
            idea_summary = str(idea.get("summary", "candidate")) if isinstance(idea, Mapping) else "candidate"
            profile_failures = int(entry.get("parent_profile_failures", 0) or 0)
            if profile_failures >= 2:
                submission = entry.get("submission", {})
                mechanism = (
                    str(submission.get("idea_summary", "")).strip()
                    if isinstance(submission, Mapping)
                    else ""
                )
                feedback.append(
                    "Quarantined after repeated sustained-profile failures: "
                    f"`{mechanism or idea_summary}`. Do not repeat this mechanism; choose a "
                    "different operation boundary or implementation strategy."
                )
                if len(feedback) >= limit:
                    break
                continue
            if not bool(result.get("valid")):
                # An evaluator that reports no text leaves `error` present but
                # empty, which a `.get` default does not catch.
                reason = str(result.get("error") or "invalid or incorrect")
                status = str(result.get("status", ""))
                lowered = reason.lower()
                if any(token in lowered for token in ("filenotfound", "no such file", "importerror")):
                    category = "shared packaging/import failure"
                    action = "preserve the idea; fix relocation/import preflight before optimization"
                elif not bool(result.get("compiled", False)):
                    category = "compile failure"
                    action = "rewrite the candidate, correcting the reported error"
                elif status in {"invalid_compliance", "invalid_codegen"}:
                    category = status.replace("_", " ")
                    action = "repair the reported evidence contract before changing hypotheses"
                elif not bool(result.get("correctness", False)):
                    category = "correctness failure"
                    action = "repair the smallest reported shape or numerical mismatch"
                else:
                    category = "invalid candidate"
                    action = "do not repeat it unchanged"
                feedback.append(
                    f"{category}: `{idea_summary}` — {reason[:360]}. Next: {action}."
                )
            elif not promoted:
                feedback.append(
                    f"`{idea_summary}` passed but was not promoted (speedup {float(result.get('speedup', 0.0)):.3f}x)."
                )
            elif same_island:
                feedback.append(
                    f"Current island elite came from `{idea_summary}` at {float(result.get('speedup', 0.0)):.3f}x."
                )
            profile = entry.get("profile_summary")
            if isinstance(profile, str) and profile.strip():
                feedback.append("Profiler: " + " ".join(profile.split())[:300])
            if len(feedback) >= limit:
                break
        return feedback[:limit]


def archive_capabilities(archive: Mapping[str, Any]) -> dict[str, str]:
    """Return latest valid archive entries that proved explicit codegen capabilities."""
    found: dict[str, str] = {}
    entries = archive.get("entries", [])
    if not isinstance(entries, list):
        return found
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        result = entry.get("result", {})
        context = entry.get("cute_context", {})
        evidence = entry.get("cute_evidence", {})
        if not isinstance(result, Mapping) or not bool(result.get("valid")):
            continue
        if not isinstance(context, Mapping) or not isinstance(evidence, Mapping):
            continue
        gate = evidence.get("codegen_gate", {})
        capability = str(context.get("idea_codegen_contract", ""))
        if capability and isinstance(gate, Mapping) and bool(gate.get("passed")):
            _record_best_capability(found, capability, entry, entries)
        produced = str(context.get("idea_produces_capability", ""))
        result_metadata = result.get("metadata", {})
        contribution = (
            result_metadata.get("production_contribution", {})
            if isinstance(result_metadata, Mapping)
            else {}
        )
        if (
            (produced or capability == "hopper_wgmma")
            and isinstance(contribution, Mapping)
            and bool(contribution.get("passed"))
        ):
            _record_best_capability(
                found, produced or "wgmma_production_output", entry, entries
            )
            _record_best_capability(found, "wgmma_production_output", entry, entries)
    return found


def _agent_authored_ideas(
    archive: Mapping[str, Any], *, limit: int
) -> tuple[dict[str, str], ...]:
    """Recover bounded profile-derived hypotheses written by prior authors."""
    entries = archive.get("entries", [])
    if not isinstance(entries, list):
        return ()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in reversed(entries):
        if not isinstance(entry, Mapping):
            continue
        review = entry.get("profile_review", {})
        submission = entry.get("submission", {})
        raw_ideas = review.get("ideas", []) if isinstance(review, Mapping) else []
        if not raw_ideas and isinstance(submission, Mapping):
            raw_ideas = submission.get("next_ideas", [])
        if isinstance(raw_ideas, str):
            raw_ideas = [raw_ideas]
        if not isinstance(raw_ideas, Sequence):
            continue
        for index, raw in enumerate(raw_ideas):
            if isinstance(raw, str):
                idea = {"summary": raw}
            elif isinstance(raw, Mapping):
                idea = {str(key): str(value) for key, value in raw.items() if value not in (None, "")}
            else:
                continue
            summary = str(idea.get("summary", "")).strip()
            key = summary.lower()
            if not summary or key in seen:
                continue
            seen.add(key)
            idea.setdefault("id", f"agent-{entry.get('id', 'archive')}-{index + 1}")
            idea["source"] = "agent_profile_review"
            found.append(idea)
            if len(found) >= max(1, limit):
                return tuple(found)
    return tuple(found)


def _record_best_capability(
    found: dict[str, str],
    capability: str,
    candidate: Mapping[str, Any],
    entries: list[Any],
) -> None:
    incumbent_id = found.get(capability, "")
    incumbent = next(
        (
            item
            for item in entries
            if isinstance(item, Mapping) and str(item.get("id", "")) == incumbent_id
        ),
        None,
    )
    if _capability_rank(candidate) > _capability_rank(incumbent):
        found[capability] = str(candidate.get("id", ""))


def _capability_rank(entry: Mapping[str, Any] | None) -> tuple[float, float, int]:
    if not isinstance(entry, Mapping):
        return (float("-inf"), -1.0, -1)
    result = entry.get("result", {})
    if not isinstance(result, Mapping):
        return (float("-inf"), -1.0, -1)
    runtime = float(result.get("runtime_us", 0.0) or 0.0)
    runtime_rank = -runtime if runtime > 0 else float("-inf")
    speedup = float(result.get("speedup", 0.0) or 0.0)
    return (runtime_rank, speedup, int(entry.get("iteration", -1) or -1))


def _archive_entry(archive: Mapping[str, Any], entry_id: str) -> Mapping[str, Any] | None:
    entries = archive.get("entries", [])
    if isinstance(entries, list):
        for entry in reversed(entries):
            if isinstance(entry, Mapping) and str(entry.get("id", "")) == entry_id:
                return entry
    return None


def _best_development_entry(archive: Mapping[str, Any], island: int) -> str:
    entries = archive.get("entries", [])
    if not isinstance(entries, list):
        return ""
    best_id = ""
    best_rank = (float("-inf"), float("-inf"), -1)
    for entry in entries:
        if not isinstance(entry, Mapping) or int(entry.get("island", -1)) != island:
            continue
        result = entry.get("result", {})
        metadata = result.get("metadata", {}) if isinstance(result, Mapping) else {}
        progress = metadata.get("development_progress", {}) if isinstance(metadata, Mapping) else {}
        milestones = progress.get("milestones", {}) if isinstance(progress, Mapping) else {}
        if not (
            isinstance(result, Mapping)
            and bool(result.get("compiled"))
            and bool(result.get("correctness"))
            and isinstance(milestones, Mapping)
            and bool(milestones.get("executor_executed"))
        ):
            continue
        runtime = float(result.get("runtime_us", 0.0) or 0.0)
        rank = (
            -runtime if runtime > 0 else float("-inf"),
            float(progress.get("score", 0.0) or 0.0),
            int(entry.get("iteration", -1) or -1),
        )
        if rank > best_rank:
            best_rank = rank
            best_id = str(entry.get("id", ""))
    return best_id
