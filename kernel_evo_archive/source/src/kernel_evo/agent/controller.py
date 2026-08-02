"""Deterministic controller for visible, barrier-synchronized kernel evolution."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from kernel_evo.agent.config import AgentRunConfig, load_agent_config
from kernel_evo.agent.errors import ConfigurationError, InvalidTransitionError
from kernel_evo.agent.evaluator import CandidateEvaluator, coerce_evaluation_result, evaluator_from_config
from kernel_evo.agent.idea_store import IdeaStore, compact_entry, find_entry
from kernel_evo.agent.models import (
    AuthoringTask,
    EvaluationContext,
    EvaluationResult,
    IslandStatus,
    RunPhase,
    is_repairable_result,
)
from kernel_evo.agent.packet_builder import build_authoring_packet
from kernel_evo.agent.profiler import (
    CandidateProfiler,
    KernelBenchProfiler,
    ProfileResult,
    coerce_profile_result,
    profile_result_status,
)
from kernel_evo.agent.reporter import iteration_report, render_markdown
from kernel_evo.agent.scheduler import IslandScheduler, archive_capabilities
from kernel_evo.agent.store import RunStore, utc_now
from kernel_evo.agent.tracker import EventTracker
from kernel_evo.agent.workspaces import backend_rules, resolve_baseline, summarize_tests
from kernel_evo.core.problem import (
    build_validation_config,
    load_problem_sources,
    write_problem_artifacts,
)
from kernel_evo.cute_harness.ablation import materialize_bundle
from kernel_evo.cute_harness.b300 import baseline_candidate, load_task, starter_candidate
from kernel_evo.resources.paths import get_resources_dir
from kernel_evo.resources.workspace import prepare_problem_workspace


B300_TRACE_REASON = "b300_trace"
EvaluatorLike = CandidateEvaluator | Callable[[EvaluationContext], EvaluationResult | Mapping[str, Any]]
ProfilerLike = CandidateProfiler | Callable[[EvaluationContext, EvaluationResult], ProfileResult | str]


class KernelEvoAgent:
    """Direct API used by Codex/Claude coordinators and the nested CLI commands."""

    def __init__(
        self,
        runs_dir: str | Path = ".kernelevo/runs",
        *,
        evaluator: EvaluatorLike | None = None,
        profiler: ProfilerLike | None = None,
        scheduler: IslandScheduler | None = None,
        idea_store: IdeaStore | None = None,
    ) -> None:
        self.store = RunStore(runs_dir)
        self._evaluator = evaluator
        self._profiler = profiler
        self.scheduler = scheduler or IslandScheduler()
        self.idea_store = idea_store or IdeaStore()

    @property
    def runs_dir(self) -> Path:
        return self.store.runs_dir

    def init_run(
        self,
        config: AgentRunConfig | Mapping[str, Any] | str | Path,
        *,
        run_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = load_agent_config(config, overrides=overrides)
        effective_run_id = run_id or _new_run_id(cfg.name)
        run_dir = self.store.create(effective_run_id)
        (run_dir / "seed").mkdir(parents=True, exist_ok=True)

        problem_dir: Path | None = None
        run_config: dict[str, Any] = {}
        generated_seed: Path | None = None
        cute_b300_task = None
        if cfg.evaluator_kind == "cute_b300":
            cute_b300_task = load_task(cfg.problem_path)
        elif cfg.problem_path or cfg.level is not None:
            sources = load_problem_sources(
                problem_path=cfg.problem_path,
                level=cfg.level,
                problem_id=cfg.problem_id,
                dataset_src=cfg.dataset_src,
                dataset_name=cfg.dataset_name,
                backend=cfg.backend,
            )
            workspace = prepare_problem_workspace(
                resources_dir=get_resources_dir(),
                workspace_root=run_dir / "problem",
            )
            problem_dir = workspace.root_dir
            options = cfg.to_dict()
            options.update(
                {
                    "profile_stage_enabled": cfg.profile_enabled,
                    "profile_ncu_min_speedup": cfg.profile_min_speedup,
                    "profile_artifacts_dir": str(run_dir / "artifacts"),
                    "validator_debug_dir": str(run_dir / "validate_logs"),
                }
            )
            run_config = build_validation_config(
                sources=sources,
                problem_dir=problem_dir,
                experiment_dir=run_dir,
                options=options,
            )
            generated_seed = write_problem_artifacts(
                workspace=workspace,
                sources=sources,
                run_cfg=run_config,
                seed_note=f"Generated for KernelEvo-Agent run {effective_run_id}",
            )

        if cfg.baseline:
            baseline_source = resolve_baseline(cfg.baseline, candidate_name=cfg.candidate_name)
        elif cute_b300_task is not None:
            baseline_source = run_dir / "b300_task_baseline.py"
            baseline_source.write_text(
                (
                    starter_candidate(cute_b300_task)
                    if cfg.b300_seed == "starter"
                    else baseline_candidate(cute_b300_task)
                ),
                encoding="utf-8",
            )
        elif generated_seed is not None:
            baseline_source = generated_seed
        else:
            raise ConfigurationError("No baseline was prepared")

        candidate_name = cfg.candidate_name or (
            "submission.py" if cute_b300_task is not None else baseline_source.name
        )
        if Path(candidate_name).name != candidate_name:
            raise ConfigurationError("candidate_name must be a filename, not a path")
        seed_path = run_dir / "seed" / candidate_name
        shutil.copy2(baseline_source, seed_path)

        cute_capability_path: Path | None = None
        cute_experiments_path: Path | None = None
        cute_author_report: dict[str, Any] = {}
        if (
            cfg.backend == "cute"
            and cfg.cute_harness_enabled
            and cfg.evaluator_kind != "cute_b300"
        ):
            from kernel_evo.cute_harness.capabilities import probe_capabilities

            device_text = str(cfg.device)
            try:
                device = int(device_text.rsplit(":", 1)[-1]) if ":" in device_text else 0
            except ValueError:
                device = 0
            cute_capability_path = run_dir / "cute" / "author_capability.json"
            cute_author_report = probe_capabilities(
                device=device,
                explicit_arch=cfg.cute_arch,
                arch_list=cfg.arch_list,
            )
            self.store.write_json(
                cute_capability_path,
                cute_author_report,
            )
            cute_experiments_path = run_dir / "cute" / "experiments.jsonl"

        config_snapshot = cfg.to_dict()
        if cute_author_report:
            packages = cute_author_report.get("packages", {})
            required_version = (
                str(packages.get("nvidia-cutlass-dsl", ""))
                if isinstance(packages, Mapping)
                else ""
            )
            config_snapshot["cute_required_version"] = required_version
            if run_config:
                run_config["cute_required_version"] = required_version
                if problem_dir:
                    self.store.write_json(problem_dir / "run_config.json", run_config)
        seed_result = EvaluationResult(
            compiled=True,
            correctness=True,
            valid=True,
            speedup=1.0,
            fitness=1.0,
            status="baseline",
        )
        if cfg.seed_preflight:
            seed_preflight_dir = run_dir / "seed_preflight"
            seed_preflight_dir.mkdir(parents=True, exist_ok=True)
            seed_context = EvaluationContext(
                run_id=effective_run_id,
                iteration=0,
                island=0,
                run_dir=run_dir,
                island_dir=seed_preflight_dir,
                candidate_path=seed_path,
                baseline_path=seed_path,
                problem_dir=problem_dir,
                config=config_snapshot,
                run_config=run_config,
            )
            evaluator = self._evaluator or evaluator_from_config(config_snapshot)
            try:
                seed_result = coerce_evaluation_result(_invoke_evaluator(evaluator, seed_context))
            except Exception as exc:
                seed_result = EvaluationResult.failed(exc)
            self.store.write_json(
                seed_preflight_dir / "result.json",
                {"result": seed_result.to_dict()},
            )
            if not seed_result.valid:
                raise ConfigurationError(
                    "Packaged seed preflight failed before authoring: "
                    + (seed_result.error or seed_result.status)
                )

        state: dict[str, Any] = {
            "schema_version": 1,
            "run_id": effective_run_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "phase": RunPhase.READY.value,
            "current_iteration": 1,
            "steps": cfg.steps,
            "island_count": cfg.islands,
            "candidate_name": candidate_name,
            "config": config_snapshot,
            "paths": {
                "run_dir": ".",
                "seed": _relative(run_dir, seed_path),
                "problem_dir": _relative(run_dir, problem_dir) if problem_dir else "",
                "run_config": _relative(run_dir, problem_dir / "run_config.json") if problem_dir else "",
                "tests": cfg.tests,
                "cute_author_capability": (
                    _relative(run_dir, cute_capability_path) if cute_capability_path else ""
                ),
                "cute_experiments": (
                    _relative(run_dir, cute_experiments_path) if cute_experiments_path else ""
                ),
            },
            "archive": {
                "seed": {
                    "id": "seed",
                    "path": _relative(run_dir, seed_path),
                    "sha256": _sha256(seed_path),
                    "result": seed_result.to_dict(),
                },
                "entries": [],
                "island_elites": {str(island): "seed" for island in range(cfg.islands)},
                "development_elites": {},
                "performance_development_elites": {},
                "global_best_id": "seed",
            },
            "profiling": {
                "regression_profiles_used": 0,
                "parent_profiles": 0,
                "capability_profiles": 0,
                "capabilities_profiled": [],
            },
            "iterations": {},
        }
        self.store.write_state(effective_run_id, state)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self._tracker(state, run_dir).emit(
            "run_initialized",
            {
                "run_id": effective_run_id,
                "steps": cfg.steps,
                "islands": cfg.islands,
                "backend": cfg.backend,
            },
        )
        if cfg.seed_preflight:
            self._tracker(state, run_dir).emit(
                "seed_preflight_passed",
                {
                    "run_id": effective_run_id,
                    "valid": seed_result.valid,
                    "speedup": seed_result.speedup,
                },
            )
        return self.status(effective_run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        state = self.store.read_state(run_id)
        run_dir = self.store.run_dir(run_id)
        iteration = int(state["current_iteration"])
        record = state.get("iterations", {}).get(str(iteration), {})
        islands: list[dict[str, Any]] = []
        if isinstance(record, Mapping):
            for key, value in sorted(
                record.get("islands", {}).items(), key=lambda item: int(item[0])
            ):
                result = value.get("result", {}) if isinstance(value, Mapping) else {}
                islands.append(
                    {
                        "island": int(key),
                        "status": value.get("status") if isinstance(value, Mapping) else "unknown",
                        "stage": value.get("stage", "") if isinstance(value, Mapping) else "",
                        "stage_started_at": value.get("stage_started_at", "")
                        if isinstance(value, Mapping)
                        else "",
                        "candidate_path": str(
                            _state_path(run_dir, value.get("candidate_path", ""))
                        )
                        if isinstance(value, Mapping) and value.get("candidate_path")
                        else "",
                        "valid": bool(result.get("valid")) if isinstance(result, Mapping) else False,
                        "speedup": float(result.get("speedup", 0.0))
                        if isinstance(result, Mapping)
                        else 0.0,
                    }
                )
        archive = state.get("archive", {})
        best = find_entry(archive, str(archive.get("global_best_id", "")))
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "phase": state["phase"],
            "current_iteration": iteration,
            "steps": int(state["steps"]),
            "islands": islands,
            "island_count": int(state["island_count"]),
            "archive_size": len(archive.get("entries", [])),
            "global_best": compact_entry(best),
            "next_action": _next_action(state, self.runs_dir),
        }

    def extend_run(
        self,
        run_id: str,
        additional_steps: int,
        *,
        ideas: Sequence[Mapping[str, Any]] | None = None,
        author_readable_files: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Add future barriers without rewriting archive or completed iteration state."""
        if additional_steps < 1:
            raise ConfigurationError("additional_steps must be at least 1")
        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            phase = RunPhase(state["phase"])
            if phase not in {RunPhase.EVALUATED, RunPhase.COMPLETE}:
                raise InvalidTransitionError(
                    "A run can only be extended after an evaluated barrier or completion"
                )
            state["steps"] = int(state["steps"]) + int(additional_steps)
            state["config"]["steps"] = state["steps"]
            if ideas is not None:
                state["config"]["ideas"] = [dict(idea) for idea in ideas]
            if author_readable_files:
                existing = [
                    str(value) for value in state["config"].get("author_readable_files", [])
                ]
                state["config"]["author_readable_files"] = list(
                    dict.fromkeys((*existing, *(str(value) for value in author_readable_files)))
                )
            if phase is RunPhase.COMPLETE:
                state["phase"] = RunPhase.EVALUATED.value
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            "run_extended",
            {
                "run_id": run_id,
                "additional_steps": additional_steps,
                "steps": state["steps"],
                "steered": ideas is not None,
            },
        )
        return self.status(run_id)

    def prepare_iteration(
        self,
        run_id: str,
        iteration: int | None = None,
        *,
        documentation_enabled: bool | None = None,
        documentation_tier: str | None = None,
    ) -> list[AuthoringTask]:
        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            current = int(state["current_iteration"])
            target = current if iteration is None else int(iteration)
            if target != current:
                raise InvalidTransitionError(
                    f"Can only prepare current iteration {current}; requested {target}."
                )
            phase = RunPhase(state["phase"])
            if phase is RunPhase.AUTHORING:
                return self._load_authoring_tasks(state, run_dir, target)
            if phase is not RunPhase.READY:
                raise InvalidTransitionError(
                    f"Cannot prepare iteration while run phase is {phase.value}; "
                    f"{_next_action(state, self.runs_dir)}"
                )

            config = state["config"]
            if documentation_enabled is not None:
                config["documentation_enabled"] = documentation_enabled
            if documentation_tier is not None:
                config["documentation_tier"] = documentation_tier
                config["documentation_enabled"] = documentation_tier != "bare"
            archive = state["archive"]
            tier = str(config.get("documentation_tier", "errors"))
            documentation_enabled = (
                bool(config.get("documentation_enabled", True)) and tier != "bare"
            )
            config["documentation_enabled"] = documentation_enabled
            tier = tier if documentation_enabled else "bare"
            delivery = str(config.get("documentation_delivery", "files"))
            airi_task = (
                load_task(str(config.get("problem_path", "")))
                if str(config.get("evaluator_kind", "")) == "cute_b300"
                else None
            )
            airi_reading_rule = (
                f"- Read the listed `{tier}` documentation bundle before editing.\n"
                if documentation_enabled
                else "- Documentation is disabled; use only the task statement and supplied candidate.\n"
            )
            rules = (
                "# CuTe B300 authoring rules\n\n"
                "- Preserve `ModelNew.forward` and the task JIT entrypoint.\n"
                "- Keep candidate code CuTe-only: no `main()`, PyTorch calls, inputs, oracle, timing, or PASS output.\n"
                f"{airi_reading_rule}"
                f"- Run `kernel-evo cute task-check {airi_task.id} <candidate>` before submission.\n"
                "- Do not run the remote benchmark; KernelEvo owns evaluation and promotion.\n"
                if airi_task is not None
                else backend_rules(str(config["backend"]), str(config.get("rules_file", "")))
            )
            tests_summary = summarize_tests(str(config.get("tests", "")))
            task_source = _reference_source(state, run_dir) if str(config["backend"]) == "cute" else ""
            routing_source = task_source
            if str(config["backend"]) == "cute" and not routing_source:
                seed_entry = archive.get("seed", {})
                if isinstance(seed_entry, Mapping) and seed_entry.get("path"):
                    seed_source = _state_path(run_dir, str(seed_entry["path"]))
                    if seed_source.is_file():
                        routing_source = seed_source.read_text(
                            encoding="utf-8", errors="replace"
                        )[:60_000]
            experiment_database = (
                _state_path(run_dir, str(state["paths"]["cute_experiments"]))
                if state["paths"].get("cute_experiments")
                else None
            )
            island_records: dict[str, Any] = {}
            tasks: list[AuthoringTask] = []
            for island in range(int(state["island_count"])):
                operation = ""
                if str(config["backend"]) == "cute":
                    from kernel_evo.cute_harness.catalog import infer_operation

                    operation = infer_operation(routing_source)
                idea = self.scheduler.select_idea(
                    backend=str(config["backend"]),
                    configured_ideas=config.get("ideas", []),
                    iteration=target,
                    island=island,
                    islands=int(state["island_count"]),
                    operation=operation,
                    precision=str(config.get("precision", "")),
                    archive=archive,
                    allow_agent_ideas=bool(config.get("profile_agent_ideas", True)),
                )
                baseline_entry_id = self.scheduler.select_baseline_entry(
                    archive=archive,
                    iteration=target,
                    island=island,
                    migration_interval=int(config.get("migration_interval", 3)),
                    required_capability=str(idea.get("requires_capability", "")),
                )
                baseline_entry = find_entry(archive, baseline_entry_id)
                if baseline_entry is None:
                    raise InvalidTransitionError(f"Archive entry disappeared: {baseline_entry_id}")
                parent_profile_status = self._ensure_parent_profile(
                    state=state,
                    run_dir=run_dir,
                    entry=baseline_entry,
                    island=island,
                )
                if (
                    bool(config.get("profile_enabled", False))
                    and bool(config.get("profile_parent_before_use", True))
                    and parent_profile_status not in {"completed", "not_required"}
                ):
                    self.store.write_state(run_id, state)
                    raise InvalidTransitionError(
                        f"Selected parent `{baseline_entry_id}` profile is not ready "
                        f"(status={parent_profile_status}); authoring was not opened."
                    )
                if (
                    bool(config.get("profile_agent_ideas", True))
                    and not str(idea.get("requires_capability", ""))
                    and not str(idea.get("produces_capability", ""))
                    and str(idea.get("source", ""))
                    not in {"agent_profile_review", "manual_steering"}
                ):
                    profile_idea = self.scheduler.select_profile_idea(
                        parent_entry=baseline_entry,
                        iteration=target,
                        island=island,
                    )
                    if profile_idea:
                        idea = {**idea, **profile_idea}
                source = _state_path(run_dir, str(baseline_entry["path"]))
                island_dir = run_dir / f"iter_{target:03d}" / f"island_{island}"
                baseline_path = island_dir / "baseline" / str(state["candidate_name"])
                candidate_path = island_dir / "candidate" / str(state["candidate_name"])
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, baseline_path)
                shutil.copy2(source, candidate_path)

                if not operation and str(config["backend"]) == "cute":
                    operation = infer_operation(
                        baseline_path.read_text(encoding="utf-8", errors="replace")[:60_000]
                    )
                feedback = self.scheduler.compact_feedback(
                    archive,
                    island=island,
                    parent_entry=baseline_entry,
                )
                # The critique of the turn that just ran leads, whatever parent the
                # scheduler picked: it is the only feedback about this island's last
                # actual attempt when that attempt failed and was never promoted.
                feedback = [*_critic_feedback(state, island), *feedback]
                supplemental_context = ""
                documentation_prompt = ""
                cute_context_metadata: dict[str, Any] = {}
                configured_readables = (
                    tuple(
                        Path(str(value)).expanduser().resolve()
                        for value in config.get("author_readable_files", ())
                    )
                    if documentation_enabled
                    else ()
                )
                if airi_task is not None:
                    bundle = materialize_bundle(
                        airi_task,
                        tier,
                        run_dir / "documentation" / tier,
                    )
                    # Same bytes either way; only how the author receives them differs.
                    if delivery == "prompt":
                        documentation_prompt = bundle.text
                    else:
                        configured_readables = tuple(
                            dict.fromkeys((*bundle.files, *configured_readables))
                        )
                    cute_context_metadata = {
                        "documentation_tier": bundle.tier,
                        "documentation_delivery": delivery,
                        "documentation_tokens_cl100k": bundle.tokens_cl100k,
                        "documentation_files": [str(path) for path in bundle.files],
                    }
                supplemental_readable_files: tuple[Path, ...] = configured_readables
                if str(config["backend"]) == "cute" and bool(
                    config.get("cute_harness_enabled", True)
                ) and str(config.get("evaluator_kind", "")) != "cute_b300" and documentation_enabled:
                    from kernel_evo.cute_harness import build_agent_context

                    bundle = build_agent_context(
                        config=config,
                        idea=idea,
                        baseline_path=baseline_path,
                        task_source=task_source,
                        experiment_database=experiment_database,
                    )
                    supplemental_context = bundle.text
                    supplemental_readable_files = tuple(
                        dict.fromkeys((*bundle.readable_files, *configured_readables))
                    )
                    cute_context_metadata = bundle.metadata
                task = build_authoring_packet(
                    run_id=run_id,
                    backend=str(config["backend"]),
                    iteration=target,
                    island=island,
                    island_dir=island_dir,
                    baseline_path=baseline_path,
                    candidate_path=candidate_path,
                    idea=idea,
                    feedback=feedback,
                    parent_profile_summary=str(baseline_entry.get("profile_summary", "")),
                    rules=rules,
                    tests_summary=tests_summary,
                    supplemental_context=supplemental_context,
                    supplemental_readable_files=supplemental_readable_files,
                    documentation_prompt=documentation_prompt,
                    compile_check_command=_compile_check_command(
                        config=config,
                        run_dir=run_dir,
                        island_dir=island_dir,
                        candidate_path=candidate_path,
                    ),
                    require_graph_capturable=(
                        bool(config.get("profile_require_graph_capturable", True))
                        and str(config.get("measurement_mode", "wall-clock"))
                        != "device-time"
                    ),
                )
                tasks.append(task)
                island_records[str(island)] = {
                    "status": IslandStatus.AWAITING_AUTHOR.value,
                    "baseline_entry_id": baseline_entry_id,
                    "baseline_path": _relative(run_dir, baseline_path),
                    "candidate_path": _relative(run_dir, candidate_path),
                    "prepared_sha256": _sha256(candidate_path),
                    "task_file": _relative(run_dir, task.task_file),
                    "idea": idea,
                    "feedback": list(feedback),
                    "submission": {},
                    "result": None,
                    "profile_summary": "",
                    "cute_context": cute_context_metadata,
                    "cute_evidence": {},
                    "promoted": False,
                    "repair_count": 0,
                    "failure_history": [],
                }

            state["iterations"][str(target)] = {
                "iteration": target,
                "status": RunPhase.AUTHORING.value,
                "prepared_at": utc_now(),
                "islands": island_records,
            }
            state["phase"] = RunPhase.AUTHORING.value
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            "iteration_prepared",
            {"run_id": run_id, "iteration": target, "islands": len(tasks)},
        )
        return tasks

    def island_context(self, run_id: str, iteration: int, island: int) -> dict[str, Any]:
        state = self.store.read_state(run_id)
        run_dir = self.store.run_dir(run_id)
        island_record = _island_record(state, iteration, island)
        task_path = _state_path(run_dir, str(island_record["task_file"]))
        packet_path = task_path.parent / "packet.json"
        if not packet_path.exists():
            raise InvalidTransitionError(f"Authoring packet is missing: {packet_path}")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["status"] = island_record["status"]
        packet["submission"] = island_record.get("submission", {})
        return packet

    def retarget_island(
        self,
        run_id: str,
        iteration: int,
        island: int,
        idea: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replace an unopened authoring packet idea without touching its parent or archive."""
        run_dir = self.store.run_dir(run_id)
        normalized = {str(key): value for key, value in idea.items()}
        if not str(normalized.get("summary", "")).strip():
            raise ConfigurationError("A retargeted idea requires a non-empty summary")
        normalized.setdefault("id", "manual-retarget")
        normalized.setdefault("source", "manual_steering")
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            _require_current_authoring(state, iteration)
            island_record = _island_record(state, iteration, island)
            if island_record.get("status") != IslandStatus.AWAITING_AUTHOR.value:
                raise InvalidTransitionError(
                    "Only an unopened awaiting-author island can be retargeted"
                )
            if island_record.get("submission"):
                raise InvalidTransitionError("Cannot retarget an island after submission")
            task_path = _state_path(run_dir, str(island_record["task_file"]))
            idea_path = task_path.parent / "IDEA.md"
            packet_path = task_path.parent / "packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            idea_path.write_text(
                f"# Seed hypothesis {normalized['id']}\n\n"
                f"{str(normalized['summary']).strip()}\n\n"
                "This is a starting hypothesis/capability contract, not a closed idea list. "
                "Use the compact parent profile to refine or replace its optimization mechanism.\n",
                encoding="utf-8",
            )
            packet["idea"] = normalized
            packet_path.write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            island_record["idea"] = normalized
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            "island_retargeted",
            {"run_id": run_id, "iteration": iteration, "island": island, "idea": normalized},
        )
        return self.island_context(run_id, iteration, island)

    def submit_candidate(
        self,
        run_id: str,
        iteration: int,
        island: int,
        candidate: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_dir = self.store.run_dir(run_id)
        candidate_source = Path(candidate).expanduser().resolve()
        if not candidate_source.is_file():
            raise ConfigurationError(f"Candidate file not found: {candidate_source}")
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            _require_current_authoring(state, iteration)
            island_record = _island_record(state, iteration, island)
            target = _state_path(run_dir, str(island_record["candidate_path"]))
            if candidate_source != target.resolve():
                shutil.copy2(candidate_source, target)
            submission = {str(key): value for key, value in (metadata or {}).items()}
            submission.update(
                {
                    "candidate_path": str(target),
                    "sha256": _sha256(target),
                    "submitted_at": utc_now(),
                    "explicit": True,
                }
            )
            island_record["submission"] = submission
            island_record["status"] = IslandStatus.SUBMITTED.value
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            "candidate_submitted",
            {"run_id": run_id, "iteration": iteration, "island": island, "sha256": submission["sha256"]},
        )
        return {
            "run_id": run_id,
            "iteration": iteration,
            "island": island,
            "status": IslandStatus.SUBMITTED.value,
            "candidate_path": str(target),
            "submission": submission,
        }

    def reopen_island_for_repair(
        self,
        run_id: str,
        iteration: int,
        island: int,
    ) -> dict[str, Any]:
        """Open one bounded repair turn for a localized invalid result."""

        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            current = int(state["current_iteration"])
            if int(iteration) != current:
                raise InvalidTransitionError(f"Can only repair current iteration {current}")
            if state["phase"] != RunPhase.EVALUATED.value:
                raise InvalidTransitionError(
                    "Repairs can only be opened after the compact evaluated report"
                )
            island_record = _island_record(state, iteration, island)
            result = island_record.get("result", {})
            repair_count = int(island_record.get("repair_count", 0) or 0)
            repair_limit = int(state["config"].get("max_repairs_per_island", 1) or 0)
            if not is_repairable_result(result, repair_count, repair_limit):
                raise InvalidTransitionError(
                    "Result is not eligible for another bounded repair; advance the iteration"
                )

            island_record.setdefault("failure_history", []).append(
                {
                    "repair": repair_count,
                    "result": result,
                    "submission": island_record.get("submission", {}),
                    "archive_entry_id": island_record.get("archive_entry_id", ""),
                    "evaluated_at": island_record.get("evaluated_at", ""),
                }
            )
            repair_count += 1
            repair_path = (
                run_dir
                / f"iter_{iteration:03d}"
                / f"island_{island}"
                / "context"
                / f"REPAIR_{repair_count}.md"
            )
            candidate_path = _state_path(run_dir, str(island_record["candidate_path"]))
            idea = island_record.get("idea", {})
            compact_error = str(result.get("error", "localized candidate failure"))[:2_000]
            compile_check = _compile_check_command(
                config=state["config"],
                run_dir=run_dir,
                island_dir=run_dir / f"iter_{iteration:03d}" / f"island_{island}",
                candidate_path=candidate_path,
            )
            compile_check_block = (
                "\nRun the same bounded compile/execute check before resubmission:\n\n"
                f"```bash\n{compile_check}\n```\n\n"
                "This check may compile and execute the candidate once, but must not benchmark, "
                "profile, promote, or mutate archive state.\n"
                if compile_check
                else ""
            )
            repair_path.write_text(
                "# Bounded KernelEvo repair\n\n"
                f"- iteration: `{iteration}`\n"
                f"- island: `{island}`\n"
                f"- repair: `{repair_count}/{repair_limit}`\n"
                f"- editable candidate: `{candidate_path}`\n"
                f"- original idea: `{idea.get('summary', '') if isinstance(idea, Mapping) else ''}`\n\n"
                "## Compact failure\n\n"
                f"```text\n{compact_error}\n```\n\n"
                "Fix only this localized failure without redesigning the branch. Run Python syntax "
                "compilation and `kernel-evo cute lint`; do not run evaluation or benchmarks. "
                f"{compile_check_block}"
                "Return the candidate path and updated compact rationale.\n",
                encoding="utf-8",
            )
            island_record.update(
                {
                    "status": IslandStatus.AWAITING_AUTHOR.value,
                    "submission": {},
                    "result": None,
                    "evaluation_checkpoint": {},
                    "profile_summary": "",
                    "profile": {},
                    "profile_status": "not_selected",
                    "profile_reason": "",
                    "profile_review": {},
                    "profile_review_task": "",
                    "cute_evidence": {},
                    "promoted": False,
                    "archive_entry_id": "",
                    "repair_count": repair_count,
                    "repair_file": _relative(run_dir, repair_path),
                }
            )
            state["iterations"][str(iteration)]["status"] = RunPhase.AUTHORING.value
            state["phase"] = RunPhase.AUTHORING.value
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            "island_repair_opened",
            {
                "run_id": run_id,
                "iteration": iteration,
                "island": island,
                "repair": repair_count,
            },
        )
        return {
            "run_id": run_id,
            "iteration": iteration,
            "island": island,
            "status": IslandStatus.AWAITING_AUTHOR.value,
            "candidate_path": str(candidate_path),
            "repair_file": str(repair_path),
            "compact_error": compact_error,
        }

    def evaluate_iteration(self, run_id: str, iteration: int | None = None) -> dict[str, Any]:
        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            current = int(state["current_iteration"])
            target = current if iteration is None else int(iteration)
            if target != current:
                raise InvalidTransitionError(
                    f"Can only evaluate current iteration {current}; requested {target}."
                )
            phase = RunPhase(state["phase"])
            if phase is RunPhase.EVALUATED:
                return iteration_report(state, target)
            if phase not in {RunPhase.AUTHORING, RunPhase.EVALUATING}:
                raise InvalidTransitionError(
                    f"Cannot evaluate iteration while run phase is {phase.value}; "
                    f"{_next_action(state, self.runs_dir)}"
                )
            record = state["iterations"][str(target)]
            record["status"] = RunPhase.EVALUATING.value
            state["phase"] = RunPhase.EVALUATING.value
            self.store.write_state(run_id, state)

        config = state["config"]
        evaluator = self._evaluator or evaluator_from_config(config)
        b300_trace_feedback = str(config.get("evaluator_kind", "")) == "cute_b300" and bool(
            config.get("profile_enabled", False)
        )
        profiler = self._profiler
        if profiler is None and bool(config.get("profile_enabled", False)) and not b300_trace_feedback:
            profiler = KernelBenchProfiler()

        for island in range(int(state["island_count"])):
            with self.store.locked(run_id):
                state = self.store.read_state(run_id)
                island_record = _island_record(state, target, island)
                if island_record["status"] == IslandStatus.EVALUATED.value:
                    continue
                candidate_path = _state_path(run_dir, str(island_record["candidate_path"]))
                if not candidate_path.is_file():
                    result = EvaluationResult.failed(f"Candidate file is missing: {candidate_path}")
                    profile_result = ProfileResult()
                    island_record["status"] = IslandStatus.EVALUATING.value
                else:
                    if not island_record.get("submission"):
                        island_record["submission"] = {
                            "candidate_path": str(candidate_path),
                            "sha256": _sha256(candidate_path),
                            "submitted_at": utc_now(),
                            "explicit": False,
                        }
                    submitted_sha = str(island_record["submission"].get("sha256", ""))
                    actual_sha = _sha256(candidate_path)
                    if submitted_sha and submitted_sha != actual_sha:
                        result = EvaluationResult.failed(
                            "Candidate changed after submission; resubmit it before evaluation "
                            f"(submitted {submitted_sha[:12]}, current {actual_sha[:12]})."
                        )
                        profile_result = ProfileResult()
                    else:
                        result = None
                        profile_result = None
                    island_record["status"] = IslandStatus.EVALUATING.value
                self.store.write_state(run_id, state)

            context = self._evaluation_context(state, run_dir, target, island)
            checkpoint = island_record.get("evaluation_checkpoint", {})
            checkpoint_sha = (
                str(checkpoint.get("sha256", ""))
                if isinstance(checkpoint, Mapping)
                else ""
            )
            checkpoint_result = (
                checkpoint.get("result") if isinstance(checkpoint, Mapping) else None
            )
            if (
                result is None
                and checkpoint_sha
                and checkpoint_sha == _sha256(candidate_path)
                and isinstance(checkpoint_result, Mapping)
            ):
                result = EvaluationResult.from_metrics(checkpoint_result)
            if result is None:
                self._set_island_stage(run_id, target, island, "candidate_evaluation")
                try:
                    result = coerce_evaluation_result(_invoke_evaluator(evaluator, context))
                except Exception as exc:
                    result = EvaluationResult.failed(exc)
                with self.store.locked(run_id):
                    checkpoint_state = self.store.read_state(run_id)
                    checkpoint_record = _island_record(
                        checkpoint_state, target, island
                    )
                    checkpoint_record["evaluation_checkpoint"] = {
                        "sha256": _sha256(candidate_path),
                        "result": result.to_dict(),
                        "completed_at": utc_now(),
                    }
                    checkpoint_record["stage"] = "capability_validation"
                    checkpoint_record["stage_started_at"] = utc_now()
                    self.store.write_state(run_id, checkpoint_state)
                self._set_island_stage(run_id, target, island, "capability_validation")
            if profile_result is None:
                profile_result = ProfileResult()

            cute_evidence: dict[str, Any] = {}
            if (
                str(config.get("backend", "")) == "cute"
                and str(config.get("evaluator_kind", "")) != "cute_b300"
            ):
                from kernel_evo.cute_harness.capabilities import capability_issues
                from kernel_evo.cute_harness.lint import candidate_kernel_delta, lint_cute_source

                context_metadata = island_record.get("cute_context", {})
                if not isinstance(context_metadata, Mapping):
                    context_metadata = {}
                source_lint = lint_cute_source(
                    candidate_path.read_text(encoding="utf-8", errors="replace")
                    if candidate_path.is_file()
                    else "",
                    precision=str(config.get("precision", "bf16")),
                    arch=str(context_metadata.get("arch", config.get("cute_arch", ""))) or "sm_90a",
                    operation=str(context_metadata.get("operation", "")),
                    codegen_contract=str(context_metadata.get("idea_codegen_contract", "")),
                )
                idea = island_record.get("idea", {})
                if not isinstance(idea, Mapping):
                    idea = {}
                requires_candidate_kernel = bool(idea.get("requires_candidate_kernel", False))
                ownership_evidence: dict[str, Any] = {}
                if requires_candidate_kernel:
                    baseline_source = (
                        context.baseline_path.read_text(encoding="utf-8", errors="replace")
                        if context.baseline_path.is_file()
                        else ""
                    )
                    candidate_source = (
                        candidate_path.read_text(encoding="utf-8", errors="replace")
                        if candidate_path.is_file()
                        else ""
                    )
                    ownership_evidence = candidate_kernel_delta(
                        candidate_source, baseline_source
                    )
                evaluator_environment = result.metadata.get("cute_environment", {})
                if not isinstance(evaluator_environment, Mapping):
                    evaluator_environment = {}
                environment_issues = (
                    capability_issues(
                        evaluator_environment,
                        precision=str(config.get("precision", "bf16")),
                        required_arch=str(config.get("cute_arch", "")),
                        required_version=str(
                            context_metadata.get(
                                "version", config.get("cute_required_version", "")
                            )
                        ),
                    )
                    if evaluator_environment
                    else []
                )
                codegen_reports = result.metadata.get("cute_codegen", [])
                if not isinstance(codegen_reports, list):
                    codegen_reports = []
                codegen_gate: dict[str, Any] = {}
                contract_values = context_metadata.get("codegen_contracts", [])
                contracts = (
                    [str(value) for value in contract_values]
                    if isinstance(contract_values, list)
                    else []
                )
                idea_id = str(context_metadata.get("idea_id", ""))
                contract_kind = str(context_metadata.get("idea_codegen_contract", ""))
                if contract_kind and codegen_reports:
                    from kernel_evo.cute_harness.codegen import verify_codegen_reports

                    usable_reports = [
                        item
                        for item in codegen_reports
                        if isinstance(item, Mapping) and not item.get("error")
                    ]
                    if usable_reports:
                        marker = (
                            "hopper_wgmma_gemm"
                            if contract_kind == "hopper_wgmma"
                            else "bf16_vector_add_aligned/expected_codegen.yaml"
                        )
                        contract_path = next(
                            (value for value in contracts if marker in value),
                            "",
                        )
                        if not contract_path:
                            from kernel_evo.cute_harness.paths import harness_root

                            example = (
                                "hopper_wgmma_gemm"
                                if contract_kind == "hopper_wgmma"
                                else "bf16_vector_add_aligned"
                            )
                            contract_path = str(
                                harness_root() / "examples" / example / "expected_codegen.yaml"
                            )
                        codegen_gate = verify_codegen_reports(usable_reports, contract_path)
                elif contract_kind and not codegen_reports:
                    runtime = result.metadata.get("cute_runtime", {})
                    artifacts = result.metadata.get("cute_artifacts", [])
                    executed = (
                        int(runtime.get("executed_executor_count", 0) or 0)
                        if isinstance(runtime, Mapping)
                        else 0
                    )
                    if executed == 0:
                        artifact_failure = (
                            "the production compiled executor did not execute; the candidate likely "
                            "took a Torch/fallback path"
                        )
                    elif isinstance(artifacts, list) and artifacts:
                        artifact_failure = (
                            "candidate artifacts were retained, but inspection produced no usable "
                            "codegen report"
                        )
                    else:
                        artifact_failure = "the evaluator retained no candidate artifact"
                    codegen_gate = {
                        "passed": False,
                        "failures": [
                            {
                                "message": (
                                    f"Idea `{idea_id}` requires `{contract_kind}` codegen evidence, "
                                    f"but {artifact_failure}."
                                )
                            }
                        ],
                    }
                cute_evidence = {
                    "source_lint": source_lint,
                    "author_capability_fingerprint": str(
                        context_metadata.get("capability_fingerprint", "")
                    ),
                    "evaluator_environment": dict(evaluator_environment),
                    "capability_issues": environment_issues,
                    "artifacts": result.metadata.get("cute_artifacts", []),
                    "codegen": codegen_reports,
                    "codegen_gate": codegen_gate,
                }
                result.metadata["cute_evidence"] = cute_evidence
                blocking_capability = any(
                    item.get("severity") == "error" for item in environment_issues
                )
                runtime_evidence = result.metadata.get("cute_runtime", {})
                executed_count = (
                    int(runtime_evidence.get("executed_executor_count", 0) or 0)
                    if isinstance(runtime_evidence, Mapping)
                    else 0
                )
                if requires_candidate_kernel:
                    baseline_entry = find_entry(
                        state.get("archive", {}),
                        str(island_record.get("baseline_entry_id", "")),
                    )
                    baseline_result = (
                        baseline_entry.get("result", {})
                        if isinstance(baseline_entry, Mapping)
                        else {}
                    )
                    baseline_metadata = (
                        baseline_result.get("metadata", {})
                        if isinstance(baseline_result, Mapping)
                        else {}
                    )
                    baseline_runtime = (
                        baseline_metadata.get("cute_runtime", {})
                        if isinstance(baseline_metadata, Mapping)
                        else {}
                    )
                    baseline_executed = (
                        int(baseline_runtime.get("executed_executor_count", 0) or 0)
                        if isinstance(baseline_runtime, Mapping)
                        else 0
                    )
                    min_new_executors = int(idea.get("min_new_executors", 1) or 1)
                    executor_delta = executed_count - baseline_executed
                    changed_kernels = ownership_evidence.get("changed", [])
                    ownership_evidence.update(
                        {
                            "required": True,
                            "baseline_executed_executor_count": baseline_executed,
                            "candidate_executed_executor_count": executed_count,
                            "executed_executor_delta": executor_delta,
                            "min_new_executors": min_new_executors,
                            "passed": bool(changed_kernels)
                            and executed_count >= min_new_executors,
                        }
                    )
                    cute_evidence["candidate_kernel_ownership"] = ownership_evidence
                    result.metadata["candidate_kernel_ownership"] = ownership_evidence
                source_errors = [
                    item
                    for item in source_lint.get("issues", [])
                    if isinstance(item, Mapping)
                    and item.get("severity") == "error"
                    and not (
                        item.get("code") == "compiled-executor-unused"
                        and executed_count > 0
                    )
                ]
                blocking_source = bool(source_errors)
                blocking_codegen = bool(codegen_gate) and not bool(codegen_gate.get("passed"))
                blocking_ownership = requires_candidate_kernel and not bool(
                    ownership_evidence.get("passed")
                )
                if (
                    result.compiled
                    and blocking_capability
                    and bool(config.get("cute_capability_gate", True))
                ):
                    result.valid = False
                    result.status = "invalid_environment"
                    result.error = "; ".join(str(item.get("message", "")) for item in environment_issues)
                elif (
                    result.compiled
                    and blocking_source
                    and bool(config.get("cute_compliance_gate", True))
                ):
                    result.valid = False
                    result.status = "invalid_compliance"
                    result.error = "; ".join(
                        f"{item.get('code')}: {item.get('message')}"
                        for item in source_errors
                    )[:2_000]
                elif result.compiled and blocking_ownership:
                    changed = ownership_evidence.get("changed", [])
                    delta = int(ownership_evidence.get("executed_executor_delta", 0) or 0)
                    required_delta = int(ownership_evidence.get("min_new_executors", 1) or 1)
                    result.valid = False
                    result.status = "invalid_compliance"
                    if not changed:
                        result.error = (
                            "candidate-owned-kernel: no new or materially modified candidate-local "
                            "@cute.kernel body was found relative to the parent"
                        )
                    else:
                        result.error = (
                            "candidate-owned-kernel: changed kernel(s) "
                            f"{', '.join(str(value) for value in changed)} were found, but executed "
                            f"executor count was {executed_count}; required at least "
                            f"{required_delta} (delta versus parent: {delta})"
                        )

                produces_capability = str(
                    context_metadata.get("idea_produces_capability", "")
                )
                contribution = result.metadata.get("production_contribution", {})
                if (
                    produces_capability == "wgmma_production_output"
                    and result.compiled
                    and result.correctness
                    and not (
                        isinstance(contribution, Mapping)
                        and bool(contribution.get("passed"))
                    )
                ):
                    result.valid = False
                    result.status = "invalid_compliance"
                    result.error = str(
                        contribution.get(
                            "error", "WGMMA output contribution was not demonstrated."
                        )
                        if isinstance(contribution, Mapping)
                        else "WGMMA output contribution was not demonstrated."
                    )
                elif (
                    result.compiled
                    and result.correctness
                    and blocking_codegen
                    and bool(config.get("cute_codegen_gate", True))
                ):
                    result.valid = False
                    result.status = "invalid_codegen"
                    result.error = "; ".join(
                        str(item.get("message", ""))
                        for item in codegen_gate.get("failures", [])
                        if isinstance(item, Mapping)
                    )[:2_000]

                if contract_kind:
                    families = {"wgmma": 0, "tma": 0, "mbarrier": 0, "vector": 0}
                    for report in codegen_reports:
                        if not isinstance(report, Mapping) or report.get("error"):
                            continue
                        observed = report.get("instruction_families", {})
                        if not isinstance(observed, Mapping):
                            continue
                        for name in ("wgmma", "tma", "mbarrier"):
                            families[name] += int(observed.get(name, 0) or 0)
                        families["vector"] += int(
                            observed.get("vector_global_load_128", 0) or 0
                        ) + int(observed.get("vector_global_store_128", 0) or 0)
                    targets = (
                        ("wgmma", "tma", "mbarrier")
                        if contract_kind == "hopper_wgmma"
                        else ("vector",)
                    )
                    milestones = {
                        "compiled": bool(result.compiled),
                        "correctness": bool(result.correctness),
                        "executor_executed": executed_count > 0,
                        "artifact_retained": bool(codegen_reports),
                        **{f"has_{name}": families[name] > 0 for name in targets},
                    }
                    if requires_candidate_kernel:
                        milestones["candidate_owned_kernel"] = bool(
                            ownership_evidence.get("passed")
                        )
                    score = (
                        0.15 * milestones["compiled"]
                        + 0.35 * milestones["correctness"]
                        + 0.15 * milestones["executor_executed"]
                        + 0.10 * milestones["artifact_retained"]
                        + 0.25
                        * sum(milestones[f"has_{name}"] for name in targets)
                        / len(targets)
                    )
                    result.metadata["development_progress"] = {
                        "contract": contract_kind,
                        "score": round(score, 6),
                        "milestones": milestones,
                        "instruction_families": families,
                        "production_valid": bool(result.valid),
                    }

            candidate_sha = _sha256(candidate_path) if candidate_path.is_file() else ""
            duplicate_repair = _is_byte_identical_repair(
                state, island_record, candidate_sha
            )
            profile_reason = (
                "byte_identical_repair"
                if duplicate_repair
                else self._candidate_profile_reason(state, island_record, result)
            )
            profile_was_interrupted = (
                island_record.get("profile_status") == "running"
                and bool(island_record.get("profile_reason"))
            )
            if duplicate_repair:
                profile_result = ProfileResult(
                    summary="Profiling skipped: repaired source is byte-identical to its parent.",
                    data={"status": "skipped", "reason": "byte_identical_repair"},
                )
                result.metadata["profile_reason"] = profile_reason
            elif profile_was_interrupted:
                profile_reason = str(island_record.get("profile_reason", profile_reason))
                profile_result = ProfileResult(
                    summary=(
                        "Optional profiling was interrupted after evaluation; "
                        "the valid evaluation result was preserved."
                    ),
                    data={"status": "failed", "reason": "interrupted_optional_profile"},
                )
                result.metadata["profile_reason"] = profile_reason
            elif b300_trace_feedback:
                profile_result = _b300_trace_profile(context.island_dir / "b300")
                profile_reason = B300_TRACE_REASON if profile_result.summary else ""
                result.metadata["profile_reason"] = profile_reason
            elif profiler is not None and result.valid and (profile_reason or self._profiler is not None):
                profile_reason = profile_reason or "injected_profiler"
                self._set_island_stage(run_id, target, island, "candidate_profiling")
                with self.store.locked(run_id):
                    profile_state = self.store.read_state(run_id)
                    profile_record = _island_record(profile_state, target, island)
                    profile_record["profile_status"] = "running"
                    profile_record["profile_reason"] = profile_reason
                    self.store.write_state(run_id, profile_state)
                try:
                    profile_result = coerce_profile_result(
                        _invoke_profiler(profiler, context, result)
                    )
                except Exception as exc:
                    profile_result = ProfileResult(
                        summary=f"Profiling failed: {type(exc).__name__}: {exc}",
                        data={"status": "failed", "reason": f"{type(exc).__name__}: {exc}"},
                    )
                torch_profile = profile_result.data.get("torch", {})
                graph_gate = (
                    torch_profile.get("graph_capturability_gate", {})
                    if isinstance(torch_profile, Mapping)
                    else {}
                )
                if (
                    bool(config.get("profile_require_graph_capturable", True))
                    and str(config.get("measurement_mode", "wall-clock"))
                    != "device-time"
                    and isinstance(graph_gate, Mapping)
                    and graph_gate.get("passed") is False
                ):
                    parent_entry = find_entry(
                        state.get("archive", {}),
                        str(island_record.get("baseline_entry_id", "")),
                    )
                    parent_profile = (
                        parent_entry.get("profile", {})
                        if isinstance(parent_entry, Mapping)
                        else {}
                    )
                    parent_torch = (
                        parent_profile.get("torch", {})
                        if isinstance(parent_profile, Mapping)
                        else {}
                    )
                    parent_gate = (
                        parent_torch.get("graph_capturability_gate", {})
                        if isinstance(parent_torch, Mapping)
                        else {}
                    )
                    idea = island_record.get("idea", {})
                    requires_graph = (
                        bool(idea.get("requires_graph_capturable", False))
                        if isinstance(idea, Mapping)
                        else False
                    )
                    parent_capturable = (
                        parent_gate.get("passed") is True
                        if isinstance(parent_gate, Mapping)
                        else False
                    )
                    enforce_graph_gate = parent_capturable or requires_graph
                    result.metadata["graph_capturability"] = {
                        "passed": False,
                        "enforced": enforce_graph_gate,
                        "parent_capturable": parent_capturable,
                        "requires_graph_capturable": requires_graph,
                        "failure": str(graph_gate.get("failure", ""))[:1_000],
                    }
                    if enforce_graph_gate:
                        result.valid = False
                        result.status = "invalid_graph"
                        result.error = (
                            "graph-capturability regression: "
                            + str(
                                graph_gate.get(
                                    "failure", "candidate forward is not CUDA-graph capturable"
                                )
                            )
                        )[:2_000]
                result.metadata["profile_reason"] = profile_reason
                progress = result.metadata.get("development_progress", {})
                if isinstance(progress, dict):
                    progress["production_valid"] = bool(result.valid)
            self._set_island_stage(run_id, target, island, "archiving")

            result_payload = result.to_dict()
            self.store.write_json(
                context.island_dir / "result.json",
                {"result": result_payload, "profile_summary": profile_result.summary, "profile": profile_result.data},
            )
            with self.store.locked(run_id):
                state = self.store.read_state(run_id)
                island_record = _island_record(state, target, island)
                island_record["result"] = result_payload
                island_record["profile_summary"] = profile_result.summary
                island_record["profile"] = profile_result.data
                island_record["profile_reason"] = profile_reason
                island_record["profile_status"] = (
                    "skipped_duplicate"
                    if duplicate_repair
                    else profile_result_status(profile_result)
                    if profile_reason
                    else "not_selected"
                )
                island_record["cute_evidence"] = cute_evidence
                island_record["evaluated_at"] = utc_now()
                island_record["status"] = IslandStatus.EVALUATED.value
                if profile_reason:
                    stats = state.setdefault("profiling", {})
                    if profile_reason == "bounded_regression":
                        stats["regression_profiles_used"] = int(
                            stats.get("regression_profiles_used", 0) or 0
                        ) + 1
                    elif profile_reason.startswith("first_capability:"):
                        stats["capability_profiles"] = int(
                            stats.get("capability_profiles", 0) or 0
                        ) + 1
                        capability = profile_reason.split(":", 1)[1]
                        profiled = stats.setdefault("capabilities_profiled", [])
                        if capability not in profiled:
                            profiled.append(capability)
                self.store.write_state(run_id, state)
            self._tracker(state, run_dir).emit(
                "candidate_evaluated",
                {
                    "run_id": run_id,
                    "iteration": target,
                    "island": island,
                    "valid": result.valid,
                    "speedup": result.speedup,
                },
            )

        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            archived = self.idea_store.archive_iteration(state, run_dir, target)
            if str(state["config"].get("backend", "")) == "cute" and bool(
                state["config"].get("cute_record_experiments", True)
            ):
                from kernel_evo.cute_harness.experiments import record_archive_evaluation

                database_value = str(state["paths"].get("cute_experiments", ""))
                if database_value:
                    database = _state_path(run_dir, database_value)
                    for entry in archived:
                        context_metadata = entry.get("cute_context", {})
                        if not isinstance(context_metadata, Mapping):
                            context_metadata = {}
                        try:
                            record_archive_evaluation(
                                database,
                                entry=entry,
                                context=context_metadata,
                            )
                        except (OSError, TypeError, ValueError) as exc:
                            entry["experiment_record_error"] = f"{type(exc).__name__}: {exc}"
            state["iterations"][str(target)]["status"] = RunPhase.EVALUATED.value
            state["iterations"][str(target)]["evaluated_at"] = utc_now()
            state["phase"] = RunPhase.EVALUATED.value
            self.store.write_state(run_id, state)

        report = iteration_report(state, target)
        self.store.write_json(run_dir / f"iter_{target:03d}" / "report.json", report)
        (run_dir / f"iter_{target:03d}" / "report.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
        for entry in archived:
            self.store.append_jsonl(run_dir / "ideas.jsonl", entry)
        self._tracker(state, run_dir).emit(
            "iteration_evaluated",
            {
                "run_id": run_id,
                "iteration": target,
                "valid_candidates": report["valid_candidates"],
                "promoted_candidates": report["promoted_candidates"],
            },
        )
        return report

    def record_critic_hints(
        self,
        run_id: str,
        hints: Sequence[str],
        *,
        island: int = 0,
        iteration: int | None = None,
    ) -> list[str]:
        """Attach one between-turn critique; the next packet leads its feedback with it.

        Hints are keyed by island rather than by archive entry because a failed
        candidate is never promoted, so the next turn's parent is usually not the
        candidate the critique is about.
        """
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            target = int(state["current_iteration"]) if iteration is None else int(iteration)
            cleaned = [" ".join(str(hint).split()) for hint in hints]
            cleaned = [hint for hint in cleaned if hint][:8]
            critiques = state.setdefault("critic", {})
            if cleaned:
                critiques[str(island)] = {"iteration": target, "hints": cleaned}
            else:
                critiques.pop(str(island), None)
            record = state.get("iterations", {}).get(str(target), {})
            islands = record.get("islands", {}) if isinstance(record, Mapping) else {}
            island_record = islands.get(str(island))
            if isinstance(island_record, dict):
                island_record["critic_hints"] = cleaned
            self.store.write_state(run_id, state)
        self._tracker(state, self.store.run_dir(run_id)).emit(
            "critic_recorded",
            {"run_id": run_id, "iteration": target, "island": island, "hints": len(cleaned)},
        )
        return cleaned

    def report_iteration(
        self,
        run_id: str,
        iteration: int | None = None,
        *,
        format: str = "markdown",
    ) -> str | dict[str, Any]:
        state = self.store.read_state(run_id)
        target = int(state["current_iteration"]) if iteration is None else int(iteration)
        report = iteration_report(state, target)
        if format == "json":
            return report
        if format != "markdown":
            raise ConfigurationError("Report format must be markdown or json")
        return render_markdown(report)

    def prepare_profile_reviews(
        self, run_id: str, iteration: int | None = None
    ) -> list[dict[str, Any]]:
        """Create one read-only compact-trace review task per profiled candidate."""
        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            target = int(state["current_iteration"]) if iteration is None else int(iteration)
            if RunPhase(state["phase"]) is not RunPhase.EVALUATED:
                raise InvalidTransitionError("Profile review tasks are available after evaluation.")
            tasks: list[dict[str, Any]] = []
            record = state["iterations"][str(target)]
            limit = int(state["config"].get("profile_agent_idea_limit", 3) or 3)
            for island_key, island_record in sorted(record["islands"].items(), key=lambda item: int(item[0])):
                if (
                    island_record.get("profile_status") != "completed"
                    or island_record.get("profile_review")
                ):
                    continue
                island = int(island_key)
                context_dir = run_dir / f"iter_{target:03d}" / f"island_{island}" / "context"
                context_dir.mkdir(parents=True, exist_ok=True)
                task_path = context_dir / "PROFILE_REVIEW.md"
                output_path = context_dir / "PROFILE_REVIEW.json"
                candidate_path = _state_path(run_dir, str(island_record["candidate_path"]))
                task_path.write_text(
                    _profile_review_markdown(
                        run_id=run_id,
                        iteration=target,
                        island=island,
                        candidate_path=candidate_path,
                        output_path=output_path,
                        summary=str(island_record.get("profile_summary", "")),
                        idea_limit=limit,
                        measurement_mode=str(
                            state["config"].get("measurement_mode", "wall-clock")
                        ),
                    ),
                    encoding="utf-8",
                )
                island_record["profile_review_task"] = _relative(run_dir, task_path)
                tasks.append(
                    {
                        "run_id": run_id,
                        "role": "kernel-profile-reviewer",
                        "iteration": target,
                        "island": island,
                        "task_file": str(task_path.resolve()),
                        "editable_files": [str(output_path.resolve())],
                        "readable_files": [
                            str(candidate_path.resolve()),
                            str(task_path.resolve()),
                        ],
                        "output_file": str(output_path.resolve()),
                    }
                )
            self.store.write_state(run_id, state)
        return tasks

    def submit_profile_review(
        self,
        run_id: str,
        iteration: int,
        island: int,
        review_file: str | Path,
    ) -> dict[str, Any]:
        path = Path(review_file).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(f"Profile review file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ConfigurationError("Profile review must be a JSON object")
        findings = str(payload.get("findings", "")).strip()
        if not findings:
            raise ConfigurationError("Profile review requires non-empty findings")
        raw_ideas = payload.get("ideas", [])
        if not isinstance(raw_ideas, list):
            raise ConfigurationError("Profile review ideas must be a JSON list")
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            if RunPhase(state["phase"]) is not RunPhase.EVALUATED:
                raise InvalidTransitionError("Profile reviews can only be submitted after evaluation")
            record = _island_record(state, iteration, island)
            if record.get("profile_status") != "completed":
                raise InvalidTransitionError("Island has no completed compact profile to review")
            limit = int(state["config"].get("profile_agent_idea_limit", 3) or 3)
            ideas: list[dict[str, str]] = []
            for index, raw in enumerate(raw_ideas[:limit]):
                item = {"summary": raw} if isinstance(raw, str) else raw
                if not isinstance(item, Mapping) or not str(item.get("summary", "")).strip():
                    raise ConfigurationError(f"Profile review idea {index + 1} requires a summary")
                ideas.append(
                    {
                        str(key): str(value)
                        for key, value in item.items()
                        if value not in (None, "")
                    }
                )
            review = {
                "findings": findings,
                "ideas": ideas,
                "reviewed_at": utc_now(),
                "source": "compact_profile",
            }
            record["profile_review"] = review
            entry = find_entry(state["archive"], str(record.get("archive_entry_id", "")))
            if isinstance(entry, dict):
                entry["profile_review"] = review
            self.store.write_state(run_id, state)
        return {"run_id": run_id, "iteration": iteration, "island": island, "profile_review": review}

    def advance_iteration(self, run_id: str) -> dict[str, Any]:
        run_dir = self.store.run_dir(run_id)
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            phase = RunPhase(state["phase"])
            if phase is RunPhase.COMPLETE:
                return self.status(run_id)
            if phase is not RunPhase.EVALUATED:
                raise InvalidTransitionError(
                    f"Cannot advance while run phase is {phase.value}; "
                    f"{_next_action(state, self.runs_dir)}"
                )
            if bool(state["config"].get("profile_review_required", True)):
                record = state.get("iterations", {}).get(str(state["current_iteration"]), {})
                island_records = record.get("islands", {}) if isinstance(record, Mapping) else {}
                pending = [
                    int(key)
                    for key, value in island_records.items()
                    if isinstance(value, Mapping)
                    and value.get("profile_status") == "completed"
                    and not value.get("profile_review")
                    # The B300 trace summary is deterministic harness output, so
                    # the barrier loop needs no reviewer between turns.
                    and value.get("profile_reason") != B300_TRACE_REASON
                ]
                if pending:
                    raise InvalidTransitionError(
                        "Compact profile review required before advance for island(s): "
                        + ", ".join(str(value) for value in pending)
                    )
            current = int(state["current_iteration"])
            if current >= int(state["steps"]):
                state["phase"] = RunPhase.COMPLETE.value
                event = "run_completed"
            else:
                state["current_iteration"] = current + 1
                state["phase"] = RunPhase.READY.value
                event = "iteration_advanced"
            self.store.write_state(run_id, state)
        self._tracker(state, run_dir).emit(
            event,
            {"run_id": run_id, "iteration": int(state["current_iteration"]), "phase": state["phase"]},
        )
        return self.status(run_id)

    def _load_authoring_tasks(
        self, state: Mapping[str, Any], run_dir: Path, iteration: int
    ) -> list[AuthoringTask]:
        record = state["iterations"][str(iteration)]
        tasks: list[AuthoringTask] = []
        for key in sorted(record["islands"], key=int):
            task_file = _state_path(run_dir, record["islands"][key]["task_file"])
            packet = json.loads((task_file.parent / "packet.json").read_text(encoding="utf-8"))
            tasks.append(_task_from_packet(packet))
        return tasks

    def _set_island_stage(
        self, run_id: str, iteration: int, island: int, stage: str
    ) -> None:
        with self.store.locked(run_id):
            state = self.store.read_state(run_id)
            record = _island_record(state, iteration, island)
            record["stage"] = stage
            record["stage_started_at"] = utc_now()
            self.store.write_state(run_id, state)

    def _ensure_parent_profile(
        self,
        *,
        state: dict[str, Any],
        run_dir: Path,
        entry: Mapping[str, Any],
        island: int,
    ) -> str:
        config = state.get("config", {})
        if not (
            isinstance(config, Mapping)
            and bool(config.get("profile_enabled", False))
            and bool(config.get("profile_parent_before_use", True))
        ):
            return "not_required"
        if str(config.get("evaluator_kind", "")) == "cute_b300":
            # The remote device has no local profiler; the parent's summary was
            # attached from its own B300 trace when it was evaluated.
            return "not_required"
        if entry.get("profile_status") == "completed":
            return "completed"
        profiler = self._profiler or KernelBenchProfiler()
        entry_id = str(entry.get("id", "parent"))
        source = _state_path(run_dir, str(entry["path"]))
        problem_value = str(state.get("paths", {}).get("problem_dir", ""))
        problem_dir = _state_path(run_dir, problem_value) if problem_value else None
        run_config: dict[str, Any] = {}
        run_config_value = str(state.get("paths", {}).get("run_config", ""))
        if run_config_value:
            run_config = json.loads(
                _state_path(run_dir, run_config_value).read_text(encoding="utf-8")
            )
        profile_dir = run_dir / "parent_profiles" / entry_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        context = EvaluationContext(
            run_id=str(state["run_id"]),
            iteration=int(entry.get("iteration", 0) or 0),
            island=island,
            run_dir=run_dir,
            island_dir=profile_dir,
            candidate_path=source,
            baseline_path=source,
            problem_dir=problem_dir,
            config=config,
            run_config=run_config,
        )
        raw_result = entry.get("result", {})
        result = EvaluationResult.from_metrics(raw_result if isinstance(raw_result, Mapping) else {})
        try:
            profile = coerce_profile_result(_invoke_profiler(profiler, context, result))
            status = profile_result_status(profile)
            if status in {"failed", "skipped"} and not (profile.data or profile.summary):
                status = "unavailable"
                profile = ProfileResult(
                    summary="Parent profile unavailable: profiler produced no compact metrics.",
                    data={"status": "unavailable", "reason": "profiler produced no compact metrics"},
                )
        except Exception as exc:
            status = "failed"
            profile = ProfileResult(
                summary=f"Parent profiling failed: {type(exc).__name__}: {exc}",
                data={"status": "failed", "reason": f"{type(exc).__name__}: {exc}"},
            )
        if isinstance(entry, dict):
            previous_status = str(entry.get("profile_status", ""))
            entry["profile_status"] = status
            entry["profile_summary"] = profile.summary
            entry["profile"] = profile.data
            entry["profiled_at"] = utc_now()
            if status == "completed":
                entry["parent_profile_failures"] = 0
            else:
                previous_failures = int(entry.get("parent_profile_failures", 0) or 0)
                if previous_failures == 0 and previous_status in {"failed", "unavailable"}:
                    previous_failures = 1
                entry["parent_profile_failures"] = previous_failures + 1
        stats = state.setdefault("profiling", {})
        stats["parent_profiles"] = int(stats.get("parent_profiles", 0) or 0) + 1
        self.store.write_json(
            profile_dir / "summary.json",
            {"entry_id": entry_id, "status": status, "summary": profile.summary, "profile": profile.data},
        )
        return status

    @staticmethod
    def _candidate_profile_reason(
        state: Mapping[str, Any], island_record: Mapping[str, Any], result: EvaluationResult
    ) -> str:
        config = state.get("config", {})
        if not isinstance(config, Mapping) or not bool(config.get("profile_enabled", False)):
            return ""
        if not result.valid:
            return ""
        archive = state.get("archive", {})
        context = island_record.get("cute_context", {})
        capability = ""
        if isinstance(context, Mapping):
            capability = str(context.get("idea_produces_capability", ""))
            if not capability:
                capability = str(context.get("idea_codegen_contract", ""))
        if (
            capability
            and bool(config.get("profile_first_capability", True))
            and capability not in archive_capabilities(archive if isinstance(archive, Mapping) else {})
        ):
            stats = state.get("profiling", {})
            already_profiled = (
                stats.get("capabilities_profiled", []) if isinstance(stats, Mapping) else []
            )
            if capability not in already_profiled:
                return f"first_capability:{capability}"
        if result.speedup >= float(config.get("profile_min_speedup", 1.0)):
            return "speedup_threshold"
        stats = state.get("profiling", {})
        used = int(stats.get("regression_profiles_used", 0) or 0) if isinstance(stats, Mapping) else 0
        budget = int(config.get("profile_regression_budget", 4) or 0)
        if used < budget:
            return "bounded_regression"
        return ""

    def _evaluation_context(
        self,
        state: Mapping[str, Any],
        run_dir: Path,
        iteration: int,
        island: int,
    ) -> EvaluationContext:
        record = _island_record(state, iteration, island)
        problem_value = str(state["paths"].get("problem_dir", ""))
        problem_dir = _state_path(run_dir, problem_value) if problem_value else None
        run_config: dict[str, Any] = {}
        run_config_value = str(state["paths"].get("run_config", ""))
        if run_config_value:
            run_config_path = _state_path(run_dir, run_config_value)
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        return EvaluationContext(
            run_id=str(state["run_id"]),
            iteration=iteration,
            island=island,
            run_dir=run_dir,
            island_dir=run_dir / f"iter_{iteration:03d}" / f"island_{island}",
            candidate_path=_state_path(run_dir, str(record["candidate_path"])),
            baseline_path=_state_path(run_dir, str(record["baseline_path"])),
            problem_dir=problem_dir,
            config=state["config"],
            run_config=run_config,
        )

    @staticmethod
    def _tracker(state: Mapping[str, Any], run_dir: Path) -> EventTracker:
        config = state.get("config", {})
        remote = str(config.get("tracker", "")) if isinstance(config, Mapping) else ""
        return EventTracker(run_dir, remote)


def _invoke_evaluator(evaluator: EvaluatorLike, context: EvaluationContext) -> Any:
    method = getattr(evaluator, "evaluate", None)
    if callable(method):
        return method(context)
    if callable(evaluator):
        return evaluator(context)
    raise TypeError("Evaluator is neither callable nor a CandidateEvaluator")


def _invoke_profiler(
    profiler: ProfilerLike, context: EvaluationContext, result: EvaluationResult
) -> Any:
    method = getattr(profiler, "profile", None)
    if callable(method):
        return method(context, result)
    if callable(profiler):
        return profiler(context, result)
    raise TypeError("Profiler is neither callable nor a CandidateProfiler")


def _is_byte_identical_repair(
    state: Mapping[str, Any], island_record: Mapping[str, Any], candidate_sha: str
) -> bool:
    """Return whether a repair restored the exact parent source.

    Correctness and performance evaluation still run because environment and
    measurement evidence may have changed. The redundant optional profile is
    skipped; the parent already owns profile evidence for these exact bytes.
    """
    if not candidate_sha or int(island_record.get("repair_count", 0) or 0) <= 0:
        return False
    archive = state.get("archive", {})
    if not isinstance(archive, Mapping):
        return False
    parent = find_entry(archive, str(island_record.get("baseline_entry_id", "")))
    return isinstance(parent, Mapping) and str(parent.get("sha256", "")) == candidate_sha


def _profile_review_markdown(
    *,
    run_id: str,
    iteration: int,
    island: int,
    candidate_path: Path,
    output_path: Path,
    summary: str,
    idea_limit: int,
    measurement_mode: str,
) -> str:
    return f"""# KernelEvo compact profile review

- role: `kernel-profile-reviewer`
- run: `{run_id}`
- iteration: `{iteration}`
- island: `{island}`
- measured objective: `{measurement_mode}`

## Permissions

- Read candidate source: `{candidate_path.resolve()}`
- Read only this generated compact profile packet and the candidate source.
- Do not inspect raw Torch, Nsight Systems, NCU, evaluator, or profiler artifacts.
- Do not open evaluator logs or any file outside the explicit readable list.
- Write only: `{output_path.resolve()}`
- Do not edit the candidate or run benchmarks.

## Compact per-operation profile

{summary.strip() or "Profile summary unavailable."}

## Task

Answer this engineering question: **How should this kernel/layer be optimized to obtain a large,
measurable performance improvement? Which optimizations are most valuable, exactly where should
they be implemented, and why?**

First produce a causal report covering inner device work, eager complete-layer latency, dispatch
gaps, CUDA-graph replay, launch fragmentation, memory movement, and compute/memory ceilings. Distinguish
host-only improvements from device-time improvements and match recommendations to the configured measured
objective. Identify boundaries that can be optimized locally and boundaries that require fusion or a larger
rewrite. Do not merely summarize the trace or repeat static harness suggestions.

Then produce up to {idea_limit} **ranked** optimization ideas, highest expected value first. Every idea must
name the candidate function, operation sequence, or tensor boundary where work should occur; cite measured
evidence; explain the performance mechanism; estimate the likely upside; give a concrete implementation
plan; and state correctness and implementation risks. Prefer changes capable of material speedups over
cosmetic Python or dispatch edits when device time is the objective. Ideas are evolutionary memory, not
mandatory promotions; preserve correctness and explicit codegen contracts.

Write JSON with this schema:

```json
{{
  "findings": "full causal optimization report explaining the bottlenecks and required rewrite scope",
  "ideas": [
    {{
      "rank": 1,
      "summary": "one concrete, bounded optimization",
      "implementation_location": "specific function, operation sequence, or tensor boundary",
      "profile_evidence": "measured counters/timings/launches supporting this priority",
      "expected_perf_mechanism": "which measured cost it should reduce and how",
      "estimated_upside": "realistic impact and confidence",
      "implementation_plan": "specific code-level approach",
      "risk": "correctness and implementation risks"
    }}
  ]
}}
```
"""



def _new_run_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")[:28] or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    entropy = hashlib.sha256(f"{time.time_ns()}:{name}".encode()).hexdigest()[:6]
    return f"{slug}-{stamp}-{entropy}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _critic_feedback(state: Mapping[str, Any], island: int) -> list[str]:
    critiques = state.get("critic", {})
    record = critiques.get(str(island), {}) if isinstance(critiques, Mapping) else {}
    if not isinstance(record, Mapping):
        return []
    hints = record.get("hints", [])
    if not isinstance(hints, list):
        return []
    turn = record.get("iteration")
    label = f"Critic on turn {turn}" if turn else "Critic"
    return [f"{label}: {str(hint).strip()}" for hint in hints if str(hint).strip()]


def _b300_trace_profile(b300_dir: Path) -> ProfileResult:
    """Reuse the island's own remote B300 trace summary as author feedback.

    Only this island's evaluation directory is read. The run-level baseline
    profile belongs to the verified reference and must never reach an author.
    """
    summary_path = b300_dir / "profile_summary.md"
    if not summary_path.is_file():
        return ProfileResult()
    data: dict[str, Any] = {"status": "completed", "source": "b300_torch_trace"}
    record_path = b300_dir / "result.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        if isinstance(record, Mapping) and isinstance(record.get("profile_summary"), Mapping):
            data["b300_trace"] = dict(record["profile_summary"])
        if isinstance(record, Mapping) and isinstance(record.get("profile_timeline"), Mapping):
            data["b300_timeline"] = dict(record["profile_timeline"])
    return ProfileResult(
        summary=summary_path.read_text(encoding="utf-8").strip(), data=data
    )


def _reference_source(state: Mapping[str, Any], run_dir: Path) -> str:
    """Read the immutable task source used by validation, when available."""
    run_config_value = str(state.get("paths", {}).get("run_config", ""))
    if not run_config_value:
        return ""
    try:
        payload = json.loads(_state_path(run_dir, run_config_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ConfigurationError):
        return ""
    return str(payload.get("ref_arch_src", ""))[:60_000] if isinstance(payload, Mapping) else ""


def _compile_check_command(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    island_dir: Path,
    candidate_path: Path,
) -> str:
    command_value = config.get("evaluator_command", ())
    if not isinstance(command_value, (list, tuple)) or not command_value:
        return ""
    replacements = {
        "{candidate}": str(candidate_path.resolve()),
        "{baseline}": str(candidate_path.resolve()),
        "{run_dir}": str(run_dir.resolve()),
        "{island_dir}": str(island_dir.resolve()),
        "{device}": str(config.get("device", "cuda:0")),
        "{measurement_mode}": str(config.get("measurement_mode", "wall-clock")),
        "{custom_tests}": str(config.get("custom_tests", "")),
        "{iteration}": "0",
        "{island}": "0",
        "{run_id}": str(run_dir.name),
    }
    command: list[str] = []
    for value in command_value:
        rendered = str(value)
        for marker, replacement in replacements.items():
            rendered = rendered.replace(marker, replacement)
        command.append(rendered)
    if "--artifact-dir" in command:
        index = command.index("--artifact-dir") + 1
        if index < len(command):
            command[index] = str((island_dir / "compile_check_artifacts").resolve())
    command.append("--compile-check")
    return f"cd {shlex.quote(str(run_dir.resolve()))} && {shlex.join(command)}"


def _relative(run_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(run_dir.resolve()))
    except ValueError as exc:
        raise ConfigurationError(f"Run artifact escaped run directory: {resolved}") from exc


def _state_path(run_dir: Path, value: str) -> Path:
    if not value:
        raise ConfigurationError("Run state contains an empty artifact path")
    path = (run_dir / value).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"Run state path escaped run directory: {value}") from exc
    return path


def _island_record(state: Mapping[str, Any], iteration: int, island: int) -> dict[str, Any]:
    iterations = state.get("iterations", {})
    record = iterations.get(str(iteration)) if isinstance(iterations, Mapping) else None
    if not isinstance(record, Mapping):
        raise InvalidTransitionError(f"Iteration {iteration} has not been prepared")
    islands = record.get("islands", {})
    value = islands.get(str(island)) if isinstance(islands, Mapping) else None
    if not isinstance(value, dict):
        raise InvalidTransitionError(f"Island {island} does not exist in iteration {iteration}")
    return value


def _require_current_authoring(state: Mapping[str, Any], iteration: int) -> None:
    current = int(state["current_iteration"])
    if int(iteration) != current:
        raise InvalidTransitionError(f"Can only submit to current iteration {current}")
    if state["phase"] != RunPhase.AUTHORING.value:
        raise InvalidTransitionError(
            f"Candidates can only be submitted during authoring; phase is {state['phase']}"
        )


def _task_from_packet(packet: Mapping[str, Any]) -> AuthoringTask:
    idea = packet.get("idea", {})
    return AuthoringTask(
        run_id=str(packet["run_id"]),
        backend=str(packet["backend"]),
        iteration=int(packet["iteration"]),
        island=int(packet["island"]),
        task_file=Path(str(packet["task_file"])),
        candidate_path=Path(str(packet["candidate_path"])),
        editable_files=tuple(Path(str(value)) for value in packet.get("editable_files", [])),
        readable_files=tuple(Path(str(value)) for value in packet.get("readable_files", [])),
        idea_id=str(idea.get("id", "")) if isinstance(idea, Mapping) else "",
        idea_summary=str(idea.get("summary", "")) if isinstance(idea, Mapping) else "",
        prompt_context_file=(
            Path(str(packet["prompt_context_file"]))
            if packet.get("prompt_context_file")
            else None
        ),
    )


def _next_action(state: Mapping[str, Any], runs_dir: Path) -> str:
    phase = RunPhase(state["phase"])
    run_id = state["run_id"]
    iteration = int(state["current_iteration"])
    identity = f"--runs-dir {shlex.quote(str(runs_dir))} --run-id {run_id}"
    if phase is RunPhase.READY:
        return f"kernel-evo iter prepare {identity}"
    if phase is RunPhase.AUTHORING:
        return f"author each island candidate, then kernel-evo iter evaluate {identity}"
    if phase is RunPhase.EVALUATING:
        return f"resume kernel-evo iter evaluate {identity}"
    if phase is RunPhase.EVALUATED:
        pending_reviews = iteration_report(state, iteration).get(
            "pending_profile_reviews", []
        )
        if pending_reviews:
            return (
                f"kernel-evo iter review-profiles {identity} --iter {iteration}; "
                "submit every returned review with kernel-evo island review-submit"
            )
        record = state.get("iterations", {}).get(str(iteration), {})
        islands = record.get("islands", {}) if isinstance(record, Mapping) else {}
        repairable = []
        if isinstance(islands, Mapping):
            repair_limit = int(state.get("config", {}).get("max_repairs_per_island", 1) or 0)
            for key, value in islands.items():
                if not isinstance(value, Mapping):
                    continue
                if is_repairable_result(
                    value.get("result", {}),
                    int(value.get("repair_count", 0) or 0),
                    repair_limit,
                ):
                    repairable.append(int(key))
        repair_hint = ""
        if repairable:
            repair_hint = (
                "; optional bounded repair: kernel-evo island repair "
                f"{identity} --iter {iteration} --island {repairable[0]}"
            )
        return (
            f"kernel-evo iter report {identity} --iter {iteration}{repair_hint}; "
            f"then kernel-evo iter advance {identity}"
        )
    return "run complete"
