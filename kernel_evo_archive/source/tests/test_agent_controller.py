from __future__ import annotations

from pathlib import Path

import pytest

from kernel_evo.agent import (
    ConfigurationError,
    EvaluationResult,
    InvalidTransitionError,
    KernelEvoAgent,
)


class DeterministicEvaluator:
    def evaluate(self, context):
        source = context.candidate_path.read_text(encoding="utf-8")
        score = 1.1 + context.island * 0.2 + source.count("optimized") * 0.01
        return EvaluationResult.from_metrics(
            {
                "compiled": 1,
                "correctness": 1,
                "is_valid": 1,
                "runtime_us": 10.0 / score,
                "ref_runtime_us": 10.0,
                "speedup": score,
                "fitness": score,
            }
        )


def test_evaluated_run_can_be_extended_without_rewriting_archive(tmp_path: Path) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())
    controller.init_run(
        {"baseline": str(baseline), "backend": "triton", "steps": 1, "islands": 1},
        run_id="extendable",
    )
    controller.prepare_iteration("extendable")
    controller.evaluate_iteration("extendable")
    archive_before = controller.store.read_state("extendable")["archive"]["entries"]
    status = controller.extend_run("extendable", 2)
    assert status["steps"] == 3
    assert status["phase"] == "evaluated"
    assert controller.store.read_state("extendable")["archive"]["entries"] == archive_before
    advanced = controller.advance_iteration("extendable")
    assert advanced["current_iteration"] == 2
    assert advanced["phase"] == "ready"


def test_extended_run_can_be_steered_without_rewriting_archive(tmp_path: Path) -> None:
    baseline = tmp_path / "kernel.py"
    reference = tmp_path / "reference.py"
    baseline.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
    reference.write_text("# reference\n", encoding="utf-8")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())
    controller.init_run(
        {"baseline": str(baseline), "backend": "triton", "steps": 1, "islands": 1},
        run_id="steered",
    )
    controller.prepare_iteration("steered")
    controller.evaluate_iteration("steered")
    archive_before = controller.store.read_state("steered")["archive"]["entries"]
    controller.extend_run(
        "steered",
        1,
        ideas=[{"id": "recurrent", "summary": "Optimize recurrence."}],
        author_readable_files=[str(reference)],
    )
    state = controller.store.read_state("steered")
    assert state["archive"]["entries"] == archive_before
    assert state["config"]["ideas"] == [
        {"id": "recurrent", "summary": "Optimize recurrence."}
    ]
    assert str(reference) in state["config"]["author_readable_files"]


def test_barrier_run_prepares_isolated_packets_evaluates_and_advances(tmp_path: Path) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())

    initialized = controller.init_run(
        {
            "name": "test-agent",
            "baseline": str(baseline),
            "backend": "triton",
            "steps": 2,
            "islands": 2,
        },
        run_id="visible-run",
    )
    assert initialized["phase"] == "ready"

    tasks = controller.prepare_iteration("visible-run")
    assert len(tasks) == 2
    assert tasks[0].candidate_path != tasks[1].candidate_path
    assert tasks[0].editable_files == (tasks[0].candidate_path,)
    assert tasks[0].task_file.exists()
    assert "Do not run the full benchmark" in tasks[0].task_file.read_text(encoding="utf-8")

    for task in tasks:
        task.candidate_path.write_text(
            task.candidate_path.read_text(encoding="utf-8") + f"# optimized island {task.island}\n",
            encoding="utf-8",
        )
    controller.submit_candidate(
        "visible-run",
        1,
        0,
        tasks[0].candidate_path,
        metadata={"idea_summary": "bounded change"},
    )
    with pytest.raises(InvalidTransitionError):
        controller.advance_iteration("visible-run")

    report = controller.evaluate_iteration("visible-run")
    assert report["valid_candidates"] == 2
    assert report["promoted_candidates"] == 2
    assert report["global_best"]["id"] == "iter-001-island-1"
    assert controller.status("visible-run")["phase"] == "evaluated"
    assert "| Island |" in controller.report_iteration("visible-run")

    advanced = controller.advance_iteration("visible-run")
    assert advanced["phase"] == "ready"
    assert advanced["current_iteration"] == 2
    second_tasks = controller.prepare_iteration("visible-run")
    assert "optimized island 0" in second_tasks[0].candidate_path.read_text(encoding="utf-8")
    assert "optimized island 1" in second_tasks[1].candidate_path.read_text(encoding="utf-8")


def test_prepare_is_idempotent_and_context_is_machine_readable(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())
    controller.init_run(
        {"baseline": str(baseline), "backend": "triton", "islands": 1},
        run_id="idempotent",
    )

    first = controller.prepare_iteration("idempotent")
    second = controller.prepare_iteration("idempotent")
    assert first[0].to_dict() == second[0].to_dict()
    packet = controller.island_context("idempotent", 1, 0)
    assert packet["role"] == "kernel-author"
    assert packet["status"] == "awaiting_author"
    assert packet["candidate_path"] == str(first[0].candidate_path)


def test_failed_candidate_is_archived_without_aborting_other_islands(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        if context.island == 0:
            raise RuntimeError("compile error")
        return {"compiled": 1, "correctness": 1, "is_valid": 1, "speedup": 1.25, "fitness": 1.25}

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {"baseline": str(baseline), "backend": "triton", "islands": 2},
        run_id="failure-run",
    )
    controller.prepare_iteration("failure-run")
    report = controller.evaluate_iteration("failure-run")

    assert report["valid_candidates"] == 1
    assert report["islands"][0]["error"] == "compile error"
    assert report["islands"][1]["valid"] is True
    assert controller.status("failure-run")["archive_size"] == 2


def test_problem_path_reuses_shared_seed_and_validation_preparation(tmp_path: Path) -> None:
    problem = tmp_path / "task.py"
    problem.write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n"
        "def get_inputs():\n"
        "    return [torch.randn(8)]\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())
    initialized = controller.init_run(
        {"problem_path": str(problem), "backend": "triton", "islands": 1},
        run_id="prepared-problem",
    )

    run_dir = Path(initialized["run_dir"])
    assert (run_dir / "problem" / "run_config.json").exists()
    assert "class ModelNew" in (run_dir / "seed" / "seed.py").read_text(encoding="utf-8")
    assert "Backend compliance" in (run_dir / "problem" / "task_description.txt").read_text(encoding="utf-8")


def test_seed_preflight_blocks_broken_packaged_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        assert context.iteration == 0
        return {"compiled": 0, "correctness": 0, "is_valid": 0, "error": "missing asset"}

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    with pytest.raises(ConfigurationError, match="Packaged seed preflight failed.*missing asset"):
        controller.init_run(
            {
                "baseline": str(baseline),
                "backend": "triton",
                "islands": 1,
                "seed_preflight": True,
            },
            run_id="broken-seed",
        )


def test_seed_is_incumbent_and_regression_is_not_promoted(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        speedup = 1.0 if context.iteration == 0 else 0.8
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": speedup,
            "fitness": speedup,
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "seed_preflight": True,
        },
        run_id="seed-incumbent",
    )
    controller.prepare_iteration("seed-incumbent")
    report = controller.evaluate_iteration("seed-incumbent")
    assert report["promoted_candidates"] == 0
    assert report["global_best"]["id"] == "seed"


def test_valid_sub_reference_progress_becomes_developmental_parent(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        speedup = 0.885 if context.iteration == 0 else 0.96
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "runtime_us": 1_000.0 / speedup,
            "ref_runtime_us": 1_000.0,
            "speedup": speedup,
            "fitness": 0.0,
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "steps": 2,
            "islands": 1,
            "seed_preflight": True,
        },
        run_id="developmental-parent",
    )
    task = controller.prepare_iteration("developmental-parent")[0]
    task.candidate_path.write_text(
        task.candidate_path.read_text(encoding="utf-8") + "# incremental progress\n",
        encoding="utf-8",
    )
    report = controller.evaluate_iteration("developmental-parent")

    assert report["promoted_candidates"] == 0
    assert report["global_best"]["id"] == "seed"
    state = controller.store.read_state("developmental-parent")
    assert state["archive"]["performance_development_elites"]["0"] == "iter-001-island-0"

    controller.advance_iteration("developmental-parent")
    next_task = controller.prepare_iteration("developmental-parent")[0]
    assert "# incremental progress" in next_task.candidate_path.read_text(encoding="utf-8")


def test_localized_failure_can_be_repaired_once_with_immutable_archive(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        source = context.candidate_path.read_text(encoding="utf-8")
        if "# fixed" not in source:
            return {
                "compiled": 0,
                "correctness": 0,
                "is_valid": 0,
                "error": "localized compile error",
            }
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 1.2,
            "fitness": 1.2,
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "max_repairs_per_island": 1,
        },
        run_id="repairable",
    )
    task = controller.prepare_iteration("repairable")[0]
    first = controller.evaluate_iteration("repairable")
    assert first["repairable_islands"] == [0]

    state = controller.store.read_state("repairable")
    first_entry = state["archive"]["entries"][0]
    snapshot = Path(controller.status("repairable")["run_dir"]) / first_entry["path"]
    original_snapshot = snapshot.read_text(encoding="utf-8")
    assert snapshot != task.candidate_path

    repair = controller.reopen_island_for_repair("repairable", 1, 0)
    assert Path(repair["repair_file"]).is_file()
    task.candidate_path.write_text(
        task.candidate_path.read_text(encoding="utf-8") + "# fixed\n",
        encoding="utf-8",
    )
    controller.submit_candidate("repairable", 1, 0, task.candidate_path)
    repaired = controller.evaluate_iteration("repairable")
    assert repaired["valid_candidates"] == 1
    assert repaired["global_best"]["id"] == "iter-001-island-0-repair-1"
    assert snapshot.read_text(encoding="utf-8") == original_snapshot
    with pytest.raises(InvalidTransitionError, match="not eligible"):
        controller.reopen_island_for_repair("repairable", 1, 0)


def test_changed_after_submission_requires_resubmission(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=DeterministicEvaluator())
    controller.init_run(
        {"baseline": str(baseline), "backend": "triton", "islands": 1},
        run_id="hash-guard",
    )
    task = controller.prepare_iteration("hash-guard")[0]
    controller.submit_candidate("hash-guard", 1, 0, task.candidate_path)
    task.candidate_path.write_text("class ModelNew: pass\n", encoding="utf-8")
    report = controller.evaluate_iteration("hash-guard")
    assert report["valid_candidates"] == 0
    assert "changed after submission" in report["islands"][0]["error"]


def test_profiled_parent_and_bounded_reviewer_feed_agent_authored_idea(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def evaluator(context):
        speedup = 1.0 if context.iteration == 0 else 0.8
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "runtime_us": 12.5,
            "ref_runtime_us": 10.0,
            "speedup": speedup,
            "fitness": speedup,
        }

    profile_calls: list[tuple[int, str]] = []
    profile_artifacts: list[Path] = []

    def profiler(context, result):
        profile_calls.append((context.iteration, context.candidate_path.name))
        trace = context.island_dir / "profile" / "torch" / "trace.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text('{"traceEvents": []}', encoding="utf-8")
        profile_artifacts.append(trace)
        return {
            "torch": {
                "status": "completed",
                "trace_file": str(trace),
                "inner_kernel": {
                    "active_device_time_us": 6.0,
                    "kernel_count": 11.0,
                    "memcpy_count": 1.0,
                },
                "eager_complete_layer": {
                    "end_to_end_us": 12.5,
                    "inferred_dispatch_gaps_us": 6.5,
                },
                "cuda_graph_complete_layer": {"capturable": True, "replay_us": 6.2},
                "graph_capturability_gate": {"passed": True, "failure": ""},
                "optimization_ideas": ["Fuse launch-heavy pointwise work."],
            }
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator, profiler=profiler)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "steps": 2,
            "islands": 1,
            "seed_preflight": True,
            "profile_enabled": True,
            "profile_min_speedup": 1.0,
            "profile_regression_budget": 1,
            "profile_parent_before_use": True,
            "profile_review_required": True,
        },
        run_id="profile-review",
    )
    first_task = controller.prepare_iteration("profile-review")[0]
    assert "PARENT_PROFILE.md" in first_task.task_file.read_text(encoding="utf-8")
    report = controller.evaluate_iteration("profile-review")
    assert report["islands"][0]["profile_reason"] == "bounded_regression"
    assert report["pending_profile_reviews"] == [0]
    assert len(profile_calls) == 2  # seed parent, then valid slower candidate
    next_action = controller.status("profile-review")["next_action"]
    assert "iter review-profiles" in next_action
    assert "iter advance" not in next_action

    with pytest.raises(InvalidTransitionError, match="profile review required"):
        controller.advance_iteration("profile-review")

    review_task = controller.prepare_profile_reviews("profile-review")[0]
    assert str(profile_artifacts[-1].resolve()) not in review_task["readable_files"]
    assert review_task["readable_files"] == [
        str(first_task.candidate_path.resolve()),
        str(Path(review_task["task_file"]).resolve()),
    ]
    review_contract = Path(review_task["task_file"]).read_text(encoding="utf-8")
    assert "Do not inspect raw Torch, Nsight Systems, NCU" in review_contract
    assert "Profiler artifacts" not in review_contract
    review_path = Path(review_task["output_file"])
    review_path.write_text(
        '{"findings":"Dispatch gaps dominate.","ideas":'
        '[{"summary":"Fuse the recurrent epilogue launches.",'
        '"expected_perf_mechanism":"reduce dispatch gaps"}]}',
        encoding="utf-8",
    )
    controller.submit_profile_review("profile-review", 1, 0, review_path)
    controller.advance_iteration("profile-review")
    next_task = controller.prepare_iteration("profile-review")[0]
    idea_text = (next_task.task_file.parent / "IDEA.md").read_text(encoding="utf-8")
    assert "Fuse the recurrent epilogue launches" in idea_text


def test_non_capturable_parent_defers_absolute_graph_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def profiler(context, result):
        return {
            "torch": {
                "status": "completed",
                "inner_kernel": {"active_device_time_us": 4.0},
                "eager_complete_layer": {"end_to_end_us": 8.0},
                "cuda_graph_complete_layer": {"capturable": False},
                "graph_capturability_gate": {
                    "passed": False,
                    "failure": "inherited allocation",
                },
            }
        }

    controller = KernelEvoAgent(
        tmp_path / "runs",
        evaluator=DeterministicEvaluator(),
        profiler=profiler,
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "seed_preflight": True,
            "profile_enabled": True,
            "profile_require_graph_capturable": True,
        },
        run_id="relative-graph-gate",
    )
    controller.prepare_iteration("relative-graph-gate")
    report = controller.evaluate_iteration("relative-graph-gate")
    assert report["valid_candidates"] == 1
    state = controller.store.read_state("relative-graph-gate")
    graph = state["iterations"]["1"]["islands"]["0"]["result"]["metadata"][
        "graph_capturability"
    ]
    assert graph["enforced"] is False
    assert graph["parent_capturable"] is False


def test_graph_regression_can_be_reviewed_then_reopened_for_repair(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    profile_calls = 0

    def profiler(_context, _result):
        nonlocal profile_calls
        profile_calls += 1
        parent_profile = profile_calls == 1
        return {
            "torch": {
                "status": "completed",
                "cuda_graph_complete_layer": {"capturable": parent_profile},
                "graph_capturability_gate": {
                    "passed": parent_profile,
                    "failure": "" if parent_profile else "candidate allocation",
                },
            }
        }

    controller = KernelEvoAgent(
        tmp_path / "runs",
        evaluator=DeterministicEvaluator(),
        profiler=profiler,
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "seed_preflight": True,
            "profile_enabled": True,
            "profile_require_graph_capturable": True,
            "profile_review_required": True,
        },
        run_id="graph-repair",
    )
    controller.prepare_iteration("graph-repair")
    report = controller.evaluate_iteration("graph-repair")

    assert report["islands"][0]["valid"] is False
    assert report["repairable_islands"] == [0]
    assert report["pending_profile_reviews"] == [0]

    review_task = controller.prepare_profile_reviews("graph-repair")[0]
    review_path = Path(review_task["output_file"])
    review_path.write_text(
        '{"findings":"A candidate allocation broke graph capture.","ideas":[]}',
        encoding="utf-8",
    )
    controller.submit_profile_review("graph-repair", 1, 0, review_path)
    assert "island repair" in controller.status("graph-repair")["next_action"]

    controller.reopen_island_for_repair("graph-repair", 1, 0)
    record = controller.store.read_state("graph-repair")["iterations"]["1"]["islands"]["0"]
    assert record["profile_review"] == {}
    assert record["profile_review_task"] == ""


def test_review_required_false_emits_optional_tasks_without_blocking_advance(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def profiler(_context, _result):
        return {"torch": {"status": "completed", "inner_kernel": {}}}

    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=DeterministicEvaluator(), profiler=profiler
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "profile_enabled": True,
            "profile_parent_before_use": False,
            "profile_review_required": False,
        },
        run_id="optional-review",
    )
    controller.prepare_iteration("optional-review")
    report = controller.evaluate_iteration("optional-review")

    assert report["pending_profile_reviews"] == []
    review_tasks = controller.prepare_profile_reviews("optional-review")
    assert len(review_tasks) == 1
    assert review_tasks[0]["role"] == "kernel-profile-reviewer"
    review_prompt = Path(review_tasks[0]["task_file"]).read_text(encoding="utf-8")
    assert "How should this kernel/layer be optimized" in review_prompt
    assert "measured objective: `wall-clock`" in review_prompt
    assert "ranked" in review_prompt
    assert "implementation_location" in review_prompt
    assert "estimated_upside" in review_prompt
    assert "Do not merely summarize the trace" in review_prompt
    controller.advance_iteration("optional-review")


def test_profile_failure_preserves_valid_evaluation_and_checkpoint(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    evaluations = 0

    def evaluator(_context):
        nonlocal evaluations
        evaluations += 1
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "runtime_us": 8.0,
            "ref_runtime_us": 10.0,
            "speedup": 1.25,
            "fitness": 1.25,
        }

    def profiler(_context, _result):
        raise RuntimeError("profiler unavailable")

    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=evaluator, profiler=profiler
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "profile_enabled": True,
            "profile_parent_before_use": False,
            "profile_review_required": False,
        },
        run_id="profile-failure",
    )
    controller.prepare_iteration("profile-failure")
    report = controller.evaluate_iteration("profile-failure")

    assert evaluations == 1
    assert report["valid_candidates"] == 1
    assert report["islands"][0]["profile_status"] == "failed"
    state = controller.store.read_state("profile-failure")
    checkpoint = state["iterations"]["1"]["islands"]["0"][
        "evaluation_checkpoint"
    ]
    assert checkpoint["result"]["valid"] is True


def test_nested_runner_failure_does_not_require_profile_review(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")

    def profiler(_context, _result):
        return {"torch": {"status": "failed", "reason": "GPU lease unavailable"}}

    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=DeterministicEvaluator(), profiler=profiler
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "profile_enabled": True,
            "profile_parent_before_use": False,
            "profile_review_required": True,
        },
        run_id="nested-profile-failure",
    )
    controller.prepare_iteration("nested-profile-failure")
    report = controller.evaluate_iteration("nested-profile-failure")

    assert report["islands"][0]["profile_status"] == "failed"
    assert report["pending_profile_reviews"] == []
    assert controller.prepare_profile_reviews("nested-profile-failure") == []
    assert controller.advance_iteration("nested-profile-failure")["phase"] == "complete"


def test_byte_identical_repair_skips_redundant_profile(tmp_path: Path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    profile_calls = 0

    def evaluator(context):
        # First pass is localized-invalid; the repaired pass is valid even when
        # its source is restored to the exact parent bytes.
        valid = bool(
            controller.store.read_state("duplicate-repair")["iterations"]["1"]
            ["islands"]["0"].get("repair_count")
        )
        return {
            "compiled": 1,
            "correctness": int(valid),
            "is_valid": int(valid),
            "speedup": 1.0,
            "fitness": 1.0,
            "error": "mismatch" if not valid else "",
        }

    def profiler(_context, _result):
        nonlocal profile_calls
        profile_calls += 1
        return {"torch": {"status": "completed"}}

    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=evaluator, profiler=profiler
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "profile_enabled": True,
            "profile_parent_before_use": False,
            "profile_review_required": False,
            "max_repairs_per_island": 1,
            "evaluator_command": [
                "python",
                "evaluate.py",
                "--candidate",
                "{candidate}",
            ],
        },
        run_id="duplicate-repair",
    )
    task = controller.prepare_iteration("duplicate-repair")[0]
    controller.evaluate_iteration("duplicate-repair")
    repair = controller.reopen_island_for_repair("duplicate-repair", 1, 0)
    repair_packet = Path(repair["repair_file"]).read_text(encoding="utf-8")
    assert "bounded compile/execute check" in repair_packet
    assert "--compile-check" in repair_packet
    controller.submit_candidate("duplicate-repair", 1, 0, task.candidate_path)
    report = controller.evaluate_iteration("duplicate-repair")

    assert report["valid_candidates"] == 1
    assert report["islands"][0]["profile_status"] == "skipped_duplicate"
    assert profile_calls == 0


def test_interrupted_optional_profile_resumes_without_reevaluation(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    evaluations = 0
    profiles = 0

    class SimulatedProcessInterruption(BaseException):
        pass

    def evaluator(_context):
        nonlocal evaluations
        evaluations += 1
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "runtime_us": 8.0,
            "ref_runtime_us": 10.0,
            "speedup": 1.25,
            "fitness": 1.25,
        }

    def profiler(_context, _result):
        nonlocal profiles
        profiles += 1
        raise SimulatedProcessInterruption()

    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=evaluator, profiler=profiler
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "profile_enabled": True,
            "profile_parent_before_use": False,
            "profile_review_required": False,
        },
        run_id="interrupted-profile",
    )
    controller.prepare_iteration("interrupted-profile")
    with pytest.raises(SimulatedProcessInterruption):
        controller.evaluate_iteration("interrupted-profile")

    resumed = controller.evaluate_iteration("interrupted-profile")

    assert evaluations == 1
    assert profiles == 1
    assert resumed["valid_candidates"] == 1
    assert resumed["islands"][0]["profile_status"] == "failed"
