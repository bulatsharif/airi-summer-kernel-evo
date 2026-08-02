from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from kernel_evo.agent import KernelEvoAgent
from kernel_evo.agent.config import AgentRunConfig
from kernel_evo.agent.models import EvaluationResult
from kernel_evo.cute_harness.b300 import (
    EVALUATOR_MARKER,
    EvaluationConfig,
    assemble_submission,
    baseline_candidate,
    discover_tasks,
    evolution_task_description,
    starter_candidate,
)
from kernel_evo.cute_harness.ablation import (
    DOCUMENTATION_TIERS,
    documentation_bundle,
    materialize_bundle,
)
from kernel_evo.cute_harness.b300_policy import check_candidate


def _passing_evaluator(_) -> EvaluationResult:
    return EvaluationResult(
        compiled=True,
        correctness=True,
        valid=True,
        runtime_us=100.0,
        ref_runtime_us=100.0,
        speedup=1.0,
        fitness=1.0,
        status="passed",
    )


def test_all_airi_tasks_have_valid_modelnew_baselines() -> None:
    tasks = discover_tasks()
    assert len(tasks) == 10
    for task in tasks.values():
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "submission.py"
            candidate.write_text(baseline_candidate(task), encoding="utf-8")
            report = check_candidate(candidate, task.policy)
            assembled = assemble_submission(
                task,
                candidate,
                EvaluationConfig(seed=0, warmup=1, repeats=1),
            )
        assert report.passed, report.errors
        assert "class ModelNew:" in assembled
        evaluator = assembled.split(EVALUATOR_MARKER, 1)[1]
        assert task.entrypoint in evaluator
        assert "ModelNew.forward" not in evaluator


def test_airi_config_prepares_one_isolated_island(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = AgentRunConfig.from_file(root / "examples" / "agent" / "airi_cute_b300.yaml")
    assert config.evaluator_kind == "cute_b300"
    assert config.islands == 1
    assert config.measurement_mode == "device-time"

    controller = KernelEvoAgent(
        tmp_path / "runs",
        evaluator=_passing_evaluator,
    )
    controller.init_run(config, run_id="airi-smoke")
    tasks = controller.prepare_iteration("airi-smoke")

    assert len(tasks) == 1
    assert tasks[0].candidate_path.name == "submission.py"
    assert "class ModelNew:" in tasks[0].candidate_path.read_text(encoding="utf-8")
    assert any(
        path.name == "02-language-compilation-and-launch.md"
        for path in tasks[0].readable_files
    )
    assert any(path.name == "01-diagnostic-workflow.md" for path in tasks[0].readable_files)
    assert all("__pycache__" not in path.parts for path in tasks[0].readable_files)
    rules = (tasks[0].task_file.parent / "RULES.md").read_text(encoding="utf-8")
    assert "kernel-evo cute task-check" in rules


def test_airi_bare_packet_has_only_the_task_document(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = AgentRunConfig.from_file(root / "examples" / "agent" / "airi_cute_b300.yaml")
    controller = KernelEvoAgent(
        tmp_path / "runs",
        evaluator=_passing_evaluator,
    )
    controller.init_run(config, run_id="airi-bare")
    task = controller.prepare_iteration("airi-bare", documentation_tier="bare")[0]

    readable_names = {path.name for path in task.readable_files}
    assert "TASK.md" in readable_names
    assert "01-core-api-scope.md" not in readable_names
    assert not any(name.startswith("01-") for name in readable_names)
    assert "SKILL.md" not in readable_names
    assert "CUTE_HARNESS.md" not in readable_names
    rules = (task.task_file.parent / "RULES.md").read_text(encoding="utf-8")
    assert "task statement and supplied candidate" in rules
    assert controller.store.read_state("airi-bare")["config"]["documentation_enabled"] is False


def test_airi_bare_evolution_prompt_keeps_only_the_task_document() -> None:
    task = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    prompt = task.prompt_path.read_text(encoding="utf-8")
    core_api = (
        task.skill_paths[0]
        / "tiers"
        / "tier-2-foundations"
        / "02-language-compilation-and-launch.md"
    ).read_text(encoding="utf-8")
    foundations = (
        task.skill_paths[0] / "tiers" / "tier-2-foundations" / "07-layout-reasoning.md"
    ).read_text(encoding="utf-8")

    documented = evolution_task_description(task, documentation_enabled=True)
    disabled = evolution_task_description(task, documentation_enabled=False)
    bare = evolution_task_description(
        task,
        documentation_enabled=True,
        documentation_tier="bare",
    )

    assert prompt in disabled
    assert core_api not in disabled
    assert foundations not in disabled
    assert core_api not in bare
    assert foundations in documented
    assert foundations not in bare


def test_from_scratch_run_hands_the_author_the_skeleton_not_the_solution(tmp_path: Path) -> None:
    """The whole point of the study: the model writes the kernel, it is not given one."""
    root = Path(__file__).resolve().parents[1]
    task_spec = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    config = AgentRunConfig.from_file(root / "examples" / "agent" / "airi_cute_b300.yaml")
    config = replace(config, b300_seed="starter", seed_preflight=False)

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=_passing_evaluator)
    controller.init_run(config, run_id="from-scratch")
    task = controller.prepare_iteration("from-scratch")[0]

    candidate = task.candidate_path.read_text(encoding="utf-8")
    assert candidate == starter_candidate(task_spec)
    assert "pass" in candidate
    assert "TODO" in candidate

    # The skeleton supplies imports and constants; the kernel bodies are the task.
    baseline = baseline_candidate(task_spec)
    assert candidate != baseline
    assert len(candidate.splitlines()) < len(baseline.splitlines()) // 4
    assert "MMA_TILER_MNK" not in candidate

    # The contract the author must satisfy is still present.
    assert "class ModelNew:" in candidate
    assert task_spec.entrypoint in candidate


def test_no_tier_leaks_task_specific_answers() -> None:
    """General documentation may teach mechanics but not an evaluated task's answer."""
    tuning_name = re.compile(r"^(\w*(?:TILER|STAGES|THREADS|WARP|CLUSTER|SWIZZLE)\w*)\s*=\s*(.+)$")
    tasks = discover_tasks()
    for task in tasks.values():
        # This task's own tuning answer, taken from its verified reference.
        answers = {
            f"{match.group(1)} = {match.group(2)}"
            for line in baseline_candidate(task).splitlines()
            if (match := tuning_name.match(line.strip()))
        }
        assert answers, f"{task.id}: no tuning constants found; guard would be vacuous"
        for tier in DOCUMENTATION_TIERS:
            bundle = documentation_bundle(task, tier)
            general_files = bundle.files[1:]
            general_text = "\n".join(path.read_text(encoding="utf-8") for path in general_files)
            for answer in answers:
                assert answer not in general_text, f"{task.id}/{tier} leaks `{answer}`"
            assert all(path.suffix == ".md" for path in general_files)
            assert all("references" not in path.parts for path in general_files)
            assert all(other.id not in general_text for other in tasks.values())
            assert all(str(other.data["title"]) not in general_text for other in tasks.values())
            assert not re.search(r"https?://|www\.", general_text)


def test_prepared_documentation_is_isolated_inside_the_run(tmp_path: Path) -> None:
    """An author must not be able to reach documentation above its own tier."""
    root = Path(__file__).resolve().parents[1]
    config = AgentRunConfig.from_file(root / "examples" / "agent" / "airi_cute_b300.yaml")
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=_passing_evaluator)
    controller.init_run(config, run_id="airi-isolated")
    task = controller.prepare_iteration("airi-isolated", documentation_tier="docs")[0]

    run_dir = (tmp_path / "runs" / "airi-isolated").resolve()
    assert all(path.is_relative_to(run_dir) for path in task.readable_files), [
        str(path) for path in task.readable_files if not path.is_relative_to(run_dir)
    ]
    assert all(path.is_file() for path in task.readable_files)

    readable_names = {path.name for path in task.readable_files}
    assert "02-language-compilation-and-launch.md" in readable_names
    assert "08-blackwell-architecture.md" in readable_names
    assert "01-language-and-launch-patterns.md" not in readable_names
    assert "01-diagnostic-workflow.md" not in readable_names

    shared = root / "tasks" / "cute" / "opencode" / ".opencode" / "skills" / "cute-fp8-kernels"
    assert not any(path.is_relative_to(shared.resolve()) for path in task.readable_files)


def test_materialized_bundle_is_idempotent(tmp_path: Path) -> None:
    task = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    first = materialize_bundle(task, "errors", tmp_path / "docs")
    stamps = {path: path.stat().st_mtime_ns for path in first.files}
    second = materialize_bundle(task, "errors", tmp_path / "docs")

    assert first.files == second.files
    assert first.tokens_cl100k == second.tokens_cl100k
    assert {path: path.stat().st_mtime_ns for path in second.files} == stamps


def test_documentation_ablation_is_cumulative_and_starter_is_incomplete() -> None:
    task = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    bundles = [documentation_bundle(task, tier) for tier in DOCUMENTATION_TIERS]

    assert not (task.skill_paths[0] / "tiers" / "tier-1-bare").exists()
    assert [bundle.tokens_cl100k for bundle in bundles] == sorted(bundle.tokens_cl100k for bundle in bundles)
    assert [path.name for path in bundles[0].files] == ["TASK.md"]
    assert "01-core-api-scope.md" in {path.name for path in bundles[1].files}
    assert "08-blackwell-architecture.md" in {path.name for path in bundles[1].files}
    assert "12-low-precision-numerics.md" in {path.name for path in bundles[1].files}
    assert "05-mma-epilogue-and-numerical-patterns.md" in {
        path.name for path in bundles[2].files
    }
    assert not any(path.name == "01-diagnostic-workflow.md" for path in bundles[2].files)
    assert bundles[3].files[-1].name == "06-numerical-and-performance-errors.md"
    assert len(bundles[0].files) == 1
    assert len(bundles[1].files) == 14
    # tier-3 gained 06-complete-worked-kernels.md; tier-4 inherits it.
    assert len(bundles[2].files) == 21
    assert len(bundles[3].files) == 27
    assert bundles[1].tokens_cl100k > 10_000
    assert bundles[3].tokens_cl100k > 15_000
    # tier-3 now carries a verified single-tile FP8 GEMM building block, which
    # uses the same epilogue-tiling idiom. Guard the task's own answers instead:
    # its shapes and its dequantization scales, none of which the example holds.
    assert "232,448" not in bundles[-1].text
    assert "M = 1024" not in bundles[-1].text
    assert "WEIGHT_BOUND" not in bundles[-1].text
    assert "SCALE_A = 1.0 / FP8_MAX" not in bundles[-1].text
    assert "MMA_TILER_MNK = (128, 256, 128)" not in bundles[-1].text
    assert "pass" in starter_candidate(task)
    assert "class ModelNew:" in starter_candidate(task)


def test_every_task_receives_the_same_general_documentation() -> None:
    tasks = list(discover_tasks().values())
    assert not list(tasks[0].directory.parent.glob("*/TASK_REFERENCE.md"))
    assert all("references" not in task.data for task in tasks)
    for tier in DOCUMENTATION_TIERS:
        expected = [
            (path.name, path.read_bytes())
            for path in documentation_bundle(tasks[0], tier).files[1:]
        ]
        assert all(
            [(path.name, path.read_bytes()) for path in documentation_bundle(task, tier).files[1:]]
            == expected
            for task in tasks[1:]
        )


def test_b300_policy_requires_the_manifest_jit_entrypoint(tmp_path: Path) -> None:
    task = discover_tasks()["level1_02_vector_scale_fp4"]
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        baseline_candidate(task).replace(
            "@cute.jit\ndef scale_fp4(",
            "def scale_fp4(",
            1,
        ),
        encoding="utf-8",
    )

    report = check_candidate(candidate, task.policy)

    assert not report.passed
    assert any("@cute.jit def scale_fp4" in error for error in report.errors)
