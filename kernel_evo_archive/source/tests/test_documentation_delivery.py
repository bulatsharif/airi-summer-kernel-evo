from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from kernel_evo.agent import EvaluationResult, KernelEvoAgent
from kernel_evo.agent.config import AgentRunConfig
from kernel_evo.agent.errors import ConfigurationError
from kernel_evo.cute_harness.ablation import documentation_bundle
from kernel_evo.cute_harness.b300 import discover_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "experiments" / "cute_ablation" / "study-iter.yaml"


def passing_evaluator(_) -> EvaluationResult:
    return EvaluationResult(
        compiled=True, correctness=True, valid=True, speedup=1.0, fitness=1.0, status="passed"
    )


def b300_config(*, delivery: str, tier: str = "errors") -> AgentRunConfig:
    config = AgentRunConfig.from_file(REPO_ROOT / "examples" / "agent" / "airi_cute_b300.yaml")
    return replace(
        config,
        documentation_delivery=delivery,
        documentation_tier=tier,
        b300_seed="starter",
        seed_preflight=False,
    )


def prepared(tmp_path: Path, *, delivery: str, tier: str = "errors"):
    run_id = f"delivery-{delivery}-{tier}"
    controller = KernelEvoAgent(tmp_path / run_id, evaluator=passing_evaluator)
    controller.init_run(b300_config(delivery=delivery, tier=tier), run_id=run_id)
    return controller, run_id, controller.prepare_iteration(run_id)[0]


def load_run_iter_matrix():
    path = REPO_ROOT / "experiments" / "cute_ablation" / "run_iter_matrix.py"
    spec = importlib.util.spec_from_file_location("run_iter_matrix_delivery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_files_delivery_lists_the_tier_and_injects_nothing(tmp_path: Path) -> None:
    _, _, task = prepared(tmp_path, delivery="files")

    names = {path.name for path in task.readable_files}
    assert "02-language-compilation-and-launch.md" in names
    assert "08-blackwell-architecture.md" in names
    assert task.prompt_context_file is None
    assert not (task.task_file.parent / "DOCUMENTATION.md").exists()


def test_prompt_delivery_hands_over_the_bundle_and_lists_no_tier_file(tmp_path: Path) -> None:
    _, _, task = prepared(tmp_path, delivery="prompt")
    spec = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    bundle = documentation_bundle(spec, "errors")

    assert task.prompt_context_file is not None
    assert task.prompt_context_file.read_text(encoding="utf-8") == bundle.text

    # Not readable: re-reading what it was handed is exactly the cost being removed.
    names = {path.name for path in task.readable_files}
    assert "DOCUMENTATION.md" not in names
    assert "02-language-compilation-and-launch.md" not in names
    assert "08-blackwell-architecture.md" not in names
    assert not any(
        path.is_relative_to(task.task_file.parents[2].parent / "documentation")
        for path in task.readable_files
    )
    task_text = task.task_file.read_text(encoding="utf-8")
    assert "given to you in full at the start of this session" in task_text


def test_both_deliveries_carry_identical_documentation(tmp_path: Path) -> None:
    """A delivery difference must not smuggle in a content difference."""
    spec = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    bundle = documentation_bundle(spec, "errors")

    files_controller, files_run, files_task = prepared(tmp_path, delivery="files")
    _, _, prompt_task = prepared(tmp_path, delivery="prompt")

    documentation = files_controller.store.run_dir(files_run) / "documentation"
    delivered_as_files = "\n\n".join(
        f"# File: {path.name}\n\n{path.read_text(encoding='utf-8')}"
        for path in files_task.readable_files
        if path.is_relative_to(documentation)
    ).rstrip() + "\n"
    assert delivered_as_files == bundle.text
    assert prompt_task.prompt_context_file.read_text(encoding="utf-8") == bundle.text


def test_delivery_is_recorded_for_analysis(tmp_path: Path) -> None:
    for delivery in ("files", "prompt"):
        controller, run_id, task = prepared(tmp_path, delivery=delivery)
        context = controller.store.read_state(run_id)["iterations"]["1"]["islands"]["0"][
            "cute_context"
        ]
        assert context["documentation_delivery"] == delivery
        assert context["documentation_tier"] == "errors"
        assert context["documentation_tokens_cl100k"] > 15_000
        packet = json.loads((task.task_file.parent / "packet.json").read_text(encoding="utf-8"))
        assert bool(packet["prompt_context_file"]) is (delivery == "prompt")


def test_prompt_delivery_still_cannot_reach_above_its_tier(tmp_path: Path) -> None:
    controller, run_id, task = prepared(tmp_path, delivery="prompt", tier="docs")

    text = task.prompt_context_file.read_text(encoding="utf-8")
    assert "# File: 08-blackwell-architecture.md" in text
    # Tier 3 and 4 material is in no delivered file and in no injected text.
    assert "01-diagnostic-workflow.md" not in text
    assert "05-mma-epilogue-and-numerical-patterns.md" not in text
    run_dir = controller.store.run_dir(run_id).resolve()
    assert all(path.is_relative_to(run_dir) for path in task.readable_files)


def test_bare_tier_prompt_delivery_carries_only_the_task_statement(tmp_path: Path) -> None:
    _, _, task = prepared(tmp_path, delivery="prompt", tier="bare")

    text = task.prompt_context_file.read_text(encoding="utf-8")
    assert text.startswith("# File: TASK.md")
    assert "# File: 01-core-api-scope.md" not in text


def test_unknown_delivery_is_rejected() -> None:
    valid = AgentRunConfig.from_file(REPO_ROOT / "examples" / "agent" / "airi_cute_b300.yaml")
    with pytest.raises(ConfigurationError, match="documentation_delivery"):
        replace(valid, documentation_delivery="telepathy").validate()


def test_prompt_delivery_is_refused_where_it_would_silently_do_nothing(tmp_path: Path) -> None:
    """Only the CuTe task bundle is injectable; elsewhere the flag must not no-op."""
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
    kernelbench = AgentRunConfig(baseline=str(baseline), backend="triton")

    replace(kernelbench, documentation_delivery="files").validate()
    with pytest.raises(ConfigurationError, match="cute_b300"):
        replace(kernelbench, documentation_delivery="prompt").validate()


def test_a_results_root_refuses_a_second_protocol(tmp_path: Path) -> None:
    module = load_run_iter_matrix()
    frozen = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    root = tmp_path / "results"
    root.mkdir()

    module.check_protocol_matches(root, frozen)  # nothing written yet
    (root / "study.yaml").write_text(yaml.safe_dump(frozen), encoding="utf-8")
    module.check_protocol_matches(root, frozen)  # a plain resume

    # Adding replications or narrowing tiers still resumes the same protocol.
    widened = {**frozen, "tiers": ["errors"], "agent": {**frozen["agent"], "replications": 5}}
    module.check_protocol_matches(root, widened)

    for changed in (
        {**frozen, "documentation": {"delivery": "prompt"}},
        {**frozen, "profiling": {"enabled": True}},
        {**frozen, "feedback": {"critic": True}},
        {**frozen, "model": "some/other-model"},
    ):
        with pytest.raises(ValueError, match="different protocol"):
            module.check_protocol_matches(root, changed)


def test_default_is_files_so_the_frozen_protocol_is_unchanged() -> None:
    assert AgentRunConfig().documentation_delivery == "files"

    module = load_run_iter_matrix()
    base = {
        "agent": {
            "steps": 6,
            "islands": 1,
            "b300_seed": "starter",
            "max_repairs_per_island": 0,
        },
        "evaluator": {"seed": 0, "warmup": 5, "repeats": 50, "timeout_seconds": 900},
    }
    task_path = REPO_ROOT / "tasks" / "cute" / "tasks" / "level1_01_square_matrix_multiplication_fp8"

    assert module.agent_config(base, "bare", task_path)["run"]["documentation_delivery"] == "files"
    switched = {**base, "documentation": {"delivery": "prompt"}}
    assert module.agent_config(switched, "bare", task_path)["run"]["documentation_delivery"] == "prompt"

    frozen = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    assert module.agent_config(frozen, "bare", task_path)["run"]["documentation_delivery"] == "files"


def test_runner_prepends_the_bundle_to_the_authoring_prompt(tmp_path: Path) -> None:
    module = load_run_iter_matrix()
    _, _, files_task = prepared(tmp_path, delivery="files")
    _, _, prompt_task = prepared(tmp_path, delivery="prompt")

    plain = module.authoring_prompt(files_task)
    injected = module.authoring_prompt(prompt_task)

    assert plain == module.PROMPT.format(task_file=files_task.task_file)
    assert injected.endswith(module.PROMPT.format(task_file=prompt_task.task_file))
    assert "# File: 08-blackwell-architecture.md" in injected
    assert len(injected) > len(plain) + 50_000

    # The whole bundle has to survive as one argv string.
    assert len(injected.encode("utf-8")) < 512_000
