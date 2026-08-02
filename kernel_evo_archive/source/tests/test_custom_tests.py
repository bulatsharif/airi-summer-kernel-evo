from __future__ import annotations

from types import SimpleNamespace

from kernel_evo.agent import AgentRunConfig
from kernel_evo.agent.evaluator import CommandEvaluator
from kernel_evo.agent.models import EvaluationContext
from kernel_evo.core.eval.custom_tests import run_custom_test_file


def test_custom_test_config_resolves_evaluation_path(tmp_path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: pass\n", encoding="utf-8")
    suite = tmp_path / "checks.py"
    suite.write_text("def run_tests(context): return None\n", encoding="utf-8")

    config = AgentRunConfig.from_mapping(
        {
            "problem": {"baseline": "candidate.py"},
            "evaluation": {"custom_tests": "checks.py"},
        },
        base_dir=tmp_path,
    )

    assert config.custom_tests == str(suite.resolve())


def test_custom_test_runner_supports_assertions_and_structured_metrics(tmp_path) -> None:
    suite = tmp_path / "checks.py"
    suite.write_text(
        "def run_tests(context):\n"
        "    assert context.atol == 0.25\n"
        "    return [{'name': 'norm', 'passed': True, 'max_abs_error': 0.125}]\n",
        encoding="utf-8",
    )

    result = run_custom_test_file(suite, SimpleNamespace(atol=0.25))

    assert result["passed"] is True
    assert result["tests"] == [
        {"name": "norm", "passed": True, "max_abs_error": 0.125}
    ]


def test_custom_test_runner_turns_arbitrary_assertion_into_failure(tmp_path) -> None:
    suite = tmp_path / "checks.py"
    suite.write_text(
        "def run_tests(context):\n    assert False, 'norm invariant failed'\n",
        encoding="utf-8",
    )

    result = run_custom_test_file(suite, SimpleNamespace())

    assert result["passed"] is False
    assert "norm invariant failed" in result["error"]


def test_command_evaluator_expands_custom_tests_placeholder(tmp_path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("class ModelNew: pass\n", encoding="utf-8")
    suite = tmp_path / "checks.py"
    suite.write_text("def run_tests(context): return None\n", encoding="utf-8")
    command = (
        "python",
        "-c",
        "import json,sys; print(json.dumps({'compiled':1,'correctness':1,'is_valid':1,'path':sys.argv[1]}))",
        "{custom_tests}",
    )
    context = EvaluationContext(
        run_id="custom",
        iteration=1,
        island=0,
        run_dir=tmp_path,
        island_dir=tmp_path,
        candidate_path=candidate,
        baseline_path=candidate,
        problem_dir=None,
        config={"custom_tests": str(suite)},
        run_config={},
    )

    result = CommandEvaluator(command).evaluate(context)

    assert result.metadata["path"] == str(suite)


def test_command_evaluator_exposes_measurement_mode(tmp_path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("class ModelNew: pass\n", encoding="utf-8")
    command = (
        "python",
        "-c",
        (
            "import json,os,sys; "
            "print(json.dumps({'compiled':1,'correctness':1,'is_valid':1,"
            "'arg':sys.argv[1],'env':os.environ['KERNELEVO_MEASUREMENT_MODE']}))"
        ),
        "{measurement_mode}",
    )
    context = EvaluationContext(
        run_id="measurement",
        iteration=1,
        island=0,
        run_dir=tmp_path,
        island_dir=tmp_path,
        candidate_path=candidate,
        baseline_path=candidate,
        problem_dir=None,
        config={"measurement_mode": "device-time"},
        run_config={},
    )

    result = CommandEvaluator(command).evaluate(context)

    assert result.metadata["arg"] == "device-time"
    assert result.metadata["env"] == "device-time"
