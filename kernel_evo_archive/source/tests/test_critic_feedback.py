from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import yaml

from kernel_evo.agent import EvaluationResult, KernelEvoAgent
from kernel_evo.agent.config import AgentRunConfig
from kernel_evo.cute_harness.critic import (
    HINT_MAX_CHARS,
    build_diagnostic,
    critic_task_markdown,
    parse_critic_hints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "experiments" / "cute_ablation" / "study-iter.yaml"


def failing_evaluator(_) -> EvaluationResult:
    return EvaluationResult.failed("cute.gemm: layout mismatch between A and the MMA tiler")


def passing_evaluator(_) -> EvaluationResult:
    return EvaluationResult(
        compiled=True, correctness=True, valid=True, speedup=2.0, fitness=2.0, status="passed"
    )


def b300_config(steps: int = 3) -> AgentRunConfig:
    config = AgentRunConfig.from_file(REPO_ROOT / "examples" / "agent" / "airi_cute_b300.yaml")
    return replace(config, b300_seed="starter", seed_preflight=False, steps=steps)


def transcript(*texts: str) -> str:
    """Shape of `opencode run --format json`: one JSON event per line."""
    lines = [json.dumps({"type": "session.start", "sessionID": "ses_123"})]
    lines.extend(
        json.dumps({"type": "message.part", "part": {"type": "text", "text": text}})
        for text in texts
    )
    return "\n".join(lines) + "\n"


def load_run_iter_matrix():
    path = REPO_ROOT / "experiments" / "cute_ablation" / "run_iter_matrix.py"
    spec = importlib.util.spec_from_file_location("run_iter_matrix_critic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parses_hints_from_a_fenced_json_reply() -> None:
    reply = 'Looking at the error.\n\n```json\n{"hints": ["Use cute.make_layout", "Swap the tiler"]}\n```'

    assert parse_critic_hints(transcript(reply)) == ["Use cute.make_layout", "Swap the tiler"]


def test_parses_hints_from_bare_json_and_a_plain_list() -> None:
    assert parse_critic_hints(transcript('{"hints": ["one thing"]}')) == ["one thing"]
    assert parse_critic_hints(transcript('["a", "b"]')) == ["a", "b"]
    assert parse_critic_hints(transcript('{"hint": "single"}')) == ["single"]
    # Not JSONL at all — a raw transcript still yields its payload.
    assert parse_critic_hints('{"hints": ["raw"]}') == ["raw"]


def test_the_last_payload_wins_and_prose_is_ignored() -> None:
    reply = transcript(
        '{"hints": ["first draft"]}',
        "On reflection that was wrong.",
        '```json\n{"hints": ["final answer"]}\n```',
    )

    assert parse_critic_hints(reply) == ["final answer"]


def test_hints_are_bounded_deduped_and_flattened() -> None:
    reply = transcript(
        json.dumps(
            {
                "hints": [
                    "  spaced   out\n  hint  ",
                    "spaced out hint",
                    "b" * 400,
                    "third",
                    "fourth",
                ]
            }
        )
    )

    hints = parse_critic_hints(reply, limit=3)

    assert len(hints) == 3
    assert hints[0] == "spaced out hint"
    assert all(len(hint) <= HINT_MAX_CHARS for hint in hints)
    assert all("\n" not in hint for hint in hints)


def test_unusable_replies_yield_no_hints() -> None:
    assert parse_critic_hints("") == []
    assert parse_critic_hints(transcript("I could not determine the cause.")) == []
    assert parse_critic_hints(transcript('{"unrelated": true}')) == []
    assert parse_critic_hints(transcript("```json\n{not json}\n```")) == []


def test_diagnostic_carries_the_error_and_log_tails() -> None:
    diagnostic = build_diagnostic(
        result={"valid": False, "compiled": False, "error": "layout mismatch", "speedup": 0.0},
        stderr="Traceback\nDSLRuntimeError: bad layout",
        stdout="task=x FAIL",
        profile_summary="| kernel | ms |",
    )

    assert "status: invalid (compiled=False, speedup=0.000x)" in diagnostic
    assert "layout mismatch" in diagnostic
    assert "DSLRuntimeError: bad layout" in diagnostic
    assert "task=x FAIL" in diagnostic
    assert "| kernel | ms |" in diagnostic
    # Empty sections are omitted rather than left as blank headings.
    assert "profile" not in build_diagnostic(result={"valid": True, "compiled": True})


def test_long_logs_are_truncated_from_the_front() -> None:
    diagnostic = build_diagnostic(result={"valid": False}, stderr="x" * 9000 + "THE-REAL-ERROR")

    assert "THE-REAL-ERROR" in diagnostic
    assert len(diagnostic) < 4_000


def test_task_markdown_names_the_candidate_and_bounds_the_reply() -> None:
    markdown = critic_task_markdown(
        task_id="level1_01_square_matrix_multiplication_fp8",
        iteration=2,
        candidate_path="/runs/run/iter_002/island_0/candidate/submission.py",
        diagnostic="DSLRuntimeError: bad layout",
        hint_limit=3,
    )

    assert "/runs/run/iter_002/island_0/candidate/submission.py" in markdown
    assert "DSLRuntimeError: bad layout" in markdown
    assert "at most 3 hints" in markdown
    assert '{"hints": ["..."]}' in markdown
    assert "Write no code" in markdown


def test_hints_lead_the_next_turns_feedback(tmp_path: Path) -> None:
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=failing_evaluator)
    controller.init_run(b300_config(), run_id="critic-on")
    controller.prepare_iteration("critic-on")
    controller.evaluate_iteration("critic-on")
    controller.record_critic_hints(
        "critic-on", ["The MMA tiler shape disagrees with the A layout."]
    )
    controller.advance_iteration("critic-on")

    task = controller.prepare_iteration("critic-on")[0]

    feedback = (task.task_file.parent / "FEEDBACK.md").read_text(encoding="utf-8")
    lines = [line for line in feedback.splitlines() if line.startswith("- ")]
    assert lines[0] == "- Critic on turn 1: The MMA tiler shape disagrees with the A layout."
    assert len(lines) > 1, "critic hints lead the harness feedback, they do not replace it"
    assert any(path.name == "FEEDBACK.md" for path in task.readable_files)


def test_hints_survive_a_failing_turn_that_is_never_promoted(tmp_path: Path) -> None:
    """The case the critic exists for: nothing is promoted, so no parent carries it."""
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=failing_evaluator)
    controller.init_run(b300_config(), run_id="critic-unpromoted")
    controller.prepare_iteration("critic-unpromoted")
    controller.evaluate_iteration("critic-unpromoted")
    controller.record_critic_hints("critic-unpromoted", ["Start from a static layout."])
    controller.advance_iteration("critic-unpromoted")

    state = controller.store.read_state("critic-unpromoted")
    assert state["archive"]["island_elites"]["0"] == "seed"
    assert not state["archive"]["entries"][0]["promoted"]

    task = controller.prepare_iteration("critic-unpromoted")[0]
    assert "Start from a static layout." in (
        task.task_file.parent / "FEEDBACK.md"
    ).read_text(encoding="utf-8")


def test_the_newest_critique_replaces_the_previous_one(tmp_path: Path) -> None:
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=failing_evaluator)
    controller.init_run(b300_config(), run_id="critic-replace")
    for turn, hint in ((1, "first critique"), (2, "second critique")):
        controller.prepare_iteration("critic-replace")
        controller.evaluate_iteration("critic-replace")
        controller.record_critic_hints("critic-replace", [hint])
        controller.advance_iteration("critic-replace")
        assert controller.store.read_state("critic-replace")["critic"]["0"]["iteration"] == turn

    feedback = (
        controller.prepare_iteration("critic-replace")[0].task_file.parent / "FEEDBACK.md"
    ).read_text(encoding="utf-8")
    assert "Critic on turn 2: second critique" in feedback
    assert "first critique" not in feedback


def test_recording_no_hints_clears_the_channel(tmp_path: Path) -> None:
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=failing_evaluator)
    controller.init_run(b300_config(), run_id="critic-empty")
    controller.prepare_iteration("critic-empty")
    controller.evaluate_iteration("critic-empty")
    controller.record_critic_hints("critic-empty", ["something"])
    assert controller.record_critic_hints("critic-empty", ["", "   "]) == []
    controller.advance_iteration("critic-empty")

    feedback = (
        controller.prepare_iteration("critic-empty")[0].task_file.parent / "FEEDBACK.md"
    ).read_text(encoding="utf-8")
    assert "Critic" not in feedback
    assert "something" not in feedback


def test_a_critique_only_prepends_and_never_rewrites(tmp_path: Path) -> None:
    bullets = {}
    for run_id, hints in (("with-critic", ["a hint"]), ("no-critic", [])):
        controller = KernelEvoAgent(tmp_path / run_id, evaluator=passing_evaluator)
        controller.init_run(b300_config(), run_id=run_id)
        controller.prepare_iteration(run_id)
        controller.evaluate_iteration(run_id)
        if hints:
            controller.record_critic_hints(run_id, hints)
        controller.advance_iteration(run_id)
        task = controller.prepare_iteration(run_id)[0]
        text = (task.task_file.parent / "FEEDBACK.md").read_text(encoding="utf-8")
        bullets[run_id] = [line for line in text.splitlines() if line.startswith("- ")]

    assert bullets["no-critic"]
    assert not any("Critic" in line for line in bullets["no-critic"])
    assert bullets["with-critic"] == ["- Critic on turn 1: a hint", *bullets["no-critic"]]


def test_critic_is_off_by_default_in_the_frozen_study() -> None:
    module = load_run_iter_matrix()
    base = {"agent": {}, "evaluator": {}}

    assert module.critic_enabled(base) is False
    assert module.critic_enabled({**base, "feedback": {"critic": True}}) is True
    assert module.critic_enabled(yaml.safe_load(STUDY.read_text(encoding="utf-8"))) is False


def test_runner_records_hints_and_replays_them_without_a_second_call(tmp_path: Path) -> None:
    module = load_run_iter_matrix()
    controller = KernelEvoAgent(tmp_path / "runs", evaluator=failing_evaluator)
    controller.init_run(b300_config(), run_id="critic-runner")
    controller.prepare_iteration("critic-runner")
    controller.evaluate_iteration("critic-runner")

    calls = []

    def fake_session(agent, model, timeout, prompt, transcript_path):
        calls.append((agent, prompt))
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = transcript('```json\n{"hints": ["Fix the tiler shape."]}\n```')
        transcript_path.write_text(stdout, encoding="utf-8")
        return type("Completed", (), {"stdout": stdout, "stderr": "", "returncode": 0})()

    module.run_session = fake_session
    module.session_tokens = lambda session_id: {"total_tokens": 1234}
    trace_dir = tmp_path / "critic-trace"

    first = module.critique_iteration(
        controller, "critic-runner", 1, "level1_01", trace_dir, "model", 900
    )
    second = module.critique_iteration(
        controller, "critic-runner", 1, "level1_01", trace_dir, "model", 900
    )

    assert first["hints"] == ["Fix the tiler shape."]
    assert first["critic_tokens"] == 1234
    assert second["hints"] == first["hints"]
    assert len(calls) == 1, "a resumed arm must not pay for the same critique twice"
    assert calls[0][0] == "cute-fp8-critic"
    assert (trace_dir / "CRITIC.md").is_file()
    assert (trace_dir / "hints.json").is_file()
    assert "layout mismatch" in (trace_dir / "CRITIC.md").read_text(encoding="utf-8")

    controller.advance_iteration("critic-runner")
    task = controller.prepare_iteration("critic-runner")[0]
    assert "Critic on turn 1: Fix the tiler shape." in (
        task.task_file.parent / "FEEDBACK.md"
    ).read_text(encoding="utf-8")


def test_the_critic_agent_definition_can_write_nothing() -> None:
    definition = (REPO_ROOT / ".opencode" / "agents" / "cute-fp8-critic.md").read_text(
        encoding="utf-8"
    )
    front_matter = yaml.safe_load(definition.split("---")[1])

    assert front_matter["permission"]["edit"] == {"*": "deny"}
    assert front_matter["permission"]["bash"] == "deny"
    assert front_matter["permission"]["glob"] == "deny"
    assert front_matter["permission"]["grep"] == "deny"
    assert front_matter["permission"]["webfetch"] == "deny"
