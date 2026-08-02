from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kernel_evo.agent import EvaluationResult, KernelEvoAgent
from kernel_evo.agent.config import AgentRunConfig
from kernel_evo.cute_harness import b300
from kernel_evo.cute_harness.b300 import (
    EvaluationConfig,
    baseline_candidate,
    discover_tasks,
    metrics,
)
from kernel_evo.cute_harness.trace_summary import (
    summarize_chrome_trace,
    summarize_chrome_timeline,
    trace_summary_markdown,
    trace_timeline_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "b300_profile_trace.json"
TABLE_HEADER = "| kernel | ms | calls | % GPU |"
TIMELINE_HEADER = "| # | start ms | gap before us | duration us | lane | activity |"


def real_trace() -> dict[str, Any]:
    """A trimmed capture from an actual B300 harness run."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def synthetic_trace() -> dict[str, Any]:
    """Round microsecond durations so every aggregate is checkable by hand."""
    return {
        "displayTimeUnit": "ms",
        "deviceProperties": [
            {"id": 0, "name": "NVIDIA B300 SXM6 AC", "numSms": 148, "sharedMemPerBlockOptin": 232448}
        ],
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 6000.0},
            {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 2000.0},
            {"ph": "X", "cat": "kernel", "name": "epilogue", "dur": 1500.0},
            {"ph": "X", "cat": "kernel", "name": "convert", "dur": 400.0},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "dur": 80.0},
            {"ph": "X", "cat": "gpu_memset", "name": "Memset (Device)", "dur": 20.0},
            # Host-side and flow events share the trace but never count as GPU time.
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 900_000.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 500.0},
            {"ph": "f", "cat": "ac2g", "name": "ac2g", "dur": 700.0},
            {"ph": "M", "name": "process_name"},
        ],
    }


class StubHarnessClient:
    def __init__(self, response: dict[str, Any], profile: bytes | Exception) -> None:
        self.response = response
        self.profile = profile

    def run_file(self, submission: Path, profiler: str) -> dict[str, Any]:
        return self.response

    def download_profile(self, profile_id: str) -> bytes:
        if isinstance(self.profile, Exception):
            raise self.profile
        return self.profile


def run_evaluate(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    *,
    profile_id: str | None,
    profile: bytes | Exception = b"",
    profile_timeline: bool = False,
) -> tuple[dict[str, Any], Path]:
    task = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    # Keep this unit test independent of the checkout's live B300 lock.
    monkeypatch.setattr(b300, "_b300_lock_path", lambda: output_dir.parent / "b300.lock")
    candidate = output_dir.parent / "candidate_source.py"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(baseline_candidate(task), encoding="utf-8")
    response = {
        "success": True,
        "exit_code": 0,
        "stdout": "task=level1_01_square_matrix_multiplication kernel_time_ms=1.5 PASS\n",
        "stderr": "",
        "profile_id": profile_id,
    }
    monkeypatch.setenv("CUTE_HARNESS_API_KEY", "test-key")
    monkeypatch.setattr(
        b300, "HarnessClient", lambda *_, **__: StubHarnessClient(response, profile)
    )
    record = b300.evaluate(
        task,
        candidate,
        output_dir,
        EvaluationConfig(
            seed=0,
            warmup=1,
            repeats=1,
            profile_timeline=profile_timeline,
        ),
    )
    return record, output_dir


def load_run_iter_matrix():
    path = REPO_ROOT / "experiments" / "cute_ablation" / "run_iter_matrix.py"
    spec = importlib.util.spec_from_file_location("run_iter_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def b300_config(
    *, profile_enabled: bool, profile_timeline: bool = False, steps: int = 2
) -> AgentRunConfig:
    config = AgentRunConfig.from_file(REPO_ROOT / "examples" / "agent" / "airi_cute_b300.yaml")
    return replace(
        config,
        profile_enabled=profile_enabled,
        profile_timeline=profile_timeline,
        b300_seed="starter",
        seed_preflight=False,
        steps=steps,
    )


def passing_metrics() -> dict[str, float]:
    return {
        "compiled": 1,
        "correctness": 1,
        "is_valid": 1,
        "runtime_us": 500.0,
        "ref_runtime_us": 1000.0,
        "speedup": 2.0,
        "fitness": 2.0,
    }


def island_summary_evaluator(summary_text: str):
    """Stand in for CuteB300Evaluator: write the artifacts it leaves on disk."""

    def evaluate(context) -> EvaluationResult:
        b300_dir = context.island_dir / "b300"
        b300_dir.mkdir(parents=True, exist_ok=True)
        (b300_dir / "profile_summary.md").write_text(summary_text, encoding="utf-8")
        (b300_dir / "result.json").write_text(
            json.dumps({"passed": True, "profile_summary": {"gpu_time_ms": 7.6}}),
            encoding="utf-8",
        )
        return EvaluationResult.from_metrics(passing_metrics())

    return evaluate


def test_summarizes_only_gpu_events_with_hand_checkable_arithmetic() -> None:
    summary = summarize_chrome_trace(synthetic_trace())

    # 6000 + 2000 + 1500 + 400 microseconds of kernels, 80 memcpy, 20 memset.
    assert summary["kernel_time_ms"] == 9.9
    assert summary["kernel_calls"] == 4
    assert summary["memcpy_time_ms"] == 0.08
    assert summary["memcpy_calls"] == 1
    assert summary["memset_time_ms"] == 0.02
    assert summary["memset_calls"] == 1
    assert summary["gpu_time_ms"] == 10.0
    assert summary["distinct_kernels"] == 3
    assert summary["top_kernels"] == [
        {"name": "gemm", "total_ms": 8.0, "calls": 2, "pct_gpu_time": 80.0},
        {"name": "epilogue", "total_ms": 1.5, "calls": 1, "pct_gpu_time": 15.0},
        {"name": "convert", "total_ms": 0.4, "calls": 1, "pct_gpu_time": 4.0},
    ]
    assert summary["device"] == {
        "name": "NVIDIA B300 SXM6 AC",
        "num_sms": 148,
        "shared_mem_per_block_optin": 232448,
    }


def test_top_k_bounds_the_table_and_ties_break_by_name() -> None:
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": name, "dur": 1000.0}
            for name in ("delta", "alpha", "charlie", "bravo")
        ]
    }

    ranked = summarize_chrome_trace(trace)["top_kernels"]

    assert [item["name"] for item in ranked] == ["alpha", "bravo", "charlie", "delta"]
    assert [item["name"] for item in summarize_chrome_trace(trace, top_k=2)["top_kernels"]] == [
        "alpha",
        "bravo",
    ]


def test_summarizes_a_real_trimmed_b300_trace() -> None:
    trace = real_trace()
    summary = summarize_chrome_trace(trace)

    assert summary["device"]["name"] == "NVIDIA B300 SXM6 AC"
    assert summary["kernel_calls"] == 10
    assert summary["memcpy_calls"] == 6
    assert summary["memset_calls"] == 2
    assert summary["distinct_kernels"] == 5
    assert summary["top_kernels"][0]["name"].startswith("kernel_cutlass_square_gemm_kernel")
    assert summary["top_kernels"][0]["pct_gpu_time"] > 50.0
    totals = [item["total_ms"] for item in summary["top_kernels"]]
    assert totals == sorted(totals, reverse=True)
    assert sum(item["calls"] for item in summary["top_kernels"]) == summary["kernel_calls"]

    # `dur` is microseconds even though the trace declares displayTimeUnit "ms".
    assert trace["displayTimeUnit"] == "ms"
    assert 0.0 < summary["gpu_time_ms"] < 1.0


def looping_trace(iterations: int = 5, setup: int = 9) -> dict[str, Any]:
    """A three-kernel block repeated `iterations` times, preceded by setup.

    Shaped like a real capture: setup is many distinct names launched once,
    while the block is few names launched many times. Every figure is round.

    Per iteration: alpha [0,10), 5 idle, beta [15,45), 1 idle, gamma [46,56).
    Busy 50, span 56, so 6 idle inside the block; 44 more before the next one.
    """
    events: list[dict[str, Any]] = [
        {"ph": "X", "cat": "kernel", "name": f"setup_{index}", "ts": 100.0 + index, "dur": 1.0}
        for index in range(setup)
    ]
    for index in range(iterations):
        origin = 1000.0 + index * 100.0
        events.extend(
            [
                {"ph": "X", "cat": "kernel", "name": "alpha", "ts": origin, "dur": 10.0},
                {"ph": "X", "cat": "kernel", "name": "beta", "ts": origin + 15.0, "dur": 30.0},
                {"ph": "X", "cat": "kernel", "name": "gamma", "ts": origin + 46.0, "dur": 10.0},
            ]
        )
    return {"traceEvents": events}


def test_iteration_view_reports_start_end_and_the_gap_before_each_launch() -> None:
    """The whole-capture idle figure is dominated by setup, not by the block.

    On a real trace it reads 99.4% idle over a 4922 ms span while the block
    itself is 0.4% idle, so an author steered by that number chases starvation
    that is not there. The per-iteration reduction is the honest measurement.
    """
    iteration = summarize_chrome_timeline(looping_trace())["iteration"]

    assert iteration["iterations"] == 5
    assert iteration["kernels_per_iteration"] == 3
    assert iteration["span_us"] == 56.0
    assert iteration["busy_us"] == 50.0
    assert iteration["gap_us"] == 6.0
    assert iteration["gap_pct"] == pytest.approx(10.71, abs=0.01)
    assert iteration["largest_gap_us"] == 5.0
    # Dispatch between repeats is not a bubble the author can fuse away.
    assert iteration["between_iterations_us"] == 44.0
    assert [
        (row["start_us"], row["end_us"], row["duration_us"], row["gap_before_us"])
        for row in iteration["rows"]
    ] == [(0.0, 10.0, 10.0, None), (15.0, 45.0, 30.0, 5.0), (46.0, 56.0, 10.0, 1.0)]


def test_iteration_view_finds_the_loop_when_setup_names_outnumber_it() -> None:
    """Nine names launched once must not be mistaken for the repeating block.

    Picking the most common launch count selects 1 here, and on the real
    capture, where eleven distinct kernels launch exactly once against seven
    launched fifty-five times. The bucket covering the most launches is the
    loop.
    """
    for setup in (9, 20, 40):
        iteration = summarize_chrome_timeline(looping_trace(setup=setup))["iteration"]
        assert iteration["kernels_per_iteration"] == 3, setup
        assert iteration["iterations"] == 5, setup


def test_iteration_view_is_absent_when_nothing_repeats() -> None:
    once = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": f"k{index}", "ts": float(index * 10), "dur": 1.0}
            for index in range(6)
        ]
    }

    timeline = summarize_chrome_timeline(once)
    assert timeline["iteration"] == {}
    assert "One iteration of your block" not in trace_timeline_markdown(timeline)


def test_iteration_table_names_the_kernels_and_separates_the_two_idle_figures() -> None:
    timeline = summarize_chrome_timeline(
        looping_trace(), candidate_kernel_symbols=("alpha", "beta", "gamma")
    )

    rendered = trace_timeline_markdown(timeline)

    assert "## One iteration of your block, launch by launch" in rendered
    assert "| # | kernel | id | start | end | dur | idle before |" in rendered
    # Named, not just the legend alias, so the section reads on its own.
    assert "| 2 | beta | K" in rendered
    assert "immediately before `beta`" in rendered
    assert "excluded from the idle figure above" in rendered
    assert "not a measure of your kernels" in rendered
    assert rendered == trace_timeline_markdown(
        summarize_chrome_timeline(
            looping_trace(), candidate_kernel_symbols=("alpha", "beta", "gamma")
        )
    )


def test_a_numeric_miss_is_a_correctness_failure_not_a_compile_failure() -> None:
    """The three metrics were one boolean, so every failure read as a rewrite.

    `compiled and not correctness` is what makes the scheduler say "repair the
    smallest reported numerical mismatch"; fused, that branch never fired.
    """
    def record(stderr: str, *, passed: bool = False) -> dict[str, Any]:
        return {
            "passed": passed,
            "kernel_time_ms": 0.5 if passed else None,
            "response": {"exit_code": 0 if passed else 1, "stderr": stderr},
        }

    ran = metrics(record("RuntimeError: validation failed: quantized_abs=0.785310"), 3.5923)
    assert (ran["compiled"], ran["correctness"], ran["is_valid"]) == (1.0, 0.0, 0.0)

    broken = metrics(record("AttributeError: module 'cutlass.cute' has no attribute 'maximum'"), 3.5923)
    assert (broken["compiled"], broken["correctness"], broken["is_valid"]) == (0.0, 0.0, 0.0)

    ok = metrics(record("", passed=True), 3.5923)
    assert (ok["compiled"], ok["correctness"], ok["is_valid"]) == (1.0, 1.0, 1.0)

    # A failure is never scored, whichever kind it is.
    assert ran["speedup"] == 0.0 and broken["speedup"] == 0.0
    assert ran["fitness"] == 0.0 and broken["fitness"] == 0.0


def test_the_scheduler_now_says_repair_rather_than_rewrite_for_a_numeric_miss() -> None:
    from kernel_evo.agent.scheduler import IslandScheduler

    def feedback_for(stderr: str) -> list[str]:
        result = metrics({"passed": False, "response": {"exit_code": 1, "stderr": stderr}}, 3.5923)
        return IslandScheduler().compact_feedback(
            {
                "entries": [
                    {
                        "id": "cand-1",
                        "island": 0,
                        "idea": {"summary": "implement the block"},
                        "result": {"valid": False, **result},
                    }
                ]
            },
            island=0,
        )

    numeric = feedback_for("RuntimeError: validation failed: max_abs=0.089622")
    assert any("correctness failure" in line for line in numeric)
    assert any("numerical mismatch" in line for line in numeric)
    assert not any("compile failure" in line for line in numeric)

    broken = feedback_for("AttributeError: module 'cutlass.cute' has no attribute 'maximum'")
    assert any("compile failure" in line for line in broken)
    assert any("rewrite the candidate" in line for line in broken)


def test_timeline_retains_every_gpu_activity_and_measures_idle_holes() -> None:
    trace = {
        "deviceProperties": [{"name": "NVIDIA B300 SXM6 AC"}],
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel_cutlass_candidate_kernel_cfg",
                "ts": 1000.0,
                "dur": 100.0,
                "tid": 7,
                "args": {"device": 0, "stream": 7, "grid": [1, 2, 1], "block": [128, 1, 1]},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy HtoD",
                "ts": 1050.0,
                "dur": 25.0,
                "args": {"device": 0, "stream": 8, "bytes": 64},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "framework_kernel",
                "ts": 1200.0,
                "dur": 50.0,
                "args": {"device": 0, "stream": 7},
            },
            {
                "ph": "X",
                "cat": "gpu_memset",
                "name": "Memset (Device)",
                "ts": 1400.0,
                "dur": 10.0,
                "args": {"device": 0, "stream": 7, "bytes": 4},
            },
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 1100.0, "dur": 9000.0},
        ],
    }

    timeline = summarize_chrome_timeline(
        trace,
        candidate_kernel_symbols=("candidate_kernel", "dead_kernel"),
    )

    assert timeline["event_count"] == 4
    assert timeline["timeline_span_ms"] == 0.41
    assert timeline["activity_time_ms"] == 0.185
    assert timeline["busy_time_ms"] == 0.16
    assert timeline["idle_time_ms"] == 0.25
    assert timeline["largest_gap_ms"] == 0.15
    assert [event["gap_before_us"] for event in timeline["events"]] == [0.0, 0.0, 100.0, 150.0]
    assert timeline["observed_candidate_symbols"] == ["candidate_kernel"]
    assert timeline["unobserved_candidate_symbols"] == ["dead_kernel"]
    assert timeline["kernels"][0]["source_hint"] == "candidate:candidate_kernel"
    assert timeline["kernels"][1]["source_hint"] == "framework/validation"

    markdown = trace_timeline_markdown(timeline)
    assert TIMELINE_HEADER in markdown
    assert markdown.count("\n| 1 |") == 1
    assert all(f"\n| {index} |" in markdown for index in range(1, 5))
    assert "candidate symbols not observed in this trace: `dead_kernel`" in markdown
    assert "aten::mm" not in markdown


def test_markdown_is_deterministic_and_stays_under_two_kilobytes() -> None:
    summary = summarize_chrome_trace(real_trace())

    first = trace_summary_markdown(summary)
    second = trace_summary_markdown(summarize_chrome_trace(real_trace()))

    assert first == second
    assert len(first.encode("utf-8")) <= 2048
    assert TABLE_HEADER in first
    assert "NVIDIA B300 SXM6 AC" in first
    assert "148 SMs" in first
    assert first.count("\n|") == 2 + len(summary["top_kernels"])


def test_markdown_respects_max_chars_and_reports_dropped_rows() -> None:
    summary = summarize_chrome_trace(real_trace())

    full = trace_summary_markdown(summary, max_chars=2000)
    clipped = trace_summary_markdown(summary, max_chars=700)
    tiny = trace_summary_markdown(summary, max_chars=120)

    assert len(clipped) <= 700
    assert len(tiny) <= 120
    assert clipped.count("\n|") < full.count("\n|")
    assert "lower-cost kernel(s) omitted." in clipped
    assert clipped == trace_summary_markdown(summary, max_chars=700)


def test_malformed_traces_raise_instead_of_returning_a_wrong_summary() -> None:
    for trace in ([], {}, {"traceEvents": "not-a-list"}, {"traceEvents": {}}):
        with pytest.raises(ValueError):
            summarize_chrome_trace(trace)
    with pytest.raises(ValueError):
        summarize_chrome_trace({"traceEvents": []}, top_k=0)

    # A single unusable event is skipped rather than failing the whole trace.
    summary = summarize_chrome_trace(
        {
            "traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "good", "dur": 1000.0},
                {"ph": "X", "cat": "kernel", "name": "no-duration"},
                {"ph": "X", "cat": "kernel", "name": "bad-duration", "dur": "slow"},
                "not-an-event",
            ]
        }
    )
    assert summary["kernel_calls"] == 1
    assert summary["top_kernels"] == [
        {"name": "good", "total_ms": 1.0, "calls": 1, "pct_gpu_time": 100.0}
    ]


def test_missing_device_properties_still_render(tmp_path: Path) -> None:
    summary = summarize_chrome_trace(
        {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "gemm", "dur": 1000.0}]}
    )

    assert summary["device"] == {}
    assert "device:" not in trace_summary_markdown(summary)
    assert TABLE_HEADER in trace_summary_markdown(summary)


def test_evaluate_records_the_trace_summary_beside_the_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.dumps(real_trace()).encode("utf-8")

    record, output_dir = run_evaluate(
        monkeypatch, tmp_path / "eval", profile_id="abc123", profile=payload
    )

    assert (output_dir / "profile.json").is_file()
    summary_text = (output_dir / "profile_summary.md").read_text(encoding="utf-8")
    assert TABLE_HEADER in summary_text
    assert len(summary_text.encode("utf-8")) <= 2048
    assert record["profile_summary"]["kernel_calls"] == 10
    assert "profile_summary_error" not in record
    assert "profile_download_error" not in record
    assert "profile_timeline" not in record

    persisted = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["profile_summary"] == record["profile_summary"]


def test_evaluate_can_write_the_complete_timeline_instead_of_the_legacy_packet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.dumps(real_trace()).encode("utf-8")

    record, output_dir = run_evaluate(
        monkeypatch,
        tmp_path / "eval",
        profile_id="abc123",
        profile=payload,
        profile_timeline=True,
    )

    text = (output_dir / "profile_summary.md").read_text(encoding="utf-8")
    assert "# B300 complete GPU timeline" in text
    assert TABLE_HEADER in text
    assert TIMELINE_HEADER in text
    assert record["profile_timeline"]["event_count"] == 18
    assert "events" not in record["profile_timeline"]
    assert "kernels" not in record["profile_timeline"]
    # The candidate baseline declares this symbol, and all 18 GPU activities remain visible.
    assert "candidate:square_gemm_kernel" in text
    assert all(f"\n| {index} |" in text for index in range(1, 19))


def test_evaluate_records_a_parse_error_without_failing_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record, output_dir = run_evaluate(
        monkeypatch, tmp_path / "eval", profile_id="abc123", profile=b"{not json"
    )

    assert "profile_summary_error" in record
    assert "profile_summary" not in record
    assert not (output_dir / "profile_summary.md").exists()
    # The timed evaluation itself is untouched.
    assert record["kernel_time_ms"] == 1.5
    assert record["passed"] is True


def test_evaluate_is_silent_when_the_run_produced_no_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record, output_dir = run_evaluate(monkeypatch, tmp_path / "eval", profile_id=None)

    assert "profile_summary" not in record
    assert "profile_summary_error" not in record
    assert "profile_download_error" not in record
    assert not (output_dir / "profile_summary.md").exists()


def test_evaluate_drops_a_stale_summary_when_the_rerun_has_no_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "eval"
    payload = json.dumps(real_trace()).encode("utf-8")
    run_evaluate(monkeypatch, output_dir, profile_id="abc123", profile=payload)
    assert (output_dir / "profile_summary.md").is_file()

    run_evaluate(monkeypatch, output_dir, profile_id=None)

    assert not (output_dir / "profile_summary.md").exists()


def test_profiling_flag_gives_the_next_turn_the_parent_kernel_table(tmp_path: Path) -> None:
    summary_text = trace_summary_markdown(summarize_chrome_trace(real_trace()))
    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=island_summary_evaluator(summary_text)
    )
    controller.init_run(b300_config(profile_enabled=True), run_id="b300-profile-on")
    controller.prepare_iteration("b300-profile-on")
    controller.evaluate_iteration("b300-profile-on")
    controller.advance_iteration("b300-profile-on")

    task = controller.prepare_iteration("b300-profile-on")[0]

    parent_profile = task.task_file.parent / "PARENT_PROFILE.md"
    assert parent_profile.is_file()
    text = parent_profile.read_text(encoding="utf-8")
    assert TABLE_HEADER in text
    assert "kernel_cutlass_square_gemm_kernel" in text
    assert parent_profile.resolve() in task.readable_files
    assert "PARENT_PROFILE.md" in task.task_file.read_text(encoding="utf-8")

    entry = controller.store.read_state("b300-profile-on")["archive"]["entries"][0]
    assert entry["profile_status"] == "completed"
    assert entry["profile_reason"] == "b300_trace"
    assert entry["profile"]["b300_trace"] == {"gpu_time_ms": 7.6}
    # The barrier loop advanced with no reviewer: harness output needs no review.
    assert "profile_review" not in entry


def test_profiling_flag_off_leaves_the_packet_untouched(tmp_path: Path) -> None:
    summary_text = trace_summary_markdown(summarize_chrome_trace(real_trace()))
    controller = KernelEvoAgent(
        tmp_path / "runs", evaluator=island_summary_evaluator(summary_text)
    )
    controller.init_run(b300_config(profile_enabled=False), run_id="b300-profile-off")
    controller.prepare_iteration("b300-profile-off")
    controller.evaluate_iteration("b300-profile-off")
    controller.advance_iteration("b300-profile-off")

    task = controller.prepare_iteration("b300-profile-off")[0]

    assert not (task.task_file.parent / "PARENT_PROFILE.md").exists()
    assert not any(path.name == "PARENT_PROFILE.md" for path in task.readable_files)
    entry = controller.store.read_state("b300-profile-off")["archive"]["entries"][0]
    assert entry["profile_summary"] == ""
    assert entry["profile_status"] == "not_selected"


def test_the_flag_changes_no_documentation_tier_bundle(tmp_path: Path) -> None:
    """The summary is runtime feedback; the tier bundle must stay byte-identical."""
    summary_text = trace_summary_markdown(summarize_chrome_trace(real_trace()))
    bundles = {}
    for enabled, run_id in ((True, "tier-on"), (False, "tier-off")):
        controller = KernelEvoAgent(
            tmp_path / run_id, evaluator=island_summary_evaluator(summary_text)
        )
        controller.init_run(b300_config(profile_enabled=enabled), run_id=run_id)
        controller.prepare_iteration(run_id, documentation_tier="errors")
        controller.evaluate_iteration(run_id)
        controller.advance_iteration(run_id)
        # Turn two is where the parent profile exists, so compare the tier there.
        task = controller.prepare_iteration(run_id, documentation_tier="errors")[0]
        state = controller.store.read_state(run_id)
        documentation = state["iterations"]["2"]["islands"]["0"]["cute_context"]
        bundles[enabled] = [
            (Path(name).name, Path(name).read_bytes())
            for name in documentation["documentation_files"]
        ]
        assert bundles[enabled]
        assert all(name != "PARENT_PROFILE.md" for name, _ in bundles[enabled])
        assert all(
            summary_text not in text.decode("utf-8", errors="replace")
            for _, text in bundles[enabled]
        )
        assert (task.task_file.parent / "PARENT_PROFILE.md").exists() is enabled
    assert bundles[True] == bundles[False]


def test_a_parent_profile_never_comes_from_the_reference_baseline(tmp_path: Path) -> None:
    """The run-level baseline profiles the verified reference and must stay hidden."""

    def evaluate(context) -> EvaluationResult:
        baseline_dir = context.run_dir / "b300" / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "profile_summary.md").write_text(
            "| kernel | ms | calls | % GPU |\n| reference_only_kernel | 1.0 | 1 | 100.0 |\n",
            encoding="utf-8",
        )
        return EvaluationResult.from_metrics(passing_metrics())

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluate)
    controller.init_run(b300_config(profile_enabled=True), run_id="b300-baseline-hidden")
    controller.prepare_iteration("b300-baseline-hidden")
    controller.evaluate_iteration("b300-baseline-hidden")
    controller.advance_iteration("b300-baseline-hidden")

    task = controller.prepare_iteration("b300-baseline-hidden")[0]

    assert not (task.task_file.parent / "PARENT_PROFILE.md").exists()
    assert all(
        "reference_only_kernel" not in path.read_text(encoding="utf-8", errors="replace")
        for path in task.readable_files
    )


def test_real_evaluator_delivers_its_own_trace_and_not_the_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end through CuteB300Evaluator: the two sides agree on the path."""
    reference_trace = {
        "deviceProperties": [{"name": "NVIDIA B300 SXM6 AC", "numSms": 148}],
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "reference_only_kernel", "dur": 1000.0}
        ],
    }
    # The verified baseline is timed first, then the island's own candidate.
    payloads = [
        json.dumps(reference_trace).encode("utf-8"),
        json.dumps(real_trace()).encode("utf-8"),
    ]

    class SequencedClient:
        def download_profile(self, profile_id: str) -> bytes:
            return payloads.pop(0) if len(payloads) > 1 else payloads[0]

        def run_file(self, submission: Path, profiler: str) -> dict[str, Any]:
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "task=level1_01_square_matrix_multiplication kernel_time_ms=1.5 PASS\n",
                "stderr": "",
                "profile_id": "profile-1",
            }

    monkeypatch.setenv("CUTE_HARNESS_API_KEY", "test-key")
    monkeypatch.setattr(b300, "HarnessClient", lambda *_, **__: SequencedClient())
    monkeypatch.setattr(b300, "_b300_lock_path", lambda: tmp_path / "b300.lock")

    controller = KernelEvoAgent(tmp_path / "runs")
    controller.init_run(
        b300_config(profile_enabled=True, profile_timeline=True),
        run_id="b300-end-to-end",
    )
    authoring = controller.prepare_iteration("b300-end-to-end")[0]
    # Stand in for the authoring turn: a kernel that clears the policy gate and
    # is not byte-identical to the reference, so it is timed on its own.
    spec = discover_tasks()["level1_01_square_matrix_multiplication_fp8"]
    authoring.candidate_path.write_text(
        baseline_candidate(spec) + "\n# authored by the island\n", encoding="utf-8"
    )
    report = controller.evaluate_iteration("b300-end-to-end")
    assert report["islands"][0]["valid"] is True
    controller.advance_iteration("b300-end-to-end")

    task = controller.prepare_iteration("b300-end-to-end")[0]

    run_dir = controller.store.run_dir("b300-end-to-end")
    assert (run_dir / "b300" / "baseline" / "profile_summary.md").is_file()
    text = (task.task_file.parent / "PARENT_PROFILE.md").read_text(encoding="utf-8")
    assert TIMELINE_HEADER in text
    assert "kernel_cutlass_square_gemm_kernel" in text
    assert "reference_only_kernel" not in text


def test_study_config_reads_profiling_enabled_and_defaults_off() -> None:
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

    assert module.agent_config(base, "bare", task_path)["profiling"] == {"enabled": False}
    assert module.agent_config({**base, "profiling": {"enabled": True}}, "bare", task_path)[
        "profiling"
    ] == {"enabled": True}
    assert module.agent_config(
        {**base, "profiling": {"enabled": True, "timeline": True}}, "bare", task_path
    )["profiling"] == {"enabled": True, "timeline": True}

    # The frozen study config keeps the flag off.
    import yaml

    frozen = yaml.safe_load(
        (REPO_ROOT / "experiments" / "cute_ablation" / "study-iter.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert module.agent_config(frozen, "bare", task_path)["profiling"] == {"enabled": False}
