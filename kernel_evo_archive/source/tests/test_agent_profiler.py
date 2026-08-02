from kernel_evo.agent.profiler import coerce_profile_result, profile_result_status


def test_compact_markdown_keeps_all_runner_sections() -> None:
    profile = coerce_profile_result(
        {
            "torch": {"operations": [{"name": f"kernel_{index}"} for index in range(100)]},
            "ncu": {"status": "completed", "counter_metrics_available": True},
        }
    )

    assert "kernel_99" in profile.summary
    assert '"ncu"' in profile.summary
    assert "profile summary truncated" not in profile.summary


def test_profile_status_requires_at_least_one_successful_runner() -> None:
    failed = coerce_profile_result(
        {"torch": {"status": "failed", "reason": "GPU lease unavailable"}}
    )
    partial = coerce_profile_result(
        {
            "torch": {"status": "failed"},
            "ncu": {"status": "completed", "counter_metrics_available": True},
        }
    )

    assert profile_result_status(failed) == "failed"
    assert profile_result_status(partial) == "completed"
