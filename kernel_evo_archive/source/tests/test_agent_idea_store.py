from __future__ import annotations

from kernel_evo.agent.idea_store import beats


def test_valid_candidate_replaces_quarantined_faster_incumbent() -> None:
    candidate = {"result": {"valid": True, "fitness": 0.82}}
    quarantined = {
        "result": {"valid": True, "fitness": 0.84},
        "parent_profile_failures": 2,
    }

    assert beats(candidate, quarantined) is True


def test_slower_candidate_does_not_replace_profile_safe_incumbent() -> None:
    candidate = {"result": {"valid": True, "fitness": 0.82}}
    incumbent = {"result": {"valid": True, "fitness": 0.84}}

    assert beats(candidate, incumbent) is False
