from __future__ import annotations

from kernel_evo.core.profile.torch_runner import _allocation_graph_failure


def test_dynamic_allocations_skip_cuda_graph_capture() -> None:
    graph = _allocation_graph_failure(82.0, ["per-forward allocation"])

    assert graph is not None
    assert graph["capturable"] is False
    assert graph["capture_succeeded"] is False
    assert graph["capture_skipped"] == "dynamic_allocations"
    assert "82.0 dynamic allocation" in graph["failure"]


def test_zero_dynamic_allocations_allow_capture_attempt() -> None:
    assert _allocation_graph_failure(0.0, []) is None
