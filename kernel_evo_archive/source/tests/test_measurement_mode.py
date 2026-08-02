from __future__ import annotations

import argparse
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kernel_evo.agent import AgentRunConfig, ConfigurationError
from kernel_evo.commands import compare as compare_cmd
from kernel_evo.commands import evolve as evolve_cmd
from kernel_evo.core.eval.measurement import (
    device_activity_time_ms,
    get_measurement_function,
    time_execution_with_device_time,
)


def test_agent_config_defaults_to_existing_wall_clock_behavior(tmp_path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: pass\n", encoding="utf-8")

    config = AgentRunConfig.from_mapping(
        {"problem": {"baseline": str(baseline)}}
    )

    assert config.measurement_mode == "wall-clock"


def test_agent_config_reads_nested_device_time_mode(tmp_path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: pass\n", encoding="utf-8")

    config = AgentRunConfig.from_mapping(
        {
            "problem": {"baseline": str(baseline)},
            "evaluation": {"measurement_mode": "device-time"},
        }
    )

    assert config.measurement_mode == "device-time"


def test_agent_config_rejects_unknown_measurement_mode(tmp_path) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text("class ModelNew: pass\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="wall-clock, device-time"):
        AgentRunConfig.from_mapping(
            {
                "problem": {"baseline": str(baseline)},
                "evaluation": {"measurement_mode": "cuda-event-ish"},
            }
        )


def test_device_activity_sums_kernel_and_memcpy_durations() -> None:
    profiler = SimpleNamespace(
        key_averages=lambda: [
            SimpleNamespace(self_device_time_total=120.0),
            SimpleNamespace(self_cuda_time_total=30.0),
            SimpleNamespace(self_device_time_total=0.0),
        ]
    )

    assert device_activity_time_ms(profiler) == pytest.approx(0.15)


def test_device_activity_rejects_identity_with_no_gpu_work() -> None:
    profiler = SimpleNamespace(
        key_averages=lambda: [SimpleNamespace(self_device_time_total=0.0)]
    )

    with pytest.raises(RuntimeError, match="no CUDA kernel or memcpy activity"):
        device_activity_time_ms(profiler)


def test_device_time_trials_share_one_profiler_session(monkeypatch) -> None:
    calls = {"kernel": 0, "profile": 0}

    class FakeProfile:
        def __enter__(self):
            calls["profile"] += 1
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def key_averages():
            # Four measured calls at 100 us each.
            return [SimpleNamespace(self_device_time_total=400.0)]

    monkeypatch.setattr(
        "kernel_evo.core.eval.measurement.profile", lambda **_kwargs: FakeProfile()
    )
    monkeypatch.setattr("torch.cuda.device", lambda _device: nullcontext())
    monkeypatch.setattr("torch.cuda.synchronize", lambda **_kwargs: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _device: "fake")

    samples = time_execution_with_device_time(
        lambda: calls.__setitem__("kernel", calls["kernel"] + 1),
        [],
        num_warmup=2,
        num_trials=4,
        discard_first=1,
        verbose=False,
        device="cuda:0",
    )

    assert calls == {"kernel": 7, "profile": 1}
    assert samples == pytest.approx([0.1, 0.1, 0.1, 0.1])


def test_device_time_prepares_production_state_before_warmup(monkeypatch) -> None:
    calls = {"prepare": 0, "kernel": 0}

    class Kernel:
        def prepare_for_timing(self, value):
            assert value == 7
            calls["prepare"] += 1

        def __call__(self, value):
            assert value == 7
            calls["kernel"] += 1

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def key_averages():
            return [SimpleNamespace(self_device_time_total=100.0)]

    monkeypatch.setattr(
        "kernel_evo.core.eval.measurement.profile", lambda **_kwargs: FakeProfile()
    )
    monkeypatch.setattr("torch.cuda.device", lambda _device: nullcontext())
    monkeypatch.setattr("torch.cuda.synchronize", lambda **_kwargs: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    time_execution_with_device_time(
        Kernel(),
        [7],
        num_warmup=2,
        num_trials=2,
        discard_first=1,
        verbose=False,
        device="cuda:0",
    )

    assert calls == {"prepare": 1, "kernel": 5}


def test_wall_clock_mode_uses_configured_kernelbench_timer(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "kernelbench.timing.get_timing_function",
        lambda method: sentinel if method == "cuda_event" else None,
    )

    assert get_measurement_function("wall-clock", "cuda_event") is sentinel


@pytest.mark.parametrize("command", ["evolve", "compare"])
def test_cli_accepts_device_time_mode(command: str) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    if command == "evolve":
        evolve_cmd.setup_parser(subparsers)
        argv = [
            "evolve",
            "--experiment-name",
            "x",
            "--model-name",
            "m",
            "--measurement-mode",
            "device-time",
        ]
    else:
        compare_cmd.setup_parser(subparsers)
        argv = [
            "compare",
            "--program-a",
            "a.py",
            "--program-b",
            "b.py",
            "--measurement-mode",
            "device-time",
        ]

    assert parser.parse_args(argv).measurement_mode == "device-time"
