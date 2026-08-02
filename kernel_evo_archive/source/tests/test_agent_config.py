from __future__ import annotations

from pathlib import Path

import pytest

from kernel_evo.agent import AgentRunConfig, ConfigurationError


def test_yaml_config_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "kernel.py").write_text("class ModelNew: ...\n", encoding="utf-8")
    config_path = project / "evo.yaml"
    config_path.write_text(
        "run:\n"
        "  name: visible\n"
        "  steps: 3\n"
        "  islands: 2\n"
        "problem:\n"
        "  baseline: kernel.py\n"
        "  backend: triton\n",
        encoding="utf-8",
    )

    config = AgentRunConfig.from_file(config_path)
    assert config.baseline == str((project / "kernel.py").resolve())
    assert config.steps == 3
    assert config.islands == 2


def test_config_rejects_partial_kernelbench_identity() -> None:
    with pytest.raises(ConfigurationError, match="level and problem_id"):
        AgentRunConfig.from_mapping({"level": 1, "backend": "triton"})


def test_config_can_disable_documentation(tmp_path: Path) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    config = AgentRunConfig.from_mapping(
        {
            "run": {"documentation_enabled": False},
            "problem": {"baseline": str(baseline), "backend": "triton"},
        }
    )

    assert config.documentation_enabled is False


def test_b300_timeline_is_an_explicit_profile_option(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    config = AgentRunConfig.from_mapping(
        {
            "problem": {"path": str(task), "backend": "cute"},
            "evaluation": {"kind": "cute_b300"},
            "profiling": {"enabled": True, "timeline": True},
        }
    )

    assert config.profile_enabled is True
    assert config.profile_timeline is True
    defaults = AgentRunConfig.from_mapping(
        {
            "problem": {"path": str(task), "backend": "cute"},
            "evaluation": {"kind": "cute_b300"},
        }
    )
    assert defaults.profile_timeline is False

    with pytest.raises(ConfigurationError, match="timeline=true requires"):
        AgentRunConfig.from_mapping(
            {
                "problem": {"path": str(task), "backend": "cute"},
                "evaluation": {"kind": "cute_b300"},
                "profiling": {"timeline": True},
            }
        )
    with pytest.raises(ConfigurationError, match="evaluation.kind=cute_b300"):
        AgentRunConfig.from_mapping(
            {
                "problem": {"path": str(task), "backend": "triton"},
                "profiling": {"enabled": True, "timeline": True},
            }
        )
