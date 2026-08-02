from __future__ import annotations

from pathlib import Path

import pytest

from kernel_evo.agent import AgentRunConfig, ConfigurationError, KernelEvoAgent
from kernel_evo.agent.controller import _compile_check_command
from kernel_evo.agent.scheduler import IslandScheduler, archive_capabilities
from kernel_evo.core.code.cute_backend_utils import apply_cute_build_env


def _capability_report() -> dict:
    return {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "packages": {"nvidia-cutlass-dsl": "4.2.1"},
        "python": {"version": "3.12"},
        "target_arch": "sm_90a",
        "gpu": {"native_arch": "sm_90a"},
        "cuda": {},
        "features": {"tma": True, "wgmma_bf16": True, "wgmma_fp8": True},
        "tools": {},
        "issues": [],
        "fingerprint": "test-capability",
    }


def test_scheduler_unlocks_dependent_ideas_from_valid_unpromoted_codegen() -> None:
    scheduler = IslandScheduler()
    ideas = (
        {"id": "bootstrap", "summary": "build WGMMA", "codegen_contract": "hopper_wgmma"},
        {
            "id": "tune",
            "summary": "tune TMA stages",
            "codegen_contract": "hopper_wgmma",
            "requires_capability": "hopper_wgmma",
        },
    )
    empty_archive = {"entries": [], "island_elites": {"0": "seed"}, "global_best_id": "seed"}
    locked = scheduler.select_idea(
        backend="cute",
        configured_ideas=ideas,
        iteration=2,
        island=0,
        islands=1,
        archive=empty_archive,
    )
    assert locked["id"] == "bootstrap"

    archive = {
        **empty_archive,
        "entries": [
            {
                "id": "valid-but-slower-wgmma",
                "result": {"valid": True, "speedup": 0.8},
                "cute_context": {"idea_codegen_contract": "hopper_wgmma"},
                "cute_evidence": {"codegen_gate": {"passed": True}},
                "promoted": False,
            }
        ],
    }
    assert archive_capabilities(archive) == {"hopper_wgmma": "valid-but-slower-wgmma"}
    archive["entries"].append(
        {
            "id": "newer-but-worse-wgmma",
            "iteration": 2,
            "result": {"valid": True, "speedup": 0.2},
            "cute_context": {"idea_codegen_contract": "hopper_wgmma"},
            "cute_evidence": {"codegen_gate": {"passed": True}},
        }
    )
    assert archive_capabilities(archive)["hopper_wgmma"] == "valid-but-slower-wgmma"
    unlocked = scheduler.select_idea(
        backend="cute",
        configured_ideas=ideas,
        iteration=2,
        island=0,
        islands=1,
        archive=archive,
    )
    assert unlocked["id"] == "cute-integrate-wgmma-output"
    assert scheduler.select_baseline_entry(
        archive=archive,
        iteration=2,
        island=0,
        migration_interval=4,
        required_capability="hopper_wgmma",
    ) == "valid-but-slower-wgmma"


def test_scheduler_quarantines_parent_after_repeated_profile_failures() -> None:
    scheduler = IslandScheduler()
    archive = {
        "seed": {"id": "seed", "result": {"valid": True}},
        "entries": [
            {"id": "seed", "result": {"valid": True}},
            {
                "id": "unstable-best",
                "result": {"valid": True, "speedup": 1.2},
                "parent_profile_failures": 2,
            },
        ],
        "island_elites": {"0": "unstable-best"},
        "global_best_id": "unstable-best",
    }

    assert (
        scheduler.select_baseline_entry(
            archive=archive,
            iteration=2,
            island=0,
            migration_interval=3,
        )
        == "seed"
    )


def test_scheduler_feedback_omits_invalid_entry_superseded_by_valid_repair() -> None:
    scheduler = IslandScheduler()
    archive = {
        "entries": [
            {
                "id": "iter-003-island-0",
                "island": 0,
                "idea": {"summary": "fuse copies"},
                "result": {
                    "valid": False,
                    "compiled": True,
                    "status": "invalid_compliance",
                    "error": "obsolete executor delta requirement",
                },
            },
            {
                "id": "iter-003-island-0-repair-1",
                "island": 0,
                "idea": {"summary": "fuse copies"},
                "result": {"valid": True, "speedup": 1.1},
                "promoted": True,
            },
        ]
    }

    feedback = scheduler.compact_feedback(archive, island=0)

    assert any("Current island elite" in item for item in feedback)
    assert not any("obsolete executor delta" in item for item in feedback)


def test_scheduler_rotates_compact_parent_profile_ideas() -> None:
    scheduler = IslandScheduler()
    parent = {
        "profile": {
            "torch": {
                "optimization_ideas": [
                    "Fuse adjacent launches.",
                    "Optimize the recurrent kernel.",
                ]
            }
        }
    }

    assert scheduler.select_profile_idea(
        parent_entry=parent, iteration=1, island=0
    )["summary"] == "Fuse adjacent launches."
    assert scheduler.select_profile_idea(
        parent_entry=parent, iteration=2, island=0
    )["summary"] == "Optimize the recurrent kernel."


def test_controller_builds_on_valid_slower_capability_without_promoting_it(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\nimport cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "# MmaF8Op make_tiled_tma_atom PipelineTmaAsync\n"
        "@cute.kernel\ndef kernel(x): pass\n"
        "@cute.jit\ndef launch(x): kernel(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x): self._compiled(from_dlpack(x.detach())); return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report()
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(_context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 0.8,
            "fitness": 0.8,
            "metadata": {
                "cute_environment": {
                    "nvidia_cutlass_dsl": "4.2.1",
                    "target_arch": "sm_90a",
                    "gpu": {"native_arch": "sm_90a"},
                    "features": {"wgmma_fp8": True, "tma": True, "mbarrier": True},
                },
                "cute_runtime": {"executed_executor_count": 1},
                "cute_codegen": [
                    {
                        "kind": "sass",
                        "instruction_families": {
                            "wgmma": 1,
                            "tma": 1,
                            "mbarrier": 1,
                            "local_load": 0,
                            "local_store": 0,
                        },
                        "resources": {"registers": 20},
                    }
                ],
            },
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "fp8",
            "cute_arch": "sm_90a",
            "steps": 2,
            "islands": 1,
            "ideas": [
                {
                    "id": "bootstrap",
                    "summary": "establish WGMMA",
                    "codegen_contract": "hopper_wgmma",
                },
                {
                    "id": "tune",
                    "summary": "tune TMA stages",
                    "codegen_contract": "hopper_wgmma",
                    "requires_capability": "hopper_wgmma",
                },
            ],
        },
        run_id="capability-chain",
    )
    first = controller.prepare_iteration("capability-chain")[0]
    assert first.idea_id == "bootstrap"
    first_report = controller.evaluate_iteration("capability-chain")
    assert first_report["valid_candidates"] == 1, first_report
    assert first_report["promoted_candidates"] == 0
    controller.advance_iteration("capability-chain")
    second = controller.prepare_iteration("capability-chain")[0]
    state = controller.store.read_state("capability-chain")
    second_record = state["iterations"]["2"]["islands"]["0"]
    assert second.idea_id == "cute-integrate-wgmma-output"
    assert second_record["baseline_entry_id"] == "iter-001-island-0"


def test_partial_codegen_progress_is_carried_forward_without_unlocking_or_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\nimport cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "# MmaF8Op make_tiled_tma_atom PipelineTmaAsync\n"
        "@cute.kernel\ndef kernel(x): pass\n"
        "@cute.jit\ndef launch(x): kernel(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x): self._compiled(from_dlpack(x.detach())); return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report()
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(_context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 0.7,
            "fitness": 0.7,
            "metadata": {
                "cute_environment": {
                    "nvidia_cutlass_dsl": "4.2.1",
                    "target_arch": "sm_90a",
                    "gpu": {"native_arch": "sm_90a"},
                    "features": {"wgmma_fp8": True, "tma": True, "mbarrier": True},
                },
                "cute_runtime": {"executed_executor_count": 1},
                "cute_codegen": [
                    {
                        "kind": "sass",
                        "instruction_families": {
                            "wgmma": 1,
                            "tma": 0,
                            "mbarrier": 0,
                            "local_load": 0,
                            "local_store": 0,
                        },
                        "resources": {"registers": 20},
                    }
                ],
            },
        }

    controller = KernelEvoAgent(tmp_path / "partial-runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "fp8",
            "cute_arch": "sm_90a",
            "steps": 2,
            "islands": 1,
            "ideas": [
                {
                    "id": "bootstrap",
                    "summary": "establish WGMMA",
                    "codegen_contract": "hopper_wgmma",
                },
                {
                    "id": "tune",
                    "summary": "tune TMA stages",
                    "codegen_contract": "hopper_wgmma",
                    "requires_capability": "hopper_wgmma",
                },
            ],
        },
        run_id="partial-chain",
    )
    controller.prepare_iteration("partial-chain")
    report = controller.evaluate_iteration("partial-chain")
    assert report["valid_candidates"] == 0
    assert report["promoted_candidates"] == 0
    assert report["islands"][0]["development_score"] > 0.0
    assert "has_wgmma" in report["islands"][0]["development_milestones"]
    state = controller.store.read_state("partial-chain")
    entry = state["archive"]["entries"][0]
    progress = entry["result"]["metadata"]["development_progress"]
    assert progress["milestones"]["has_wgmma"] is True
    assert progress["milestones"]["has_tma"] is False
    assert state["archive"]["development_elites"]["0"] == entry["id"]

    controller.advance_iteration("partial-chain")
    second = controller.prepare_iteration("partial-chain")[0]
    state = controller.store.read_state("partial-chain")
    assert second.idea_id == "bootstrap"
    assert state["iterations"]["2"]["islands"]["0"]["baseline_entry_id"] == entry["id"]


def test_nested_cute_config_and_hopper_fp8_validation() -> None:
    config = AgentRunConfig.from_mapping(
        {
            "baseline": "kernel.py",
            "backend": "cute",
            "precision": "bf16",
            "cute": {"arch": "sm_90a", "context_cards": 5, "context_max_chars": 9000},
        }
    )
    assert config.cute_arch == "sm_90a"
    assert config.cute_context_cards == 5
    assert config.cute_context_max_chars == 9000
    assert config.cute_context_deep_files == 1
    assert config.cute_context_lessons == 3
    assert config.cute_sanitizer_tools == ("memcheck", "synccheck")

    with pytest.raises(ConfigurationError, match="sm_90a"):
        AgentRunConfig.from_mapping(
            {"baseline": "kernel.py", "backend": "cute", "precision": "fp8", "cute_arch": "sm_90"}
        )


def test_cute_island_packet_retrieves_bounded_python_dsl_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text(
        "import torch\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, a, b): return torch.matmul(a, b)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report())
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )
    controller = KernelEvoAgent(tmp_path / "runs")
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
        },
        run_id="cute-context",
    )
    task = controller.prepare_iteration("cute-context")[0]
    assert task.idea_id == "cute-wgmma"
    harness_context = task.task_file.parent / "CUTE_HARNESS.md"
    assert harness_context in task.readable_files
    context = harness_context.read_text(encoding="utf-8")
    assert "Python CuTe DSL" in context
    assert "CuTe C++" in context
    assert "hopper-wgmma-gemm-4_2" in context
    assert "## Navigation" in context
    assert "why now" in context
    assert "Deep reference" in context
    assert "Do not read every card" in context
    assert "## Reading order" in task.task_file.read_text(encoding="utf-8")
    harness_files = [path for path in task.readable_files if "cute_harness" in str(path)]
    assert harness_files
    assert all("src/kernel_evo/cute_harness" in str(path) for path in harness_files)
    deep_kernels = [path for path in harness_files if path.name == "kernel.py"]
    assert len(deep_kernels) <= 1


def test_cute_packet_includes_configured_task_specific_readables_and_test_names(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text(
        "import torch\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, a, b): return torch.matmul(a, b)\n",
        encoding="utf-8",
    )
    author_context = tmp_path / "AUTHOR_CONTEXT.md"
    author_context.write_text("# Exact integration shapes\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_contract.py").write_text(
        "def test_exact_output_shape():\n    assert True\n",
        encoding="utf-8",
    )
    pycache = tests_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "test_contract.cpython-312.pyc").write_bytes(b"not useful to authors")
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities",
        lambda **_: _capability_report(),
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )
    controller = KernelEvoAgent(tmp_path / "runs")
    controller.init_run(
        {
            "baseline": str(baseline),
            "tests": str(tests_dir),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
            "author_readable_files": [str(author_context)],
        },
        run_id="task-readables",
    )
    task = controller.prepare_iteration("task-readables")[0]
    assert author_context.resolve() in task.readable_files
    summary = (task.task_file.parents[1] / "tests" / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "test_exact_output_shape" in summary
    assert "__pycache__" not in summary
    assert ".pyc" not in summary


def test_cute_barrier_records_and_retrieves_compact_local_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def kernel(x): pass\n"
        "@cute.jit\n"
        "def launch(x): kernel(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x):\n"
        "        self._compiled(from_dlpack(x))\n"
        "        return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report())
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(_context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "runtime_us": 8.0,
            "ref_runtime_us": 10.0,
            "speedup": 1.25,
            "fitness": 1.25,
            "metadata": {
                "cute_runtime": {"executed_executor_count": 1},
                "cute_codegen": [
                    {
                        "kind": "sass",
                        "instruction_families": {
                            "wgmma": 1,
                            "tma": 1,
                            "mbarrier": 1,
                            "local_load": 0,
                            "local_store": 0,
                        },
                        "resources": {"registers": 20},
                    }
                ],
            },
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "steps": 2,
            "islands": 1,
        },
        run_id="cute-memory",
    )
    first = controller.prepare_iteration("cute-memory")[0]
    controller.submit_candidate(
        "cute-memory",
        1,
        0,
        first.candidate_path,
        metadata={"idea_summary": "keep the verified copy path"},
    )
    controller.evaluate_iteration("cute-memory")
    experiment_path = Path(controller.status("cute-memory")["run_dir"]) / "cute" / "experiments.jsonl"
    records = experiment_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    controller.advance_iteration("cute-memory")
    second = controller.prepare_iteration("cute-memory")[0]
    context = (second.task_file.parent / "CUTE_HARNESS.md").read_text(encoding="utf-8")
    assert "## Prior local evidence" in context
    assert "`accept` at 1.250x" in context


def test_cute_controller_rejects_torch_only_candidate_even_if_external_evaluator_accepts(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import torch\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, x): return torch.relu(x)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report())
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )
    controller = KernelEvoAgent(
        tmp_path / "runs",
        evaluator=lambda _context: {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 2.0,
            "fitness": 2.0,
        },
    )
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
        },
        run_id="cute-compliance",
    )
    controller.prepare_iteration("cute-compliance")
    report = controller.evaluate_iteration("cute-compliance")
    assert report["valid_candidates"] == 0
    assert "missing-kernel" in report["islands"][0]["error"]


def test_matching_wgmma_hypothesis_uses_retained_codegen_as_a_gate(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import torch\n"
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "# Source preflight markers: MmaF8Op make_tiled_tma_atom PipelineTmaAsync\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def kernel(x): pass\n"
        "@cute.jit\n"
        "def launch(x): kernel(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, a, b):\n"
        "        self._compiled(from_dlpack(a))\n"
        "        return torch.matmul(a, b)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("kernel_evo.cute_harness.catalog.probe_capabilities", lambda **_: _capability_report())
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(_context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 2.0,
            "fitness": 2.0,
            "metadata": {
                "cute_environment": {
                    "nvidia_cutlass_dsl": "4.2.1",
                    "target_arch": "sm_90a",
                    "gpu": {"native_arch": "sm_90a"},
                    "features": {"wgmma_bf16": True},
                },
                "cute_codegen": [
                    {
                        "kind": "sass",
                        "instruction_families": {
                            "wgmma": 0,
                            "tma": 0,
                            "mbarrier": 0,
                            "local_load": 0,
                            "local_store": 0,
                        },
                        "resources": {"registers": 20},
                    }
                ],
            },
        }

    controller = KernelEvoAgent(tmp_path / "runs", evaluator=evaluator)
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
        },
        run_id="cute-codegen",
    )
    task = controller.prepare_iteration("cute-codegen")[0]
    assert task.idea_id == "cute-wgmma"
    report = controller.evaluate_iteration("cute-codegen")
    assert report["valid_candidates"] == 0
    assert "Required `wgmma` missing" in report["islands"][0]["error"]


def test_configured_fuse_requires_wgmma_but_vector_idea_uses_vector_contract(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "# Source preflight markers: MmaF8Op make_tiled_tma_atom PipelineTmaAsync\n"
        "vector_size = 128\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def kernel(x): pass\n"
        "@cute.jit\n"
        "def launch(x): kernel(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x):\n"
        "        self._compiled(from_dlpack(x))\n"
        "        return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities",
        lambda **_: _capability_report(),
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(_context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 1.1,
            "fitness": 1.1,
            "metadata": {
                "cute_environment": {
                    "nvidia_cutlass_dsl": "4.2.1",
                    "target_arch": "sm_90a",
                    "gpu": {"native_arch": "sm_90a"},
                    "features": {"wgmma_fp8": True, "wgmma_bf16": True},
                },
                "cute_runtime": {"executed_executor_count": 1},
                "cute_codegen": [
                    {
                        "kind": "sass",
                        "instruction_families": {
                            "wgmma": 0,
                            "tma": 0,
                            "mbarrier": 0,
                            "vector_global_load_128": 1,
                            "vector_global_store_128": 1,
                            "local_load": 0,
                            "local_store": 0,
                        },
                        "resources": {"registers": 20},
                    }
                ],
            },
        }

    fuse = KernelEvoAgent(tmp_path / "fuse-runs", evaluator=evaluator)
    fuse.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "fp8",
            "cute_arch": "sm_90a",
            "islands": 1,
            "ideas": [
                {
                    "id": "cute-fuse",
                    "summary": "fuse FP8 output projection",
                    "codegen_contract": "hopper_wgmma",
                }
            ],
        },
        run_id="fuse-contract",
    )
    fuse.prepare_iteration("fuse-contract")
    fuse_report = fuse.evaluate_iteration("fuse-contract")
    assert fuse_report["valid_candidates"] == 0
    assert "Required `wgmma` missing" in fuse_report["islands"][0]["error"]

    vector = KernelEvoAgent(tmp_path / "vector-runs", evaluator=evaluator)
    vector.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
            "ideas": [
                {
                    "id": "cute-tv-layout",
                    "summary": "vectorize an output copy",
                    "codegen_contract": "vector",
                }
            ],
        },
        run_id="vector-contract",
    )
    vector.prepare_iteration("vector-contract")
    vector_report = vector.evaluate_iteration("vector-contract")
    assert vector_report["valid_candidates"] == 1

    explicit_only = KernelEvoAgent(tmp_path / "explicit-only-runs", evaluator=evaluator)
    explicit_only.init_run(
        {
            "baseline": str(baseline),
            "backend": "cute",
            "precision": "bf16",
            "cute_arch": "sm_90a",
            "islands": 1,
            "ideas": [
                {
                    "id": "cute-tv-layout",
                    "summary": "layout exploration without a binary contract",
                }
            ],
        },
        run_id="explicit-contract-only",
    )
    task = explicit_only.prepare_iteration("explicit-contract-only")[0]
    assert "--contract" not in task.task_file.read_text(encoding="utf-8")
    explicit_report = explicit_only.evaluate_iteration("explicit-contract-only")
    assert explicit_report["valid_candidates"] == 1


def test_apply_cute_build_env_sets_sm90a_from_torch_arch_list(monkeypatch) -> None:
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    monkeypatch.setattr(
        "kernel_evo.core.code.cuda_backend_utils.apply_cuda_build_env",
        lambda _config: None,
    )
    apply_cute_build_env({"arch_list": "9.0", "precision": "fp8", "device": "cuda:0"})
    import os

    assert os.environ["CUTE_DSL_ARCH"] == "sm_90a"


def test_candidate_owned_kernel_contract_requires_source_delta_and_executor_delta(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def inherited(x): pass\n"
        "@cute.jit\n"
        "def launch(x): inherited(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x):\n"
        "        self._compiled(from_dlpack(x))\n"
        "        return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities",
        lambda **_: _capability_report(),
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 1.0,
            "metadata": {
                "cute_runtime": {
                    "executed_executor_count": 1 if context.iteration == 0 else 2
                }
            },
        }

    config = {
        "baseline": str(baseline),
        "backend": "cute",
        "precision": "bf16",
        "cute_arch": "sm_90a",
        "seed_preflight": True,
        "islands": 1,
        "ideas": [
            {
                "id": "owned",
                "summary": "author a candidate-local kernel",
                "requires_candidate_kernel": True,
                "min_new_executors": 1,
            }
        ],
    }

    unchanged = KernelEvoAgent(tmp_path / "unchanged", evaluator=evaluator)
    unchanged.init_run(config, run_id="unchanged-owned")
    unchanged_task = unchanged.prepare_iteration("unchanged-owned")[0]
    assert "candidate-owned kernel contract" in unchanged_task.task_file.read_text(
        encoding="utf-8"
    )
    unchanged_report = unchanged.evaluate_iteration("unchanged-owned")
    assert unchanged_report["valid_candidates"] == 0
    assert "no new or materially modified" in unchanged_report["islands"][0]["error"]

    changed = KernelEvoAgent(tmp_path / "changed", evaluator=evaluator)
    changed.init_run(config, run_id="changed-owned")
    changed_task = changed.prepare_iteration("changed-owned")[0]
    with changed_task.candidate_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n@cute.kernel\ndef candidate_owned(x): pass\n"
            "@cute.jit\ndef candidate_launch(x):\n"
            "    candidate_owned(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        )
    changed_report = changed.evaluate_iteration("changed-owned")
    assert changed_report["valid_candidates"] == 1, changed_report["islands"][0]
    state = changed.store.read_state("changed-owned")
    evidence = state["iterations"]["1"]["islands"]["0"]["cute_evidence"]
    ownership = evidence["candidate_kernel_ownership"]
    assert ownership["added"] == ["candidate_owned"]
    assert ownership["executed_executor_delta"] == 1
    assert ownership["passed"] is True


def test_candidate_owned_kernel_contract_allows_executor_fusion(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = tmp_path / "candidate.py"
    baseline.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def copy_cache(x): pass\n"
        "@cute.jit\n"
        "def launch(x): copy_cache(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x):\n"
        "        self._compiled(from_dlpack(x))\n"
        "        return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.catalog.probe_capabilities",
        lambda **_: _capability_report(),
    )
    monkeypatch.setattr(
        "kernel_evo.cute_harness.capabilities.probe_capabilities",
        lambda **_: _capability_report(),
    )

    def evaluator(context):
        return {
            "compiled": 1,
            "correctness": 1,
            "is_valid": 1,
            "speedup": 1.0,
            "metadata": {
                "cute_runtime": {
                    "executed_executor_count": 2 if context.iteration == 0 else 1
                }
            },
        }

    config = {
        "baseline": str(baseline),
        "backend": "cute",
        "precision": "bf16",
        "cute_arch": "sm_90a",
        "seed_preflight": True,
        "islands": 1,
        "ideas": [
            {
                "id": "fuse-owned",
                "summary": "fuse candidate-local executors",
                "requires_candidate_kernel": True,
                "min_new_executors": 1,
            }
        ],
    }

    agent = KernelEvoAgent(tmp_path / "fused", evaluator=evaluator)
    agent.init_run(config, run_id="fused-owned")
    task = agent.prepare_iteration("fused-owned")[0]
    task.candidate_path.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "@cute.kernel\n"
        "def copy_caches(x): pass\n"
        "@cute.jit\n"
        "def launch(x): copy_caches(x).launch(grid=[1,1,1], block=[1,1,1])\n"
        "class ModelNew:\n"
        "    _compiled = cute.compile(launch, None)\n"
        "    def forward(self, x):\n"
        "        self._compiled(from_dlpack(x))\n"
        "        return x\n",
        encoding="utf-8",
    )

    report = agent.evaluate_iteration("fused-owned")

    assert report["valid_candidates"] == 1, report["islands"][0]
    state = agent.store.read_state("fused-owned")
    ownership = state["iterations"]["1"]["islands"]["0"]["cute_evidence"][
        "candidate_kernel_ownership"
    ]
    assert ownership["executed_executor_delta"] == -1
    assert ownership["candidate_executed_executor_count"] == 1
    assert ownership["passed"] is True


def test_command_evaluator_compile_check_is_rendered_for_the_author(tmp_path: Path) -> None:
    command = _compile_check_command(
        config={
            "evaluator_command": [
                "python",
                "evaluate.py",
                "--candidate",
                "{candidate}",
                "--artifact-dir",
                "{island_dir}/artifacts",
            ]
        },
        run_dir=tmp_path / "run",
        island_dir=tmp_path / "run" / "iter_001" / "island_0",
        candidate_path=tmp_path / "candidate.py",
    )
    assert "--compile-check" in command
    assert "compile_check_artifacts" in command
    assert str((tmp_path / "candidate.py").resolve()) in command
