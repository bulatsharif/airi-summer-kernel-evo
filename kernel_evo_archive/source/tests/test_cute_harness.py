from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kernel_evo.cute_harness.api import lookup_api
from kernel_evo.cute_harness.capabilities import (
    capability_fingerprint,
    capability_issues,
    normalize_cute_arch,
)
from kernel_evo.cute_harness.catalog import load_catalog, search_catalog
from kernel_evo.cute_harness.cli import main as cute_main
from kernel_evo.cute_harness.codegen import inspect_text, verify_codegen
from kernel_evo.cute_harness.correctness import build_correctness_contract
from kernel_evo.cute_harness.experiments import query_experiments, record_experiment
from kernel_evo.cute_harness.feasibility import check_hopper_gemm_config
from kernel_evo.cute_harness.layout import probe_cute_layout, probe_layout
from kernel_evo.cute_harness.lint import candidate_kernel_delta, lint_cute_source
from kernel_evo.cute_harness.runner import run_command
from kernel_evo.cute_harness.task_spec import extract_task_spec
from kernel_evo.core.code.cute_backend_utils import build_cute_seed
from kernel_evo.core.eval.gpu_guard import acquire_idle_gpu
from kernel_evo.tools.sanitize_candidate import _error_count, _parse_tools


def test_candidate_kernel_delta_distinguishes_imports_from_owned_kernel_changes() -> None:
    baseline = "import cutlass.cute as cute\n@cute.kernel\ndef inherited(x):\n    return x\n"
    imported_only = baseline + "from reference import HopperKernel\n"
    assert candidate_kernel_delta(imported_only, baseline)["changed"] == []

    candidate = baseline + "@cute.kernel\ndef owned(x):\n    return x\n"
    evidence = candidate_kernel_delta(candidate, baseline)
    assert evidence["added"] == ["owned"]
    assert evidence["modified"] == []

    modified = baseline.replace("return x", "return x[0]")
    assert candidate_kernel_delta(modified, baseline)["modified"] == ["inherited"]


def test_gpu_guard_requires_repeated_idle_samples(tmp_path: Path, monkeypatch) -> None:
    samples = iter(
        [
            {
                "utilization_gpu": 0,
                "compute_processes": [{"pid": 999_999, "name": "other"}],
            },
            {"utilization_gpu": 0, "compute_processes": []},
            {"utilization_gpu": 0, "compute_processes": []},
        ]
    )
    monkeypatch.setattr(
        "kernel_evo.core.eval.gpu_guard._probe_gpu", lambda _index: next(samples)
    )
    lease = acquire_idle_gpu(
        "cuda:3",
        timeout=1.0,
        poll_interval=0.01,
        consecutive_idle_samples=2,
        lock_dir=tmp_path,
    )
    try:
        assert lease.metadata["samples"] == 3
        assert lease.metadata["consecutive_idle_samples"] == 2
    finally:
        lease.close()


def test_gpu_guard_can_accept_idle_co_resident_process(tmp_path: Path, monkeypatch) -> None:
    state = {
        "utilization_gpu": 0,
        "compute_processes": [{"pid": 999_999, "name": "idle-resident"}],
    }
    monkeypatch.setattr("kernel_evo.core.eval.gpu_guard._probe_gpu", lambda _index: state)
    lease = acquire_idle_gpu(
        "cuda:0",
        timeout=1.0,
        poll_interval=0.01,
        consecutive_idle_samples=2,
        lock_dir=tmp_path,
        allow_co_resident=True,
    )
    try:
        assert lease.verify_exclusive() is True
        assert lease.metadata["exclusive_through_end"] is False
        assert lease.metadata["co_resident_accepted"] is True
    finally:
        lease.close()


def test_gpu_guard_trusts_profile_coordinator_pid(tmp_path: Path, monkeypatch) -> None:
    coordinator_pid = 999_998
    state = {
        "utilization_gpu": 0,
        "compute_processes": [{"pid": coordinator_pid, "name": "kernel-evo"}],
    }
    monkeypatch.setenv("KERNELEVO_TRUSTED_GPU_PIDS", str(coordinator_pid))
    monkeypatch.setattr("kernel_evo.core.eval.gpu_guard._probe_gpu", lambda _index: state)

    lease = acquire_idle_gpu(
        "cuda:0",
        timeout=1.0,
        poll_interval=0.01,
        consecutive_idle_samples=2,
        lock_dir=tmp_path,
    )
    try:
        assert coordinator_pid in lease.metadata["trusted_gpu_pids"]
        assert lease.verify_exclusive() is True
    finally:
        lease.close()


def test_arch_normalization_preserves_missing_hopper_feature_suffix() -> None:
    assert normalize_cute_arch("sm_90") == "sm_90"
    assert normalize_cute_arch("sm_90a") == "sm_90a"
    assert normalize_cute_arch("9.0", torch_style=True) == "sm_90a"
    assert normalize_cute_arch("90") == "sm_90a"


def test_lint_explicit_codegen_contract_is_a_blocking_source_preflight() -> None:
    source = "import cutlass\nimport cutlass.cute as cute\n"
    report = lint_cute_source(
        source,
        precision="fp8",
        arch="sm_90a",
        operation="gemm",
        codegen_contract="hopper_wgmma",
    )
    codes = {item["code"] for item in report["issues"] if item["severity"] == "error"}
    assert {
        "contract-without-wgmma-source",
        "contract-without-tma-source",
        "contract-without-mbarrier-source",
    }.issubset(codes)


def test_lint_without_explicit_contract_does_not_invent_one_from_the_idea() -> None:
    report = lint_cute_source(
        "import cutlass\nimport cutlass.cute as cute\n",
        precision="fp8",
        arch="sm_90a",
        operation="gemm",
    )
    assert not any(item["code"].startswith("contract-without-") for item in report["issues"])


def test_cute_seed_generation_keeps_future_import_legal() -> None:
    wrapped = build_cute_seed(
        "from __future__ import annotations\n"
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x): return x\n"
    )
    compile(wrapped, "<cute-seed>", "exec")
    provided = (
        "from __future__ import annotations\n"
        "import cutlass.cute as cute\n"
        "@cute.kernel\n"
        "def kernel(x): pass\n"
        "class ModelNew: pass\n"
    )
    assert build_cute_seed(provided) == provided


def test_catalog_is_python_dsl_only_and_retrieves_fp8_wgmma() -> None:
    assert load_catalog()
    assert {entry.dialect for entry in load_catalog()} == {"cute_dsl_python"}
    entries = search_catalog(
        precision="fp8",
        arch="sm_90a",
        operation="gemm",
        concepts=["wgmma", "tma"],
        version="4.2.1",
    )
    ids = {entry.id for entry in entries}
    assert "fp8-wgmma-sm90-4_2" in ids
    assert "hopper-wgmma-gemm-4_2" in ids
    gemm = next(entry for entry in entries if entry.id == "hopper-wgmma-gemm-4_2")
    assert gemm.why
    assert gemm.use_when
    assert gemm.deep_files and gemm.deep_files[0].name == "kernel.py"

    copy_entries = search_catalog(
        precision="bf16",
        arch="sm_90a",
        operation="elementwise",
        concepts=["vectorization", "predication"],
        version="4.2.1",
    )
    copy_ids = {entry.id for entry in copy_entries}
    assert {"bf16-vector-add-4_2", "bf16-vector-add-aligned-4_2"} <= copy_ids


def test_layout_probe_maps_coordinate_and_reports_cosize() -> None:
    result = probe_layout((4, 8), stride=(8, 1), coordinates=[(2, 3)])
    assert result["rank"] == 2
    assert result["cosize"] == 32
    assert result["mapping"] == [{"coordinate": [2, 3], "linear_index": 19}]


def test_codegen_groups_hopper_qgmma_with_wgmma_and_warns_on_spills() -> None:
    result = inspect_text(
        "QGMMA.64x64x32.F32.E4M3.E4M3 R24; UTMALDG.3D; LDL R2;",
        expected=["wgmma", "tma"],
    )
    assert result["instruction_families"]["wgmma"] == 1
    assert result["instruction_families"]["tma"] == 1
    assert result["expectations"] == {"wgmma": True, "tma": True}
    assert any("spilling" in warning for warning in result["warnings"])
    vector = inspect_text("LDG.E.128 R4, desc[UR4][R2.64]; STG.E.128 desc[UR4][R2.64], R4; HFMA2.MMA.BF16_V2 R1;")
    assert vector["instruction_families"]["vector_global_load_128"] == 1
    assert vector["instruction_families"]["vector_global_store_128"] == 1
    assert vector["instruction_families"]["mma_sync"] == 0


def test_codegen_contract_is_an_evidence_gate() -> None:
    report = inspect_text("QGMMA.64x64x32.F32.E4M3.E4M3; UTMALDG.3D;")
    report["resources"] = {"registers": 64}
    verified = verify_codegen(
        report,
        {
            "required_instruction_families": ["wgmma", "tma"],
            "forbidden_instruction_families": ["local_store"],
            "resource_limits": {"registers_per_thread": 80},
        },
    )
    assert verified["passed"] is True
    failed = verify_codegen(report, {"required_instruction_families": ["mbarrier"]})
    assert failed["passed"] is False
    assert failed["failures"][0]["name"] == "mbarrier"


def test_lookup_uses_exact_installed_python_dsl() -> None:
    pytest.importorskip("cutlass.cute")
    result = lookup_api("cute.make_layout", max_usages=0)
    assert result["dialect"] == "cute_dsl_python"
    assert result["canonical_symbol"] == "cutlass.cute.make_layout"
    assert "shape" in result["signature"]


def test_structured_runner_captures_final_json_metrics(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import json; print(json.dumps({'correctness': True}))"],
        kind="check",
        arch="sm_90a",
        cwd=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        timeout=30,
    )
    assert result["success"] is True
    assert result["metrics"] == {"correctness": True}
    assert result["environment"]["arch"] == "sm_90a"


def test_experiment_memory_is_dialect_scoped(tmp_path: Path) -> None:
    database = tmp_path / "runs.jsonl"
    record_experiment(
        database,
        {
            "task": "fp8_gemm",
            "hypothesis": "one more stage hides TMA latency",
            "change": {"stages": [2, 3]},
            "decision": "reject",
            "lesson_tags": ["pipeline_stages"],
        },
    )
    results = query_experiments(database, task="fp8", tag="pipeline_stages")
    assert len(results) == 1
    assert results[0]["dialect"] == "cute_dsl_python"


def test_standalone_cli_probes_layout(capsys) -> None:
    cute_main(["probe-layout", "--shape", "4,8", "--stride", "8,1", "--coord", "2,3"])
    value = json.loads(capsys.readouterr().out)
    assert value["mapping"][0]["linear_index"] == 19


def test_installed_dsl_layout_probe_traces_hierarchical_divide() -> None:
    pytest.importorskip("cutlass.cute")
    value = probe_cute_layout((4, 8), stride=(8, 1), coordinate=(2, 3), tile=(2, 4))
    assert value["success"] is True
    assert "logical_divide:" in str(value["trace"])
    assert value["host_mapping"]["mapping"][0]["linear_index"] == 19


def test_task_spec_and_lint_provide_bounded_routing_facts() -> None:
    source = """
import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return torch.softmax(torch.matmul(a, b), dim=-1)
"""
    spec = extract_task_spec(source, precision="bf16", arch="sm_90a")
    assert spec["operation"] == "attention"
    assert spec["inputs"] == ["a", "b"]
    assert "torch.matmul" in spec["source_operations"]
    assert extract_task_spec("class HopperGemmKernel: pass")["operation"] == "gemm"

    candidate = """
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
@cute.kernel
def kernel(x):
    pass
@cute.jit
def launch(x):
    kernel(x).launch(grid=[1, 1, 1], block=[1, 1, 1])
class ModelNew:
    def forward(self, x):
        compiled = cute.compile(launch, from_dlpack(x))
        compiled(from_dlpack(x))
        return x.contiguous()
"""
    lint = lint_cute_source(candidate, precision="fp8", arch="sm_90a", operation="gemm")
    codes = {item["code"] for item in lint["issues"]}
    assert {"compile-in-forward", "forward-contiguous", "fp8-without-wgmma"} <= codes
    assert lint["counts"]["error"] == 0


def test_lint_follows_compiled_executor_cached_on_instance_attribute() -> None:
    candidate = """
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
@cute.kernel
def kernel(x):
    pass
@cute.jit
def launch(x):
    kernel(x).launch(grid=[1, 1, 1], block=[1, 1, 1])
class ModelNew:
    def prepare_for_evaluation(self, x):
        compiled = cute.compile(launch, from_dlpack(x))
        self._compiled = compiled
    def forward(self, x):
        self._compiled(from_dlpack(x))
        return x
"""
    lint = lint_cute_source(candidate, precision="bf16", arch="sm_90a", operation="elementwise")
    codes = {item["code"] for item in lint["issues"]}
    assert "compiled-executor-unused" not in codes
    assert lint["counts"]["error"] == 0


def test_hopper_feasibility_and_correctness_contract_explain_boundaries() -> None:
    feasible = check_hopper_gemm_config(
        tile_shape_mnk=(128, 128, 64),
        cluster_shape_mn=(1, 1),
        stages=2,
        dtype="fp8",
    )
    assert feasible["feasible"] is True
    assert feasible["instruction"]["wgmma_k"] == 32
    rejected = check_hopper_gemm_config(
        tile_shape_mnk=(96, 128, 48),
        cluster_shape_mn=(3, 1),
        stages=5,
        dtype="bf16",
    )
    assert rejected["feasible"] is False
    assert {item["code"] for item in rejected["issues"]} >= {"tile-m", "cluster-shape"}

    contract = build_correctness_contract(
        operation="attention",
        precision="fp8",
        tile_shape=(128, 128, 64),
        supports_strides=True,
    )
    case_ids = {item["id"] for item in contract["cases"]}
    assert {"tile-boundaries", "nontrivial-strides", "masking", "fp8-quantization"} <= case_ids


def test_capability_identity_checks_are_stable_and_explicit() -> None:
    identity = {
        "nvidia_cutlass_dsl": "4.2.1",
        "target_arch": "sm_90",
        "gpu": {"native_arch": "sm_90a"},
        "features": {"wgmma_fp8": False},
    }
    assert capability_fingerprint(identity) == capability_fingerprint(dict(identity))
    codes = {
        item["code"]
        for item in capability_issues(identity, precision="fp8", required_arch="sm_90a")
    }
    assert {"target-arch-mismatch", "missing-wgmma-capability"} <= codes


def test_sanitizer_plan_forces_memcheck_before_sync_checks() -> None:
    assert _parse_tools("synccheck,racecheck") == ["memcheck", "racecheck", "synccheck"]
    assert _error_count("========= ERROR SUMMARY: 3 errors") == 3
