"""Executable laboratory for NVIDIA's Python CuTe DSL.

The harness is deliberately scoped to ``nvidia-cutlass-dsl`` and ``cutlass.cute``.
It does not index CuTe C++, the legacy CUTLASS Python operation API, or examples
from a different DSL dialect.
"""

from kernel_evo.cute_harness.capabilities import probe_capabilities, resolve_target_arch
from kernel_evo.cute_harness.catalog import AgentContextBundle, build_agent_context, search_catalog
from kernel_evo.cute_harness.correctness import build_correctness_contract
from kernel_evo.cute_harness.feasibility import check_hopper_gemm_config
from kernel_evo.cute_harness.lint import lint_cute_source
from kernel_evo.cute_harness.paths import harness_root
from kernel_evo.cute_harness.task_spec import extract_task_spec

__all__ = [
    "AgentContextBundle",
    "build_agent_context",
    "build_correctness_contract",
    "check_hopper_gemm_config",
    "extract_task_spec",
    "harness_root",
    "lint_cute_source",
    "probe_capabilities",
    "resolve_target_arch",
    "search_catalog",
]
