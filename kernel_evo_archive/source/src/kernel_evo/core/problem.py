"""Shared problem preparation for headless and agent-driven evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kernel_evo.core.precision import resolve_runtime_precision_string
from kernel_evo.resources.workspace import ProblemWorkspace


@dataclass(frozen=True, slots=True)
class ProblemSources:
    """Reference source split into the pieces used by prompts and validation."""

    ref_arch_src: str
    model_src: str
    inputs_src: str
    kind: str
    source_path: str = ""


def resolve_problem_file(problem_path: str | Path) -> Path:
    path = Path(problem_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "task.py"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Custom problem dir must contain task.py: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Custom problem file not found: {path}")
    return path


def load_problem_sources(
    *,
    problem_path: str = "",
    level: int | None = None,
    problem_id: int | None = None,
    dataset_src: str = "huggingface",
    dataset_name: str = "ScalingIntelligence/KernelBench",
    backend: str = "triton",
) -> ProblemSources:
    """Load one KernelBench-format problem without choosing an execution frontend."""
    from kernel_evo.core.code import python_backend_utils

    backend_name = str(backend).lower().strip()
    if backend_name not in python_backend_utils.PYTHON_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. Only triton, cuda_inline and cute are supported; "
            "cpp/cuda/torch are not variants."
        )

    source_path = ""
    if str(problem_path).strip():
        problem_file = resolve_problem_file(problem_path)
        ref_arch_src = problem_file.read_text(encoding="utf-8")
        problem_kind = "custom"
        source_path = str(problem_file)
    else:
        if level is None or problem_id is None:
            raise ValueError("Must provide either problem_path or both level and problem_id.")

        from kernelbench.dataset import construct_kernelbench_dataset  # type: ignore[import-not-found]

        dataset = construct_kernelbench_dataset(
            level=level,
            source=dataset_src,
            dataset_name=dataset_name,
        )
        ref_arch_src = dataset.get_problem_by_id(problem_id).code
        problem_kind = "kernelbench"

    model_src, inputs_src = python_backend_utils.split_kernelbench_ref(ref_arch_src)
    return ProblemSources(
        ref_arch_src=ref_arch_src,
        model_src=model_src,
        inputs_src=inputs_src,
        kind=problem_kind,
        source_path=source_path,
    )


def build_task_description_for_backend(
    *,
    run_cfg: dict[str, Any],
    ref_arch_src: str,
    ref_model_class_src: str,
    ref_inputs_init_src: str,
) -> str:
    """Build the same backend contract for every evolution frontend."""
    from kernel_evo.core.code.cuda_backend_utils import build_task_description_cuda_inline, is_cuda_inline_backend
    from kernel_evo.core.code.cute_backend_utils import build_task_description_cute, is_cute_backend
    from kernel_evo.core.code.python_backend_utils import build_task_description_python

    backend = str(run_cfg.get("backend", "")).lower().strip()
    if is_cuda_inline_backend(backend):
        return build_task_description_cuda_inline(
            run_cfg=run_cfg,
            ref_arch_src=ref_arch_src,
            ref_model_class_src=ref_model_class_src,
            ref_inputs_init_src=ref_inputs_init_src,
        )
    if is_cute_backend(backend):
        return build_task_description_cute(
            run_cfg=run_cfg,
            ref_arch_src=ref_arch_src,
            ref_model_class_src=ref_model_class_src,
            ref_inputs_init_src=ref_inputs_init_src,
        )
    return build_task_description_python(
        run_cfg=run_cfg,
        ref_arch_src=ref_arch_src,
        ref_model_class_src=ref_model_class_src,
        ref_inputs_init_src=ref_inputs_init_src,
    )


def build_seed_program(*, backend: str, model_src: str) -> str:
    """Create the compliant initial program used by both CLI modes."""
    from kernel_evo.core.code.cute_backend_utils import build_cute_seed, is_cute_backend
    from kernel_evo.core.code.python_backend_utils import model_to_modelnew

    if is_cute_backend(backend):
        return build_cute_seed(model_src)
    return model_to_modelnew(model_src)


def write_initial_seed(problem_dir: Path, *, program_code: str, note: str) -> Path:
    seed_path = problem_dir / "initial_programs" / "seed.py"
    seed_code = (
        "# Auto-generated seed program.\n"
        f"# {note}\n"
        "#\n"
        "# IMPORTANT: this file is evaluated directly (no entrypoint wrapper).\n"
        "# It must define `class ModelNew(torch.nn.Module)`.\n\n"
        + program_code.rstrip()
        + "\n"
    )
    seed_path.write_text(seed_code, encoding="utf-8")
    return seed_path


def _value(options: Mapping[str, Any], key: str, default: Any) -> Any:
    value = options.get(key, default)
    return default if value is None else value


def build_validation_config(
    *,
    sources: ProblemSources,
    problem_dir: Path,
    experiment_dir: Path,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the validator context independently of argparse and GigaEvo."""
    precision = str(_value(options, "precision", "fp32"))
    profile_runners_value = _value(options, "profile_runners", [])
    if isinstance(profile_runners_value, str):
        profile_runners = [item.strip() for item in profile_runners_value.split(",") if item.strip()]
    else:
        profile_runners = [str(item).strip() for item in profile_runners_value if str(item).strip()]

    run_cfg: dict[str, Any] = {
        "dataset_src": str(_value(options, "dataset_src", "huggingface")),
        "dataset_name": str(_value(options, "dataset_name", "ScalingIntelligence/KernelBench")),
        "level": int(_value(options, "level", 0) or 0),
        "problem_id": int(_value(options, "problem_id", 0) or 0),
        "problem_kind": sources.kind,
        "problem_path": sources.source_path,
        "problem_dir": str(problem_dir),
        "backend": str(_value(options, "backend", "triton")),
        "codegen_kind": "python",
        "precision": precision,
        "runtime_precision": resolve_runtime_precision_string(
            precision,
            str(_value(options, "runtime_precision", "") or ""),
        ),
        "measurement_mode": str(_value(options, "measurement_mode", "wall-clock")),
        "timing_method": str(_value(options, "timing_method", "cuda_event")),
        "num_correct_trials": int(_value(options, "num_correct_trials", 5)),
        "num_perf_trials": int(_value(options, "num_perf_trials", 100)),
        "output_rtol": _optional_float(_value(options, "output_rtol", 0.01)),
        "output_atol": _optional_float(_value(options, "output_atol", 0.01)),
        "custom_tests": str(_value(options, "custom_tests", "") or ""),
        "device": str(_value(options, "device", "cuda:0")),
        "cute_harness_enabled": bool(_value(options, "cute_harness_enabled", True)),
        "cute_arch": str(_value(options, "cute_arch", "") or ""),
        "cute_context_cards": int(_value(options, "cute_context_cards", 7)),
        "cute_context_max_chars": int(_value(options, "cute_context_max_chars", 10_000)),
        "cute_context_deep_files": int(_value(options, "cute_context_deep_files", 1)),
        "cute_context_lessons": int(_value(options, "cute_context_lessons", 3)),
        "cute_keep_ir": bool(_value(options, "cute_keep_ir", False)),
        "cute_optimization_warnings": bool(_value(options, "cute_optimization_warnings", False)),
        "cute_capability_gate": bool(_value(options, "cute_capability_gate", True)),
        "cute_compliance_gate": bool(_value(options, "cute_compliance_gate", True)),
        "cute_codegen_gate": bool(_value(options, "cute_codegen_gate", True)),
        "cute_record_experiments": bool(_value(options, "cute_record_experiments", True)),
        "cute_required_version": str(_value(options, "cute_required_version", "") or ""),
        "cute_sanitizer_tools": _string_list(
            _value(options, "cute_sanitizer_tools", ("memcheck", "synccheck"))
        ),
        "ref_arch_src": sources.ref_arch_src,
        "ref_model_class_src": sources.model_src,
        "ref_inputs_init_src": sources.inputs_src,
        "validator_debug": bool(_value(options, "validator_debug", False)),
        "validator_debug_dir": str(_value(options, "validator_debug_dir", experiment_dir / "validate_logs")),
        "validator_debug_max_code_chars": int(_value(options, "validator_debug_max_code_chars", 50_000)),
        "validator_transient_retries": int(_value(options, "validator_transient_retries", 3)),
        "validator_transient_retry_delay": float(_value(options, "validator_transient_retry_delay", 1.0)),
        "stdout_dir": str(_value(options, "stdout_dir", "") or ""),
        "experiment_dir": str(experiment_dir),
        "execution_mode": str(_value(options, "execution_mode", "local_execution")),
        "remote_validator_url": str(_value(options, "remote_validator_url", "http://localhost:15000")),
        "remote_poll_interval": float(_value(options, "remote_poll_interval", 1.0)),
        "use_memory_for_errors": bool(_value(options, "use_memory_for_errors", False)),
        "profile_stage_enabled": bool(_value(options, "profile_stage_enabled", False)),
        "profile_runners": profile_runners,
        "profile_max_insights": int(_value(options, "profile_max_insights", 4)),
        "profile_torch_warmup_steps": int(_value(options, "profile_torch_warmup_steps", 2)),
        "profile_torch_active_steps": int(_value(options, "profile_torch_active_steps", 3)),
        "profile_ncu_path": str(_value(options, "profile_ncu_path", "ncu")),
        "profile_ncu_tmpdir": str(_value(options, "profile_ncu_tmpdir", "") or "").strip(),
        "profile_subprocess_timeout": float(
            _value(options, "profile_subprocess_timeout", 600.0)
        ),
        "profile_ncu_set": str(_value(options, "profile_ncu_set", "full") or "full").strip() or "full",
        "profile_ncu_kernel_name": str(_value(options, "profile_ncu_kernel_name", "") or "").strip(),
        "profile_ncu_extra_args": str(_value(options, "profile_ncu_extra_args", "") or "").strip(),
        "profile_ncu_target_steps": max(
            1, int(_value(options, "profile_ncu_target_steps", 1))
        ),
        "profile_ncu_warmup_steps": max(
            0, int(_value(options, "profile_ncu_warmup_steps", 1))
        ),
        "profile_ncu_launch_count": max(
            0, int(_value(options, "profile_ncu_launch_count", 128))
        ),
        "profile_ncu_min_speedup": float(_value(options, "profile_ncu_min_speedup", 1.0)),
        "profile_artifacts_dir": str(_value(options, "profile_artifacts_dir", experiment_dir / "artifacts")),
    }
    arch_list = str(_value(options, "arch_list", "") or "").strip()
    if arch_list:
        run_cfg["arch_list"] = arch_list
    return run_cfg


def write_problem_artifacts(
    *,
    workspace: ProblemWorkspace,
    sources: ProblemSources,
    run_cfg: dict[str, Any],
    seed_note: str,
) -> Path:
    """Persist validator context, authoring contract and initial candidate."""
    workspace.run_config_file.write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    workspace.task_description_file.write_text(
        build_task_description_for_backend(
            run_cfg=run_cfg,
            ref_arch_src=sources.ref_arch_src,
            ref_model_class_src=sources.model_src,
            ref_inputs_init_src=sources.inputs_src,
        ),
        encoding="utf-8",
    )
    return write_initial_seed(
        workspace.root_dir,
        program_code=build_seed_program(backend=str(run_cfg["backend"]), model_src=sources.model_src),
        note=seed_note,
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
