from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule

from kernel_evo.core.eval.eval import (
    _process_input_tensor,
    get_torch_dtype_from_string,
    graceful_eval_cleanup,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
    set_seed,
)
from kernel_evo.core.precision import resolve_runtime_precision_string


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _first_available(event: Any, *names: str) -> float:
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return _as_float(value)
    return 0.0


def _event_to_dict(event: Any) -> dict[str, Any]:
    input_shapes = getattr(event, "input_shapes", None)
    if input_shapes is None:
        input_shapes = []
    self_device_time_total_us = _first_available(
        event,
        "self_device_time_total",
        "self_cuda_time_total",
        "self_privateuse1_time_total",
    )
    device_time_total_us = _first_available(
        event,
        "device_time_total",
        "cuda_time_total",
        "privateuse1_time_total",
    )
    return {
        "key": str(getattr(event, "key", "")),
        "count": int(getattr(event, "count", 0) or 0),
        "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
        "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0) or 0.0),
        "self_cuda_time_total_us": self_device_time_total_us,
        "cuda_time_total_us": device_time_total_us,
        "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0),
        "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0) or 0),
        "input_shapes": input_shapes,
    }


def _build_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    total_self_cuda = sum(item["self_cuda_time_total_us"] for item in events)
    total_cuda = sum(item["cuda_time_total_us"] for item in events)
    top_ops: list[dict[str, Any]] = []
    heuristics: list[dict[str, str]] = []

    for item in events[:10]:
        share = (item["self_cuda_time_total_us"] / total_self_cuda) if total_self_cuda > 0 else 0.0
        top_ops.append(
            {
                "name": item["key"],
                "count": item["count"],
                "self_cuda_time_total_us": item["self_cuda_time_total_us"],
                "cuda_time_total_us": item["cuda_time_total_us"],
                "share_of_self_cuda_time": share,
                "self_device_memory_usage_bytes": item["self_device_memory_usage_bytes"],
            }
        )

    if top_ops:
        hottest = top_ops[0]
        if hottest["share_of_self_cuda_time"] >= 0.6:
            heuristics.append(
                {
                    "kind": "single_hotspot",
                    "detail": f"{hottest['name']} dominates self CUDA time ({hottest['share_of_self_cuda_time']:.1%}).",
                }
            )

    copies = [item for item in top_ops if "copy" in item["name"].lower() or "to" in item["name"].lower()]
    if copies:
        heuristics.append(
            {
                "kind": "data_movement",
                "detail": "Profiler saw copy/to-style operators in the hot path.",
            }
        )

    return {
        "status": "completed",
        "total_self_cuda_time_us": total_self_cuda,
        "total_cuda_time_us": total_cuda,
        "top_ops": top_ops,
        "heuristics": heuristics,
    }


def _trace_device_breakdown(trace_path: Path, active_steps: int) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
    kernels: dict[str, dict[str, float | int | str]] = {}
    kernel_time = 0.0
    memcpy_time = 0.0
    kernel_count = 0
    memcpy_count = 0
    allocation_count = 0
    for event in trace_events if isinstance(trace_events, list) else []:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        duration = _as_float(event.get("dur", 0.0))
        if "kernel" in category and duration > 0:
            kernel_time += duration
            kernel_count += 1
            aggregate = kernels.setdefault(name, {"name": name, "count": 0, "time_us": 0.0})
            aggregate["count"] = int(aggregate["count"]) + 1
            aggregate["time_us"] = float(aggregate["time_us"]) + duration
        elif "memcpy" in category and duration > 0:
            memcpy_time += duration
            memcpy_count += 1
        if name == "[memory]":
            allocation_count += 1
    divisor = max(1, active_steps)
    top_operations = []
    for item in sorted(kernels.values(), key=lambda value: float(value["time_us"]), reverse=True)[:12]:
        top_operations.append(
            {
                "name": item["name"],
                "count_per_forward": float(item["count"]) / divisor,
                "time_us_per_forward": float(item["time_us"]) / divisor,
            }
        )
    return {
        "active_device_time_us": (kernel_time + memcpy_time) / divisor,
        "kernel_time_us": kernel_time / divisor,
        "memcpy_time_us": memcpy_time / divisor,
        "kernel_count": kernel_count / divisor,
        "memcpy_count": memcpy_count / divisor,
        "dynamic_allocation_events": allocation_count / divisor,
        "top_operations": top_operations,
    }


def _source_graph_violations(source: str) -> list[str]:
    """Return concise source-level hazards; CUDA capture remains the authoritative gate."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["candidate source could not be parsed for graph-safety checks"]
    hazards: set[str] = set()
    allocation_calls = {"empty", "empty_like", "zeros", "zeros_like", "ones", "full", "new_empty"}
    compile_calls = {"compile", "load", "load_inline"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "forward":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = ""
                if isinstance(child.func, ast.Name):
                    name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    name = child.func.attr
                if name in allocation_calls:
                    hazards.add(f"per-forward dynamic allocation via {name}()")
                if name in compile_calls:
                    hazards.add(f"per-forward compilation/loading via {name}()")
                if name in {"MethodType", "setattr"}:
                    hazards.add(f"per-forward monkey-patching via {name}()")
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "forward":
                        hazards.add("per-forward monkey-patching of .forward")
    return sorted(hazards)


def _event_average_us(fn: Any, *, steps: int, device: torch.device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / max(1, steps))


def _graph_replay_objective(
    custom_model: Any,
    inputs: list[Any],
    *,
    steps: int,
    device: torch.device,
    source_violations: list[str],
) -> dict[str, Any]:
    graph = None
    try:
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                custom_model(*inputs)
        torch.cuda.current_stream(device=device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            custom_model(*inputs)
        replay_us = _event_average_us(graph.replay, steps=steps, device=device)
        return {
            "capturable": not source_violations,
            "capture_succeeded": True,
            "replay_us": replay_us,
            "source_violations": source_violations,
            "failure": "" if not source_violations else "; ".join(source_violations),
        }
    except Exception as exc:
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        return {
            "capturable": False,
            "capture_succeeded": False,
            "replay_us": None,
            "source_violations": source_violations,
            "failure": f"{type(exc).__name__}: {exc}"[:1_000],
        }


def _allocation_graph_failure(
    allocation_events: float, source_violations: list[str]
) -> dict[str, Any] | None:
    if allocation_events <= 0:
        return None
    failure = f"profiler observed {allocation_events:.1f} dynamic allocation event(s) per forward"
    return {
        "capturable": False,
        "capture_succeeded": False,
        "replay_us": None,
        "source_violations": source_violations,
        "failure": "; ".join([*source_violations, failure]),
        "capture_skipped": "dynamic_allocations",
    }


def _optimization_ideas(objectives: dict[str, Any]) -> list[str]:
    inner = objectives.get("inner_kernel", {})
    eager = objectives.get("eager_complete_layer", {})
    graph = objectives.get("cuda_graph_complete_layer", {})
    ideas: list[str] = []
    active = _as_float(inner.get("active_device_time_us"))
    gaps = _as_float(eager.get("inferred_dispatch_gaps_us"))
    if gaps > max(5.0, active * 0.1):
        ideas.append("Reduce launch/dispatch gaps by fusing adjacent work and removing Python from forward.")
    if _as_float(inner.get("memcpy_count")) > 0:
        ideas.append("Remove or fuse hot-path memcpy/state-copy operations and avoid temporary buffers.")
    if _as_float(inner.get("kernel_count")) >= 8:
        ideas.append("Prioritize launch fusion; the complete layer dispatches many device kernels per token.")
    if not bool(graph.get("capturable")):
        ideas.append("Make forward CUDA-graph capturable: precompile, preallocate, and keep module bindings static.")
    top = inner.get("top_operations", [])
    if isinstance(top, list) and top:
        ideas.append(f"Optimize or fuse the hottest device operation first: {top[0].get('name', 'unknown')}.")
    return ideas[:5]


def _preload_candidate_runtime(candidate_source: str) -> list[str]:
    """Initialize lazy native runtimes before any profiled CUDA work.

    DeepGEMM registers and initializes its JIT runtime at import time.  If its
    first import happens inside an NCU replay range, the first GEMM can observe
    an uninitialized library version.  Evaluation initializes the production
    runtime before timed work, so profiling must do the same.
    """
    loaded: list[str] = []
    for module_name in ("deep_gemm",):
        if module_name not in candidate_source:
            continue
        importlib.import_module(module_name)
        loaded.append(module_name)
    return loaded


def _prepare_model_for_timing(model: Any, inputs: list[Any]) -> bool:
    prepare = getattr(model, "prepare_for_timing", None)
    if not callable(prepare):
        return False
    prepare(*inputs)
    return True


def run_torch_profile(
    *,
    run_config: dict[str, Any],
    ref_arch_src: str,
    custom_model_src: str,
    out_dir: Path,
    target_only: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(run_config.get("backend", "triton"))
    device = torch.device(str(run_config.get("device", "cuda:0")))
    runtime_precision = resolve_runtime_precision_string(
        str(run_config.get("precision", "fp32")),
        str(run_config.get("runtime_precision", "") or ""),
    )
    precision = get_torch_dtype_from_string(runtime_precision)
    warmup_steps = max(1, int(run_config.get("profile_torch_warmup_steps", 2)))
    active_steps = max(1, int(run_config.get("profile_torch_active_steps", 3)))
    seed_num = 42

    problem_path = str(run_config.get("problem_path", "") or "")
    context: dict[str, Any] = {
        "__file__": problem_path or "<kernel-evo-profile-task>",
        "__name__": "kernel_evo_profile_reference",
    }
    temp_file = None
    gpu_lease = None
    previous_task_root = os.environ.get("KERNELEVO_TASK_ROOT")
    previous_sys_path = list(sys.path)
    try:
        preloaded_runtime_modules = _preload_candidate_runtime(custom_model_src)
        from kernel_evo.core.eval.gpu_guard import acquire_idle_gpu

        gpu_lease = acquire_idle_gpu(
            str(device),
            timeout=float(run_config.get("profile_gpu_idle_timeout", 120.0)),
            consecutive_idle_samples=int(run_config.get("profile_gpu_idle_samples", 3)),
            max_utilization=int(run_config.get("profile_gpu_max_utilization", 5)),
        )
        if problem_path:
            task_root = str(Path(problem_path).resolve().parent)
            os.environ["KERNELEVO_TASK_ROOT"] = task_root
            if task_root not in sys.path:
                sys.path.insert(0, task_root)
        torch.cuda.set_device(device)
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_arch_src, context)
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [_process_input_tensor(x, device, backend, precision) for x in init_inputs]

        if backend.lower() == "cuda_inline":
            from kernel_evo.core.code.cuda_backend_utils import apply_cuda_build_env

            apply_cuda_build_env(run_config)
        elif backend.lower() == "cute":
            from kernel_evo.core.code.cute_backend_utils import apply_cute_build_env

            apply_cute_build_env(run_config)
        ModelNew, temp_file = load_custom_model_with_tempfile(custom_model_src, entry_point="ModelNew")
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device=device, dtype=precision)
            prepare = getattr(custom_model, "prepare_for_evaluation", None)
            if callable(prepare):
                prepare()
        torch.cuda.synchronize(device=device)

        if target_only:
            configured_target_steps = run_config.get("profile_target_steps")
            steps = (
                max(1, int(configured_target_steps))
                if configured_target_steps is not None
                else warmup_steps + active_steps
            )
            target_inputs: list[list[Any]] = []
            target_warmup_steps = max(
                0, int(run_config.get("profile_target_warmup_steps", 0) or 0)
            )
            timing_prepared = False
            for step in range(target_warmup_steps):
                set_seed(seed_num - target_warmup_steps + step)
                warmup_inputs = get_inputs()
                warmup_inputs = [
                    _process_input_tensor(x, device, backend, precision)
                    for x in warmup_inputs
                ]
                if not timing_prepared:
                    timing_prepared = _prepare_model_for_timing(
                        custom_model, warmup_inputs
                    )
                with torch.no_grad():
                    custom_model(*warmup_inputs)
            torch.cuda.synchronize(device=device)
            for step in range(steps):
                set_seed(seed_num + step)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
                if not timing_prepared:
                    timing_prepared = _prepare_model_for_timing(custom_model, inputs)
                target_inputs.append(inputs)
            torch.cuda.synchronize(device=device)

            use_profiler_range = bool(run_config.get("profile_cuda_profiler_range", False))
            profiler_started = False
            try:
                if use_profiler_range:
                    torch.cuda.cudart().cudaProfilerStart()
                    profiler_started = True
                for inputs in target_inputs:
                    with torch.no_grad():
                        custom_model(*inputs)
                torch.cuda.synchronize(device=device)
            finally:
                if profiler_started:
                    torch.cuda.cudart().cudaProfilerStop()
            exclusive = gpu_lease.verify_exclusive()
            summary = {
                "status": "completed" if exclusive else "failed",
                "mode": "target_only",
                "steps": steps,
                "warmup_steps": target_warmup_steps,
                "cuda_profiler_range": use_profiler_range,
                "gpu_guard": gpu_lease.metadata,
                "preloaded_runtime_modules": preloaded_runtime_modules,
                "production_timing_prepared": timing_prepared,
            }
            if not exclusive:
                summary["reason"] = "GPU exclusivity was lost during profiling"
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        set_seed(seed_num)
        measurement_inputs = get_inputs()
        measurement_inputs = [
            _process_input_tensor(x, device, backend, precision) for x in measurement_inputs
        ]
        timing_prepared = _prepare_model_for_timing(custom_model, measurement_inputs)
        with torch.no_grad():
            for _ in range(warmup_steps):
                custom_model(*measurement_inputs)
        torch.cuda.synchronize(device=device)
        eager_us = _event_average_us(
            lambda: custom_model(*measurement_inputs), steps=active_steps, device=device
        )

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        prof_schedule = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1)

        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(warmup_steps + active_steps):
                with torch.no_grad():
                    custom_model(*measurement_inputs)
                torch.cuda.synchronize(device=device)
                prof.step()

        trace_path = out_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))

        events = sorted(
            (_event_to_dict(event) for event in prof.key_averages(group_by_input_shape=True)),
            key=lambda item: (item["self_cuda_time_total_us"], item["cuda_time_total_us"], item["cpu_time_total_us"]),
            reverse=True,
        )
        key_averages_path = out_dir / "key_averages.json"
        key_averages_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = _build_summary(events)
        inner = _trace_device_breakdown(trace_path, active_steps)
        evaluator_runtime_us = _as_float(run_config.get("profile_evaluator_runtime_us"))
        authoritative_eager_us = evaluator_runtime_us or eager_us
        eager = {
            "end_to_end_us": authoritative_eager_us,
            "profile_cuda_event_us": eager_us,
            "source": "evaluator" if evaluator_runtime_us > 0 else "profile_cuda_event",
            "inferred_dispatch_gaps_us": max(
                0.0,
                authoritative_eager_us - _as_float(inner.get("active_device_time_us")),
            ),
        }
        allocation_events = _as_float(inner.get("dynamic_allocation_events"))
        source_violations = _source_graph_violations(custom_model_src)
        attempt_graph_with_allocations = bool(
            run_config.get("profile_torch_attempt_graph_with_allocations", True)
        )
        graph = None
        if not attempt_graph_with_allocations:
            graph = _allocation_graph_failure(allocation_events, source_violations)
        if graph is None:
            graph = _graph_replay_objective(
                custom_model,
                measurement_inputs,
                steps=active_steps,
                device=device,
                source_violations=source_violations,
            )
        objectives = {
            "inner_kernel": inner,
            "eager_complete_layer": eager,
            "cuda_graph_complete_layer": graph,
        }
        summary.update(
            {
                "trace_file": str(trace_path),
                "key_averages_file": str(key_averages_path),
                "warmup_steps": warmup_steps,
                "active_steps": active_steps,
                "objectives": objectives,
                "graph_capturability_gate": {
                    "passed": bool(graph.get("capturable")),
                    "failure": graph.get("failure", ""),
                },
                "optimization_ideas": _optimization_ideas(objectives),
                "gpu_guard": gpu_lease.metadata,
                "preloaded_runtime_modules": preloaded_runtime_modules,
                "production_timing_prepared": timing_prepared,
            }
        )
        if not gpu_lease.verify_exclusive():
            summary["status"] = "failed"
            summary["reason"] = "GPU exclusivity was lost during profiling"
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    except torch.cuda.OutOfMemoryError as exc:
        return _write_oom_summary(out_dir, exc, device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return _write_oom_summary(out_dir, exc, device)
    finally:
        graceful_eval_cleanup(context, device, temp_file)
        if gpu_lease is not None:
            gpu_lease.close()
        if previous_task_root is None:
            os.environ.pop("KERNELEVO_TASK_ROOT", None)
        else:
            os.environ["KERNELEVO_TASK_ROOT"] = previous_task_root
        sys.path[:] = previous_sys_path


def _write_oom_summary(out_dir: Path, exc: BaseException, device: torch.device) -> dict[str, Any]:
    logger.warning(f"[torch_runner] GPU out of memory during profiling on {device}: {exc}")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    summary = {
        "status": "failed",
        "reason": "GPU out of memory",
        "error": str(exc),
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
