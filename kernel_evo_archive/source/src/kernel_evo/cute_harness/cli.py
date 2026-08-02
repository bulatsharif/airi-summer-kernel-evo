"""Command-line interface for the standalone Python CuTe DSL laboratory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from kernel_evo.cute_harness.ablation import (
    DOCUMENTATION_TIERS,
    documentation_bundle,
    run_stability,
)
from kernel_evo.cute_harness.api import lookup_api
from kernel_evo.cute_harness.b300 import EvaluationConfig, discover_tasks, load_task
from kernel_evo.cute_harness.b300_policy import check_candidate
from kernel_evo.cute_harness.capabilities import probe_capabilities, resolve_target_arch
from kernel_evo.cute_harness.catalog import build_agent_context, search_catalog
from kernel_evo.cute_harness.codegen import inspect_and_verify_artifact, inspect_artifact
from kernel_evo.cute_harness.correctness import build_correctness_contract
from kernel_evo.cute_harness.experiments import query_experiments, record_experiment
from kernel_evo.cute_harness.feasibility import check_hopper_gemm_config
from kernel_evo.cute_harness.layout import probe_cute_layout, probe_layout
from kernel_evo.cute_harness.lint import lint_cute_source
from kernel_evo.cute_harness.paths import harness_root
from kernel_evo.cute_harness.runner import profile_command, run_command, sanitizer_command
from kernel_evo.cute_harness.task_spec import extract_task_spec


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cute", help="Inspect and operate the Python CuTe DSL laboratory.")
    actions = parser.add_subparsers(dest="cute_action")

    actions.add_parser("path", help="Print the self-contained harness directory.")
    actions.add_parser("task-list", help="List repository CuTe B300 tasks.")
    task_check = actions.add_parser("task-check", help="Check a CuTe B300 candidate.")
    task_check.add_argument("task")
    task_check.add_argument("candidate")
    ablation_context = actions.add_parser(
        "ablation-context", help="Render one frozen cumulative documentation level."
    )
    ablation_context.add_argument("task")
    ablation_context.add_argument("--tier", choices=DOCUMENTATION_TIERS, required=True)
    ablation_context.add_argument("--output", default="")

    stability = actions.add_parser(
        "stability", help="Repeat B300 evaluation and report timing dispersion."
    )
    stability.add_argument("task")
    stability.add_argument("--candidate", default="")
    stability.add_argument("--output", required=True)
    stability.add_argument("--runs", type=int, default=10)
    stability.add_argument("--seed", type=int, default=0)
    stability.add_argument("--warmup", type=int, default=5)
    stability.add_argument("--repeats", type=int, default=50)
    stability.add_argument("--timeout", type=float, default=900.0)
    stability.add_argument("--max-cv", type=float, default=0.02)
    doctor = actions.add_parser("doctor", help="Report exact DSL/GPU/tool capabilities.")
    _add_arch(doctor)
    doctor.add_argument("--device", type=int, default=0)
    doctor.add_argument("--write", default="", help="Optional capability_report.json destination.")

    lookup = actions.add_parser("lookup", help="Look up an exact installed Python CuTe DSL symbol.")
    lookup.add_argument("symbol")
    lookup.add_argument("--max-usages", type=int, default=3)

    search = actions.add_parser("search", help="Search versioned semantic cards and examples.")
    _add_retrieval(search)
    search.add_argument("--limit", type=int, default=8)

    context = actions.add_parser("context", help="Render the compact context used by an island author.")
    _add_retrieval(context)
    context.add_argument("--runtime-precision", default="")
    context.add_argument("--idea", default="")
    context.add_argument("--baseline", default="")
    context.add_argument("--cards", type=int, default=7)
    context.add_argument("--max-chars", type=int, default=10_000)
    context.add_argument("--deep-files", type=int, default=1)
    context.add_argument("--lessons", type=int, default=3)
    context.add_argument("--database", default="")

    layout = actions.add_parser("probe-layout", help="Print coordinate-to-index layout mappings.")
    layout.add_argument("--shape", required=True)
    layout.add_argument("--stride", default="")
    layout.add_argument("--order", default="")
    layout.add_argument("--coord", action="append", default=[])
    layout.add_argument("--tile", default="", help="Optional rank-2 logical-divide tile for --dsl.")
    layout.add_argument("--dsl", action="store_true", help="Compile the probe with the installed Python CuTe DSL.")
    layout.add_argument("--max-table-entries", type=int, default=32)

    inspect = actions.add_parser("inspect-codegen", help="Summarize PTX/SASS/MLIR/CUBIN instructions.")
    inspect.add_argument("artifact")
    inspect.add_argument("--expect", action="append", default=[])
    inspect.add_argument("--contract", default="", help="Expected-codegen YAML/manifest to enforce.")

    lint = actions.add_parser("lint", help="Check a candidate for common Python CuTe DSL mistakes.")
    lint.add_argument("source")
    lint.add_argument(
        "--contract",
        choices=("hopper_wgmma", "vector"),
        default="",
        help="Require source-level evidence for the configured codegen contract.",
    )
    _add_retrieval(lint)

    specification = actions.add_parser("spec", help="Extract bounded task-routing facts from source.")
    specification.add_argument("source")
    _add_retrieval(specification)
    specification.add_argument("--runtime-precision", default="")

    hopper = actions.add_parser(
        "check-hopper-config",
        help="Reject tile/cluster/stage proposals outside the verified Hopper GEMM envelope.",
    )
    hopper.add_argument("--tile", required=True, help="CTA M,N,K.")
    hopper.add_argument("--cluster", default="1,1", help="Cluster M,N.")
    hopper.add_argument("--stages", type=int, default=2)
    hopper.add_argument("--dtype", choices=["bf16", "fp16", "fp8", "fp8_e4m3fn", "fp8_e5m2"], default="bf16")
    hopper.add_argument("--output-dtype", choices=["bf16", "fp16", "fp8"], default="bf16")
    hopper.add_argument("--arch", default="sm_90a")
    hopper.add_argument("--smem-limit", type=int, default=232_448)

    correctness = actions.add_parser(
        "correctness-plan",
        help="Render the operation-aware evaluator matrix without running it.",
    )
    correctness.add_argument("--operation", default="elementwise")
    correctness.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"], default="bf16")
    correctness.add_argument("--shape", action="append", default=[])
    correctness.add_argument("--tile", default="")
    correctness.add_argument("--supports-strides", action="store_true")
    correctness.add_argument("--supports-misalignment", action="store_true")

    for name, help_text in (
        ("compile", "Run a compile probe with structured diagnostics."),
        ("check", "Run a correctness command that may print JSON metrics."),
        ("benchmark", "Run a benchmark command that may print JSON metrics."),
    ):
        command = actions.add_parser(name, help=help_text)
        _add_runner(command)

    sanitize = actions.add_parser("sanitize", help="Run Compute Sanitizer (memcheck first).")
    _add_runner(sanitize)
    sanitize.add_argument("--tool", choices=["memcheck", "racecheck", "initcheck", "synccheck"], default="memcheck")

    profile = actions.add_parser("profile", help="Capture a bounded Nsight Compute report.")
    _add_runner(profile)
    profile.add_argument("--set", dest="section_set", default="basic")

    record = actions.add_parser("record", help="Append one structured experiment record.")
    record.add_argument("--database", default=".kernelevo/cute-experiments.jsonl")
    record.add_argument("--record", required=True, help="JSON object file.")

    history = actions.add_parser("history", help="Query structured CuTe experiment memory.")
    history.add_argument("--database", default=".kernelevo/cute-experiments.jsonl")
    history.add_argument("--task", default="")
    history.add_argument("--tag", default="")
    history.add_argument("--decision", default="")
    history.add_argument("--limit", type=int, default=20)


def _add_arch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arch", default="", help="CuTe DSL target such as sm_90a; auto-detected by default.")
    parser.add_argument("--arch-list", default="", help="Optional Torch-style target such as 9.0.")


def _add_retrieval(parser: argparse.ArgumentParser) -> None:
    _add_arch(parser)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"], default="bf16")
    parser.add_argument("--operation", default="any")
    parser.add_argument("--concept", action="append", default=[])
    parser.add_argument("--query", default="")


def _add_runner(parser: argparse.ArgumentParser) -> None:
    _add_arch(parser)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--artifacts", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--contract", default="", help="Optional expected-codegen YAML/manifest gate.")
    parser.add_argument("tool_command", nargs=argparse.REMAINDER, help="Command after --.")


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _command(args: argparse.Namespace) -> list[str]:
    command = list(args.tool_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit(f"kernel-evo cute {args.cute_action}: provide a command after --")
    return command


def _arch(args: argparse.Namespace) -> str:
    arch = resolve_target_arch(explicit_arch=args.arch, arch_list=args.arch_list)
    if not arch:
        raise SystemExit("Could not detect a target GPU; pass --arch (Hopper: sm_90a).")
    return arch


def _artifacts(args: argparse.Namespace) -> Path:
    if args.artifacts:
        return Path(args.artifacts).expanduser().resolve()
    stamp = f"{int(time.time())}-{args.cute_action}"
    return (Path(args.cwd).expanduser().resolve() / ".kernelevo" / "cute-artifacts" / stamp)


def command(args: argparse.Namespace) -> None:
    action = getattr(args, "cute_action", None)
    if not action:
        raise SystemExit("kernel-evo cute: an action is required")
    if action == "path":
        print(harness_root())
        return
    if action == "task-list":
        _print({"tasks": list(discover_tasks())})
        return
    if action == "task-check":
        task = load_task(args.task)
        report = check_candidate(Path(args.candidate), task.policy)
        _print(
            {
                "passed": report.passed,
                "errors": report.errors,
                "observed_calls": sorted(report.observed_calls),
                "cute_kernels": report.cute_kernels,
                "cute_jit_functions": report.cute_jit_functions,
                "has_model_new": report.has_model_new,
            }
        )
        return
    if action == "ablation-context":
        bundle = documentation_bundle(load_task(args.task), args.tier)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(bundle.text, encoding="utf-8")
        _print(
            {
                "tier": bundle.tier,
                "tokens_cl100k": bundle.tokens_cl100k,
                "files": [str(path) for path in bundle.files],
                "output": (
                    str(Path(args.output).expanduser().resolve()) if args.output else ""
                ),
            }
        )
        return
    if action == "stability":
        if args.runs < 2:
            raise SystemExit("--runs must be at least 2")
        _print(
            run_stability(
                load_task(args.task),
                Path(args.output).expanduser(),
                runs=args.runs,
                config=EvaluationConfig(
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    timeout=args.timeout,
                ),
                candidate=(
                    Path(args.candidate).expanduser().resolve()
                    if args.candidate
                    else None
                ),
                max_cv=args.max_cv,
            )
        )
        return
    if action == "doctor":
        value = probe_capabilities(device=args.device, explicit_arch=args.arch, arch_list=args.arch_list)
        if args.write:
            destination = Path(args.write).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        _print(value)
        return
    if action == "lookup":
        _print(lookup_api(args.symbol, max_usages=args.max_usages))
        return
    if action == "search":
        arch = resolve_target_arch(explicit_arch=args.arch, arch_list=args.arch_list) or "sm_90a"
        _print(
            {
                "dialect": "cute_dsl_python",
                "entries": [
                    item.to_dict()
                    for item in search_catalog(
                        precision=args.precision,
                        arch=arch,
                        operation=args.operation,
                        concepts=args.concept,
                        query=args.query,
                        limit=args.limit,
                    )
                ],
            }
        )
        return
    if action == "context":
        arch = resolve_target_arch(explicit_arch=args.arch, arch_list=args.arch_list) or "sm_90a"
        bundle = build_agent_context(
            config={
                "backend": "cute",
                "precision": args.precision,
                "runtime_precision": args.runtime_precision,
                "cute_arch": arch,
                "cute_operation": args.operation,
                "cute_context_cards": args.cards,
                "cute_context_max_chars": args.max_chars,
                "cute_context_deep_files": args.deep_files,
                "cute_context_lessons": args.lessons,
            },
            idea={"id": "manual", "summary": args.idea or args.query},
            baseline_path=args.baseline or None,
            experiment_database=args.database or None,
        )
        print(bundle.text, end="")
        return
    if action == "probe-layout":
        if args.dsl:
            coordinates = [_ints(item) for item in args.coord]
            _print(
                probe_cute_layout(
                    _ints(args.shape),
                    stride=_ints(args.stride) if args.stride else None,
                    coordinate=coordinates[0] if coordinates else None,
                    tile=_ints(args.tile) if args.tile else None,
                    timeout=60.0,
                )
            )
            return
        _print(
            probe_layout(
                _ints(args.shape),
                stride=_ints(args.stride) if args.stride else None,
                order=_ints(args.order) if args.order else None,
                coordinates=[_ints(item) for item in args.coord],
                max_table_entries=args.max_table_entries,
            )
        )
        return
    if action == "inspect-codegen":
        if args.contract:
            _print(inspect_and_verify_artifact(args.artifact, args.contract))
        else:
            _print(inspect_artifact(args.artifact, expected=args.expect))
        return
    if action == "lint":
        source = Path(args.source).expanduser().resolve().read_text(encoding="utf-8", errors="replace")
        arch = resolve_target_arch(explicit_arch=args.arch, arch_list=args.arch_list) or "sm_90a"
        _print(
            lint_cute_source(
                source,
                precision=args.precision,
                arch=arch,
                operation=args.operation,
                codegen_contract=args.contract,
            )
        )
        return
    if action == "spec":
        source = Path(args.source).expanduser().resolve().read_text(encoding="utf-8", errors="replace")
        arch = resolve_target_arch(explicit_arch=args.arch, arch_list=args.arch_list) or "sm_90a"
        _print(
            extract_task_spec(
                source,
                operation="" if args.operation == "any" else args.operation,
                precision=args.precision,
                runtime_precision=args.runtime_precision,
                arch=arch,
            )
        )
        return
    if action == "check-hopper-config":
        _print(
            check_hopper_gemm_config(
                tile_shape_mnk=_ints(args.tile),
                cluster_shape_mn=_ints(args.cluster),
                stages=args.stages,
                dtype=args.dtype,
                output_dtype=args.output_dtype,
                arch=args.arch,
                shared_memory_limit_bytes=args.smem_limit,
            )
        )
        return
    if action == "correctness-plan":
        _print(
            build_correctness_contract(
                operation=args.operation,
                precision=args.precision,
                representative_shapes=[_ints(value) for value in args.shape],
                tile_shape=_ints(args.tile) if args.tile else (),
                supports_strides=args.supports_strides,
                supports_misalignment=args.supports_misalignment,
            )
        )
        return
    if action in {"compile", "check", "benchmark"}:
        _print(
            run_command(
                _command(args),
                kind=action,
                arch=_arch(args),
                cwd=args.cwd,
                artifact_dir=_artifacts(args),
                timeout=args.timeout,
                debug=args.debug,
                expectations=args.contract or None,
            )
        )
        return
    if action == "sanitize":
        _print(
            sanitizer_command(
                _command(args),
                tool=args.tool,
                arch=_arch(args),
                cwd=args.cwd,
                artifact_dir=_artifacts(args),
                timeout=args.timeout,
                expectations=args.contract or None,
            )
        )
        return
    if action == "profile":
        _print(
            profile_command(
                _command(args),
                arch=_arch(args),
                cwd=args.cwd,
                artifact_dir=_artifacts(args),
                section_set=args.section_set,
                timeout=args.timeout,
                expectations=args.contract or None,
            )
        )
        return
    if action == "record":
        payload = json.loads(Path(args.record).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("Experiment record must be a JSON object")
        _print(record_experiment(args.database, payload))
        return
    if action == "history":
        _print(
            query_experiments(
                args.database,
                task=args.task,
                tag=args.tag,
                decision=args.decision,
                limit=args.limit,
            )
        )
        return
    raise SystemExit(f"Unknown kernel-evo cute action: {action}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Python CuTe DSL laboratory")
    subparsers = parser.add_subparsers(dest="command")
    setup_parser(subparsers)
    parsed = parser.parse_args(["cute", *(sys.argv[1:] if argv is None else argv)])
    command(parsed)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
