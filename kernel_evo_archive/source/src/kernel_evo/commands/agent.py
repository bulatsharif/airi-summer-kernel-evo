"""Nested CLI commands for the direct-agent evolution harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kernel_evo.agent import (
    AgentRunConfig,
    InvalidTransitionError,
    KernelEvoAgent,
    KernelEvoAgentError,
)
from kernel_evo.core.precision import VALID_PRECISIONS, VALID_RUNTIME_PRECISIONS
from kernel_evo.cute_harness.ablation import DOCUMENTATION_TIERS


def setup_parsers(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser("run", help="Initialize or inspect an agent-authored KernelEvo run.")
    run_subparsers = run_parser.add_subparsers(dest="agent_action")
    _setup_run_init(run_subparsers)
    _setup_run_status(run_subparsers)
    _setup_run_extend(run_subparsers)

    iter_parser = subparsers.add_parser("iter", help="Prepare, evaluate, report, or advance a barrier iteration.")
    iter_subparsers = iter_parser.add_subparsers(dest="agent_action")
    _setup_iter_prepare(iter_subparsers)
    _setup_iter_evaluate(iter_subparsers)
    _setup_iter_report(iter_subparsers)
    _setup_iter_review_profiles(iter_subparsers)
    _setup_iter_advance(iter_subparsers)

    island_parser = subparsers.add_parser("island", help="Inspect or submit one bounded island candidate.")
    island_subparsers = island_parser.add_subparsers(dest="agent_action")
    _setup_island_context(island_subparsers)
    _setup_island_retarget(island_subparsers)
    _setup_island_submit(island_subparsers)
    _setup_island_repair(island_subparsers)
    _setup_island_review_submit(island_subparsers)


def agent_command(args: argparse.Namespace) -> None:
    action = getattr(args, "agent_action", None)
    if not action:
        parent = getattr(args, "command", "run")
        raise SystemExit(f"kernel-evo {parent}: an action is required")
    try:
        controller = KernelEvoAgent(args.runs_dir)
        if args.command == "run" and action == "init":
            _run_init(controller, args)
        elif args.command == "run" and action == "status":
            _print_json(controller.status(args.run_id))
        elif args.command == "run" and action == "extend":
            ideas = None
            if args.idea:
                ideas = [
                    {
                        "id": f"manual-steer-{index + 1}",
                        "summary": summary,
                        "source": "manual_steering",
                    }
                    for index, summary in enumerate(args.idea)
                ]
            _print_json(
                controller.extend_run(
                    args.run_id,
                    args.additional_steps,
                    ideas=ideas,
                    author_readable_files=args.author_readable_files,
                )
            )
        elif args.command == "iter" and action == "prepare":
            tasks = controller.prepare_iteration(
                args.run_id,
                args.iteration,
                documentation_enabled=False if args.disable_documentation else None,
                documentation_tier=None
                if args.disable_documentation
                else args.documentation_tier,
            )
            _print_json(
                {
                    "run_id": args.run_id,
                    "iteration": tasks[0].iteration if tasks else args.iteration,
                    "tasks": [task.to_dict() for task in tasks],
                    "coordination": "Spawn one bounded kernel-author per task_file; wait for all before evaluate.",
                }
            )
        elif args.command == "iter" and action == "evaluate":
            _print_json(_evaluate_to_barrier(controller, args.run_id, args.iteration))
        elif args.command == "iter" and action == "report":
            report = controller.report_iteration(args.run_id, args.iteration, format=args.format)
            _print_json(report) if isinstance(report, dict) else print(report, end="")
        elif args.command == "iter" and action == "review-profiles":
            tasks = controller.prepare_profile_reviews(args.run_id, args.iteration)
            _print_json(
                {
                    "run_id": args.run_id,
                    "tasks": tasks,
                    "coordination": (
                        "Spawn one bounded kernel-profile-reviewer per task_file; each reviewer "
                        "writes only output_file, then submit with island review-submit."
                    ),
                }
            )
        elif args.command == "iter" and action == "advance":
            _print_json(controller.advance_iteration(args.run_id))
        elif args.command == "island" and action == "context":
            context = controller.island_context(args.run_id, args.iteration, args.island)
            if args.format == "markdown":
                print(Path(context["task_file"]).read_text(encoding="utf-8"), end="")
            else:
                _print_json(context)
        elif args.command == "island" and action == "retarget":
            _print_json(
                controller.retarget_island(
                    args.run_id,
                    args.iteration,
                    args.island,
                    {
                        "id": args.idea_id,
                        "summary": args.idea,
                        "source": "manual_steering",
                    },
                )
            )
        elif args.command == "island" and action == "submit":
            metadata = _submission_metadata(args)
            _print_json(
                controller.submit_candidate(
                    args.run_id,
                    args.iteration,
                    args.island,
                    args.candidate,
                    metadata=metadata,
                )
            )
        elif args.command == "island" and action == "repair":
            _print_json(
                controller.reopen_island_for_repair(
                    args.run_id,
                    args.iteration,
                    args.island,
                )
            )
        elif args.command == "island" and action == "review-submit":
            _print_json(
                controller.submit_profile_review(
                    args.run_id,
                    args.iteration,
                    args.island,
                    args.review,
                )
            )
        else:
            raise SystemExit(f"Unknown agent command: {args.command} {action}")
    except KernelEvoAgentError as exc:
        raise SystemExit(f"kernel-evo {args.command} {action}: {exc}") from exc


def _evaluate_to_barrier(
    controller: KernelEvoAgent, run_id: str, iteration: int | None
) -> dict[str, Any]:
    """Keep one CLI invocation alive until the evaluation barrier is complete."""
    while True:
        report = controller.evaluate_iteration(run_id, iteration)
        phase = str(controller.status(run_id).get("phase", ""))
        if phase in {"evaluated", "complete"}:
            return report
        if phase != "evaluating":
            raise InvalidTransitionError(
                f"Evaluation returned before the barrier in unexpected phase {phase!r}"
            )


def _setup_run_init(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("init", help="Create persistent run state and the initial barrier.")
    _add_runs_dir(parser)
    parser.add_argument("--config", default="", help="YAML/JSON AgentRunConfig file.")
    parser.add_argument("--run-id", default=None, help="Stable run id; generated when omitted.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--problem-path", default=None, help="KernelBench-format task.py or its directory.")
    parser.add_argument("--baseline", default=None, help="Initial candidate source file or unambiguous directory.")
    parser.add_argument("--tests", default=None, help="Tests path summarized into each authoring packet.")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--problem-id", type=int, default=None)
    parser.add_argument("--dataset-src", default=None, choices=["huggingface", "local"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--backend", default=None, choices=["triton", "cuda_inline", "cute"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--islands", type=int, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--seed-preflight", action="store_true", default=None)
    parser.add_argument("--max-repairs-per-island", type=int, default=None)
    parser.add_argument(
        "--author-readable-file",
        dest="author_readable_files",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--disable-documentation",
        dest="documentation_enabled",
        action="store_false",
        default=None,
        help="Omit optional references, skills, and retrieved backend guidance.",
    )
    parser.add_argument(
        "--documentation-tier",
        choices=DOCUMENTATION_TIERS,
        default=None,
        help="Cumulative CuTe context level.",
    )
    parser.add_argument(
        "--b300-seed",
        choices=["baseline", "starter"],
        default=None,
        help="Start from the verified baseline or incomplete public starter.",
    )
    parser.add_argument("--precision", default=None, choices=list(VALID_PRECISIONS))
    parser.add_argument("--runtime-precision", default=None, choices=list(VALID_RUNTIME_PRECISIONS))
    parser.add_argument(
        "--measurement-mode",
        default=None,
        choices=["wall-clock", "device-time"],
    )
    parser.add_argument("--timing-method", default=None)
    parser.add_argument("--num-correct-trials", type=int, default=None)
    parser.add_argument("--num-perf-trials", type=int, default=None)
    parser.add_argument("--output-rtol", type=float, default=None)
    parser.add_argument("--output-atol", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch-list", default=None)
    parser.add_argument("--cute-arch", default=None, help="Python CuTe DSL target, e.g. sm_90a.")
    parser.add_argument("--no-cute-harness", dest="cute_harness_enabled", action="store_false", default=None)
    parser.add_argument("--cute-context-cards", type=int, default=None)
    parser.add_argument("--cute-context-max-chars", type=int, default=None)
    parser.add_argument("--cute-keep-ir", action="store_true", default=None)
    parser.add_argument("--cute-optimization-warnings", action="store_true", default=None)
    parser.add_argument(
        "--execution-mode",
        default=None,
        choices=["local_execution", "remote_execution"],
    )
    parser.add_argument("--remote-validator-url", default=None)
    parser.add_argument("--remote-poll-interval", type=float, default=None)
    parser.add_argument(
        "--tracker",
        default=None,
        help="Optional HTTP event sink; local tracker.jsonl is always written.",
    )
    parser.add_argument(
        "--evaluator-command",
        default=None,
        help="External harness command; supports {candidate}, {baseline}, {iteration}, and {island} placeholders.",
    )
    parser.add_argument("--evaluator-kind", default=None, choices=["kernelbench", "cute_b300"])
    parser.add_argument("--evaluator-timeout", type=float, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--evaluation-warmup", type=int, default=None)
    parser.add_argument("--evaluation-repeats", type=int, default=None)
    parser.add_argument("--harness-url", default=None)
    parser.add_argument("--profile", dest="profile_enabled", action="store_true", default=None)
    parser.add_argument(
        "--profile-timeline",
        action="store_true",
        default=None,
        help="Use the complete B300 GPU timeline packet; requires --profile or profiling.enabled.",
    )
    parser.add_argument("--profile-runners", default=None, help="Comma-separated torch,nsys,ncu.")
    parser.add_argument("--profile-min-speedup", type=float, default=None)
    parser.add_argument("--migration-interval", type=int, default=None)
    parser.add_argument("--idea", dest="ideas", action="append", default=None)
    parser.add_argument("--rules-file", default=None)


def _setup_run_status(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("status", help="Return compact run state and the next legal action.")
    _add_run_identity(parser)


def _setup_run_extend(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extend", help="Add future barriers while preserving the current archive."
    )
    _add_run_identity(parser)
    parser.add_argument("--additional-steps", type=int, required=True)
    parser.add_argument(
        "--idea",
        action="append",
        default=None,
        help="Override configured ideas for the added barriers while preserving the archive.",
    )
    parser.add_argument(
        "--author-readable-file",
        dest="author_readable_files",
        action="append",
        default=None,
        help="Add a read-only reference to author packets created after the extension.",
    )


def _setup_iter_prepare(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare", help="Create one tiny isolated authoring packet per island.")
    _add_run_identity(parser)
    _add_iteration(parser, required=False)
    parser.add_argument(
        "--disable-documentation",
        action="store_true",
        help="Omit optional references, skills, and retrieved backend guidance.",
    )
    parser.add_argument(
        "--documentation-tier",
        choices=DOCUMENTATION_TIERS,
        default=None,
        help="Cumulative CuTe context level.",
    )


def _setup_iter_evaluate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate every island and update the archive at the barrier.")
    _add_run_identity(parser)
    _add_iteration(parser, required=False)


def _setup_iter_report(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Render a deterministic compact iteration report.")
    _add_run_identity(parser)
    _add_iteration(parser, required=False)
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"])


def _setup_iter_advance(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("advance", help="Open the next barrier, or complete the run.")
    _add_run_identity(parser)


def _setup_iter_review_profiles(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "review-profiles",
        help="Create bounded compact-trace analysis tasks for profiled candidates.",
    )
    _add_run_identity(parser)
    _add_iteration(parser, required=False)


def _setup_island_context(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("context", help="Return exactly one island's authoring contract.")
    _add_run_identity(parser)
    _add_iteration(parser, required=True)
    parser.add_argument("--island", type=int, required=True)
    parser.add_argument("--format", default="json", choices=["json", "markdown"])


def _setup_island_retarget(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "retarget", help="Replace the idea on an unopened authoring packet."
    )
    _add_run_identity(parser)
    _add_iteration(parser, required=True)
    parser.add_argument("--island", type=int, required=True)
    parser.add_argument("--idea-id", default="manual-retarget")
    parser.add_argument("--idea", required=True)


def _setup_island_submit(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("submit", help="Copy/mark one authored candidate and its compact rationale.")
    _add_run_identity(parser)
    _add_iteration(parser, required=True)
    parser.add_argument("--island", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metadata", default="", help="Optional JSON object file returned by the author.")
    parser.add_argument("--idea-summary", default="")
    parser.add_argument("--expected-perf-mechanism", default="")
    parser.add_argument("--risk", default="")


def _setup_island_repair(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "repair",
        help="Open the one bounded repair turn allowed by an evaluated localized failure.",
    )
    _add_run_identity(parser)
    _add_iteration(parser, required=True)
    parser.add_argument("--island", type=int, required=True)


def _setup_island_review_submit(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "review-submit", help="Store one bounded compact-profile review and its new ideas."
    )
    _add_run_identity(parser)
    _add_iteration(parser, required=True)
    parser.add_argument("--island", type=int, required=True)
    parser.add_argument("--review", required=True, help="PROFILE_REVIEW.json written by reviewer.")


def _add_runs_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-dir", default=".kernelevo/runs")


def _add_run_identity(parser: argparse.ArgumentParser) -> None:
    _add_runs_dir(parser)
    parser.add_argument("--run-id", required=True)


def _add_iteration(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--iter", dest="iteration", type=int, required=required, default=None)


def _run_init(controller: KernelEvoAgent, args: argparse.Namespace) -> None:
    keys = (
        "name",
        "problem_path",
        "baseline",
        "tests",
        "level",
        "problem_id",
        "dataset_src",
        "dataset_name",
        "backend",
        "steps",
        "islands",
        "candidate_name",
        "seed_preflight",
        "max_repairs_per_island",
        "author_readable_files",
        "documentation_enabled",
        "documentation_tier",
        "b300_seed",
        "precision",
        "runtime_precision",
        "measurement_mode",
        "timing_method",
        "num_correct_trials",
        "num_perf_trials",
        "output_rtol",
        "output_atol",
        "device",
        "arch_list",
        "cute_arch",
        "cute_harness_enabled",
        "cute_context_cards",
        "cute_context_max_chars",
        "cute_keep_ir",
        "cute_optimization_warnings",
        "execution_mode",
        "remote_validator_url",
        "remote_poll_interval",
        "tracker",
        "evaluator_kind",
        "evaluator_command",
        "evaluator_timeout",
        "evaluation_seed",
        "evaluation_warmup",
        "evaluation_repeats",
        "harness_url",
        "profile_enabled",
        "profile_timeline",
        "profile_runners",
        "profile_min_speedup",
        "migration_interval",
        "ideas",
        "rules_file",
    )
    overrides = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    if args.config:
        config: AgentRunConfig | dict[str, Any] | str = args.config
    else:
        config = overrides
        overrides = {}
    _print_json(controller.init_run(config, run_id=args.run_id, overrides=overrides))


def _submission_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if args.metadata:
        path = Path(args.metadata).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SystemExit(f"Submission metadata must be a JSON object: {path}")
        metadata.update(value)
    for key in ("idea_summary", "expected_perf_mechanism", "risk"):
        value = getattr(args, key)
        if value:
            metadata[key] = value
    metadata["needs_evaluation"] = True
    return metadata


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
