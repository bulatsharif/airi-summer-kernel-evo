from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cute_harness.tasks import TaskError, discover_tasks

from .report import render_table
from .runner import ExperimentConfig, default_output_dir, run_experiment


REQUIRED_ENVIRONMENT = (
    "QWEN_BASE_URL",
    "QWEN_API_KEY",
    "CUTE_HARNESS_API_KEY",
)


def _environment_status(
    check_endpoint: bool,
    expected_model: str | None = None,
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    passed = True
    for name in REQUIRED_ENVIRONMENT:
        present = bool(os.environ.get(name))
        messages.append(f"{name}={'set' if present else 'missing'}")
        passed = passed and present
    opencode = shutil.which("opencode")
    messages.append(f"opencode={opencode or 'missing'}")
    passed = passed and opencode is not None

    if check_endpoint and all(os.environ.get(name) for name in REQUIRED_ENVIRONMENT[:2]):
        base_url = os.environ["QWEN_BASE_URL"].rstrip("/")
        request = Request(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {os.environ['QWEN_API_KEY']}",
                "User-Agent": "cute-experiment-doctor/0.1",
            },
        )
        try:
            with urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_count = (
                len(payload.get("data", []))
                if isinstance(payload, dict)
                and isinstance(payload.get("data"), list)
                else 0
            )
            messages.append(
                f"model_endpoint=reachable models={model_count}"
            )
            if expected_model and isinstance(payload, dict):
                data = payload.get("data")
                model_ids = {
                    item.get("id")
                    for item in data
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                } if isinstance(data, list) else set()
                server_model = expected_model.split("/", 1)[-1]
                if model_ids and server_model not in model_ids:
                    passed = False
                    messages.append(
                        f"requested_model={server_model} not advertised"
                    )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            passed = False
            messages.append(f"model_endpoint=unreachable ({error})")
    return passed, messages


def command_doctor(args: argparse.Namespace) -> int:
    passed, messages = _environment_status(
        not args.no_endpoint_check,
        args.model,
    )
    tasks = discover_tasks()
    messages.append(f"tasks={len(tasks)}")
    for message in messages:
        print(message)
    return 0 if passed else 1


def command_run(args: argparse.Namespace) -> int:
    environment_ok, messages = _environment_status(True, args.model)
    if not environment_ok:
        for message in messages:
            print(message, file=sys.stderr)
        raise RuntimeError(
            "experiment environment is not ready; start the SSH tunnel and "
            "export QWEN_BASE_URL, QWEN_API_KEY, and CUTE_HARNESS_API_KEY"
        )
    tasks = discover_tasks()
    task_ids = (
        tuple(tasks)
        if args.all
        else tuple(dict.fromkeys(args.task))
    )
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else default_output_dir()
    )
    config = ExperimentConfig(
        model=args.model,
        task_ids=task_ids,
        attempts=args.attempts,
        agent_timeout=args.agent_timeout,
        gpu_timeout=args.gpu_timeout,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=output_dir,
    )
    all_passed, rows = run_experiment(config)
    print(render_table(rows))
    print()
    print(f"experiment={'PASS' if all_passed else 'FAIL'} artifacts={output_dir}")
    return 0 if all_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment",
        description="Run OpenCode agents on CuTe tasks and compare to baselines",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="check task discovery, secrets, OpenCode, and the model endpoint",
    )
    doctor.add_argument("--no-endpoint-check", action="store_true")
    doctor.add_argument("--model")
    doctor.set_defaults(handler=command_doctor)

    run = subparsers.add_parser(
        "run",
        help="run one or more agent evaluation tasks",
    )
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append")
    selection.add_argument("--all", action="store_true")
    run.add_argument("--model", required=True)
    run.add_argument("--attempts", type=int, default=1)
    run.add_argument("--agent-timeout", type=float, default=600.0)
    run.add_argument("--gpu-timeout", type=float, default=600.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warmup", type=int, default=2)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--output")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, TaskError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
