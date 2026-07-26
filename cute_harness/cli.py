from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any

from . import __version__
from .assembly import assemble_submission, candidate_starter
from .client import HarnessClient, RemoteHarnessError
from .policy import CheckReport, check_submission
from .tasks import REPO_ROOT, TaskError, TaskSpec, discover_tasks, load_task


DEFAULT_HARNESS_URL = "http://109.236.57.62:18080"
API_KEY_ENV = "CUTE_HARNESS_API_KEY"
URL_ENV = "CUTE_HARNESS_URL"


def _print_check(report: CheckReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(
        f"check={status} path={report.path} "
        f"cute_kernels={report.cute_kernel_count} "
        f"cute_jit={report.cute_jit_count}"
    )
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing output directory: {path}"
        )
    path.mkdir(parents=True)


def _resolve_submission(
    task: TaskSpec,
    submission_arg: str | None,
    use_baseline: bool,
) -> Path:
    if use_baseline:
        if submission_arg:
            raise RuntimeError(
                "pass either a submission path or --baseline, not both"
            )
        return task.baseline_path
    if not submission_arg:
        raise RuntimeError("submission path is required unless --baseline is used")
    return Path(submission_arg).resolve()


def _acceptance(task: TaskSpec, response: dict[str, Any]) -> dict[str, Any]:
    stdout = response.get("stdout")
    if not isinstance(stdout, str):
        stdout = ""
    pattern = task.validation["success_pattern"]
    pattern_matched = re.search(pattern, stdout) is not None
    server_success = (
        response.get("success") is True
        and response.get("exit_code") == 0
        and response.get("timed_out") is not True
    )
    return {
        "server_success": server_success,
        "success_pattern_matched": pattern_matched,
        "passed": server_success and pattern_matched,
    }


def _write_run_artifacts(
    output_dir: Path,
    task: TaskSpec,
    submission: Path,
    candidate: Path | None,
    harness_url: str,
    profiler: str,
    started_at: str,
    wall_seconds: float,
    response: dict[str, Any],
    acceptance: dict[str, Any],
    run_label: str | None,
) -> dict[str, Any]:
    _ensure_new_directory(output_dir)
    shutil.copy2(submission, output_dir / "submission.py")
    if candidate is not None and candidate.resolve() != submission.resolve():
        shutil.copy2(candidate, output_dir / "candidate.py")

    stdout = response.get("stdout")
    stderr = response.get("stderr")
    (output_dir / "stdout.txt").write_text(
        stdout if isinstance(stdout, str) else "",
        encoding="utf-8",
    )
    (output_dir / "stderr.txt").write_text(
        stderr if isinstance(stderr, str) else "",
        encoding="utf-8",
    )

    record = {
        "schema_version": 1,
        "harness_version": __version__,
        "task_id": task.id,
        "task_manifest_sha256": _sha256(task.directory / "task.json"),
        "task_prompt_sha256": _sha256(task.prompt_path),
        "starter_sha256": _sha256(task.starter_path),
        "submission_sha256": _sha256(submission),
        "candidate_sha256": (
            _sha256(candidate) if candidate is not None else None
        ),
        "run_label": run_label,
        "harness_url": harness_url,
        "profiler": profiler,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "acceptance": acceptance,
        "response": response,
    }
    (output_dir / "result.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def _run_one(
    task: TaskSpec,
    candidate: Path,
    submission: Path,
    candidate_mode: bool,
    output_dir: Path,
    harness_url: str,
    api_key: str,
    profiler: str,
    timeout_seconds: float,
    download_profile: bool,
    run_label: str | None,
) -> tuple[bool, dict[str, Any]]:
    report = check_submission(
        task,
        candidate,
        candidate_mode=candidate_mode,
    )
    _print_check(report)
    if not report.passed:
        raise RuntimeError("submission failed local policy check")

    print(f"run task={task.id} candidate={candidate}")
    client = HarnessClient(harness_url, api_key, timeout_seconds)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    response = client.run_file(submission, profiler)
    wall_seconds = time.monotonic() - start
    acceptance = _acceptance(task, response)
    record = _write_run_artifacts(
        output_dir,
        task,
        submission,
        candidate if candidate_mode else None,
        harness_url,
        profiler,
        started_at,
        wall_seconds,
        response,
        acceptance,
        run_label,
    )

    profile_id = response.get("profile_id")
    if download_profile and isinstance(profile_id, str) and profile_id:
        try:
            profile_bytes = client.download_profile(profile_id)
            (output_dir / "profile.json").write_bytes(profile_bytes)
            json.loads(profile_bytes.decode("utf-8"))
            record["profile_downloaded"] = True
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RemoteHarnessError,
        ) as error:
            record["profile_downloaded"] = False
            record["profile_download_error"] = str(error)
            print(f"warning: profile download failed: {error}")
        (output_dir / "result.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    stdout = response.get("stdout")
    stderr = response.get("stderr")
    if isinstance(stdout, str) and stdout:
        print("--- remote stdout ---")
        print(stdout.rstrip())
    if isinstance(stderr, str) and stderr:
        print("--- remote stderr ---")
        print(stderr.rstrip())

    status = "PASS" if acceptance["passed"] else "FAIL"
    print(
        f"result={status} task={task.id} "
        f"device_time_ms={response.get('device_time_ms')} "
        f"profile_id={response.get('profile_id')} "
        f"artifacts={output_dir}"
    )
    return bool(acceptance["passed"]), record


def command_list(_: argparse.Namespace) -> int:
    tasks = discover_tasks()
    width = max(len(task_id) for task_id in tasks)
    for task in tasks.values():
        print(f"{task.id:<{width}}  {task.title}")
    return 0


def command_show(args: argparse.Namespace) -> int:
    task = load_task(args.task_id)
    print(task.prompt_path.read_text(encoding="utf-8"))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    tasks = discover_tasks()
    failures = 0
    for task in tasks.values():
        report = check_submission(
            task,
            task.baseline_path,
            candidate_mode=False,
        )
        status = "PASS" if report.passed else "FAIL"
        print(f"baseline={status} task={task.id}")
        if not report.passed:
            failures += 1
            _print_check(report)

    key_present = bool(os.environ.get(API_KEY_ENV))
    harness_url = args.server or os.environ.get(URL_ENV, DEFAULT_HARNESS_URL)
    print(f"{API_KEY_ENV}={'set' if key_present else 'missing'}")
    print(f"harness_url={harness_url}")
    if args.require_key and not key_present:
        failures += 1
    return 1 if failures else 0


def command_prepare(args: argparse.Namespace) -> int:
    task = load_task(args.task_id)
    output_dir = Path(args.output).resolve()
    _ensure_new_directory(output_dir)
    shutil.copy2(task.prompt_path, output_dir / "TASK.md")
    (output_dir / "submission.py").write_text(
        candidate_starter(task),
        encoding="utf-8",
    )
    public_manifest = task.public_manifest()
    public_manifest["starter"] = "submission.py"
    (output_dir / "task.json").write_text(
        json.dumps(public_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"prepared task={task.id} output={output_dir}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    task = load_task(args.task_id)
    report = check_submission(task, Path(args.submission))
    _print_check(report)
    return 0 if report.passed else 1


def command_run(args: argparse.Namespace) -> int:
    task = load_task(args.task_id)
    submission = _resolve_submission(task, args.submission, args.baseline)
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set; keep the API key in the environment"
        )
    harness_url = args.server or os.environ.get(URL_ENV, DEFAULT_HARNESS_URL)
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT
        / "runs"
        / f"{_timestamp_slug()}_{task.id}"
    )
    if args.baseline:
        passed, _ = _run_one(
            task,
            submission,
            submission,
            False,
            output_dir,
            harness_url,
            api_key,
            args.profiler,
            args.timeout,
            not args.no_download_profile,
            args.label,
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="cute-harness-assembly-"
        ) as temp_dir:
            assembled = Path(temp_dir) / "submission.py"
            assembled.write_text(
                assemble_submission(task, submission),
                encoding="utf-8",
            )
            passed, _ = _run_one(
                task,
                submission,
                assembled,
                True,
                output_dir,
                harness_url,
                api_key,
                args.profiler,
                args.timeout,
                not args.no_download_profile,
                args.label,
            )
    return 0 if passed else 1


def command_run_all(args: argparse.Namespace) -> int:
    tasks = discover_tasks()
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set; keep the API key in the environment"
        )
    harness_url = args.server or os.environ.get(URL_ENV, DEFAULT_HARNESS_URL)
    suite_dir = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT / "runs" / f"{_timestamp_slug()}_baseline_suite"
    )
    _ensure_new_directory(suite_dir)

    summary: list[dict[str, Any]] = []
    all_passed = True
    for task in tasks.values():
        task_output = suite_dir / task.id
        try:
            passed, record = _run_one(
                task,
                task.baseline_path,
                task.baseline_path,
                False,
                task_output,
                harness_url,
                api_key,
                args.profiler,
                args.timeout,
                not args.no_download_profile,
                args.label,
            )
            summary.append(
                {
                    "task_id": task.id,
                    "passed": passed,
                    "response": record["response"],
                }
            )
            all_passed = all_passed and passed
        except (OSError, RuntimeError, RemoteHarnessError) as error:
            all_passed = False
            summary.append(
                {
                    "task_id": task.id,
                    "passed": False,
                    "error": str(error),
                }
            )
            print(f"error: task {task.id}: {error}", file=sys.stderr)

    (suite_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "all_passed": all_passed,
                "runs": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"suite={'PASS' if all_passed else 'FAIL'} artifacts={suite_dir}")
    return 0 if all_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cute_harness",
        description="Minimal runner for CuTe FP8/FP4 agent tasks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list task ids")
    list_parser.set_defaults(handler=command_list)

    show_parser = subparsers.add_parser("show", help="print the agent prompt")
    show_parser.add_argument("task_id")
    show_parser.set_defaults(handler=command_show)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="validate manifests/baselines and show environment status",
    )
    doctor_parser.add_argument("--server")
    doctor_parser.add_argument("--require-key", action="store_true")
    doctor_parser.set_defaults(handler=command_doctor)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="create an isolated TASK.md + submission.py workspace",
    )
    prepare_parser.add_argument("task_id")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(handler=command_prepare)

    check_parser = subparsers.add_parser(
        "check",
        help="run local AST/API compatibility checks",
    )
    check_parser.add_argument("task_id")
    check_parser.add_argument("submission")
    check_parser.set_defaults(handler=command_check)

    run_parser = subparsers.add_parser(
        "run",
        help="check and submit one file to the remote GPU harness",
    )
    run_parser.add_argument("task_id")
    run_parser.add_argument("submission", nargs="?")
    run_parser.add_argument("--baseline", action="store_true")
    run_parser.add_argument("--server")
    run_parser.add_argument("--profiler", default="pytorch")
    run_parser.add_argument("--timeout", type=float, default=360.0)
    run_parser.add_argument("--output")
    run_parser.add_argument(
        "--label",
        help="optional model/experiment label stored in result.json",
    )
    run_parser.add_argument("--no-download-profile", action="store_true")
    run_parser.set_defaults(handler=command_run)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="run every known-good baseline sequentially",
    )
    run_all_parser.add_argument("--server")
    run_all_parser.add_argument("--profiler", default="pytorch")
    run_all_parser.add_argument("--timeout", type=float, default=360.0)
    run_all_parser.add_argument("--output")
    run_all_parser.add_argument(
        "--label",
        help="optional model/experiment label stored in every result",
    )
    run_all_parser.add_argument("--no-download-profile", action="store_true")
    run_all_parser.set_defaults(handler=command_run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, TaskError, RemoteHarnessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
