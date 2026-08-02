"""Repository CuTe task loading and remote B300 evaluation."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from kernel_evo.cute_harness.b300_policy import check_candidate
from kernel_evo.cute_harness.trace_summary import (
    compact_timeline_metadata,
    summarize_chrome_trace,
    summarize_chrome_timeline,
    trace_summary_markdown,
    trace_timeline_markdown,
)
from kernel_evo.resources.paths import get_repo_root


# Port 18081 serves the same three routes as 18080 but returns a run in ~2.1s
# against ~24.1s, measured on the verified level2_12 reference. It reports the
# same kernel_time_ms in stdout (0.3828-0.3869 across both, well inside the
# 0.21% run-to-run dispersion) and still returns a profile_id, so the speedup
# denominator and the profiler feedback in E4 are both unaffected. Override with
# CUTE_HARNESS_URL.
DEFAULT_HARNESS_URL = "http://109.236.57.62:18081"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STDERR_TAIL_LINES = 25
EVALUATOR_MARKER = "# === CUTE_HARNESS_EVALUATOR_V1 ==="
RUNTIME_SENTINEL_US = 1_000_000_000.0


@dataclass(frozen=True)
class TaskSpec:
    directory: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.data["policy"])

    @property
    def validation(self) -> dict[str, Any]:
        return dict(self.data["validation"])

    @property
    def entrypoint(self) -> str:
        return str(self.policy["entrypoint_jit"])

    def path(self, field: str) -> Path:
        value = self.data.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{self.id}: {field} must be a path string")
        path = (self.directory / value).resolve()
        if not path.is_relative_to(self.directory.parent.parent):
            raise ValueError(f"{self.id}: {field} escapes the CuTe task suite")
        return path

    @property
    def prompt_path(self) -> Path:
        return self.path("prompt")

    @property
    def starter_path(self) -> Path:
        return self.path("starter")

    @property
    def baseline_path(self) -> Path:
        return self.path("baseline")

    def paths(self, field: str) -> tuple[Path, ...]:
        values = self.data.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{self.id}: {field} must be a path list")
        paths = tuple((self.directory / str(value)).resolve() for value in values)
        if any(not path.is_relative_to(self.directory.parent.parent) for path in paths):
            raise ValueError(f"{self.id}: {field} escapes the CuTe task suite")
        return paths

    @property
    def skill_paths(self) -> tuple[Path, ...]:
        return self.paths("agent_skills")


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 0
    warmup: int = 2
    repeats: int = 5
    timeout: float = 600.0
    profile_timeline: bool = False


def tasks_root() -> Path:
    return get_repo_root() / "tasks" / "cute" / "tasks"


def discover_tasks() -> dict[str, TaskSpec]:
    return {
        manifest.parent.name: _load_manifest(manifest)
        for manifest in sorted(tasks_root().glob("*/task.json"))
    }


def load_task(value: str | Path) -> TaskSpec:
    path = Path(value).expanduser()
    manifest = path / "task.json" if path.is_dir() else path
    if manifest.is_file():
        return _load_manifest(manifest.resolve())
    tasks = discover_tasks()
    if str(value) not in tasks:
        raise ValueError(f"unknown CuTe task {value!s}; known tasks: {', '.join(tasks)}")
    return tasks[str(value)]


def _load_manifest(path: Path) -> TaskSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("id") != path.parent.name:
        raise ValueError(f"invalid CuTe task manifest: {path}")
    task = TaskSpec(path.parent, data)
    if not task.entrypoint:
        raise ValueError(f"{task.id}: policy.entrypoint_jit is required")
    for member in (task.prompt_path, task.starter_path, task.baseline_path):
        if not member.is_file():
            raise ValueError(f"{task.id}: missing task file: {member}")
    for skill in task.skill_paths:
        if not (skill / "SKILL.md").is_file():
            raise ValueError(f"{task.id}: invalid skill directory: {skill}")
    return task


def _split(source: str, name: str) -> tuple[str, str]:
    candidate, marker, evaluator = source.partition(EVALUATOR_MARKER)
    if not marker or EVALUATOR_MARKER in evaluator:
        raise ValueError(f"{name}: expected exactly one evaluator marker")
    return candidate.rstrip() + "\n", evaluator.lstrip()


def _with_model_interface(source: str, task: TaskSpec) -> str:
    if re.search(r"(?m)^class ModelNew\b", source):
        return source.rstrip() + "\n"
    return (
        source.rstrip()
        + "\n\n\nclass ModelNew:\n"
        + f"    forward = staticmethod({task.entrypoint})\n"
    )


def baseline_candidate(task: TaskSpec) -> str:
    candidate, _ = _split(task.baseline_path.read_text(encoding="utf-8"), str(task.baseline_path))
    return _with_model_interface(candidate, task)


def starter_candidate(task: TaskSpec) -> str:
    candidate, _ = _split(task.starter_path.read_text(encoding="utf-8"), str(task.starter_path))
    return _with_model_interface(candidate, task)


def evolution_task_description(
    task: TaskSpec,
    *,
    documentation_enabled: bool = True,
    documentation_tier: str = "",
) -> str:
    from kernel_evo.cute_harness.ablation import documentation_bundle

    tier = documentation_tier or ("errors" if documentation_enabled else "bare")
    return (
        documentation_bundle(task, tier).text
        + "\nPreserve the common `ModelNew.forward` CuTe JIT interface. "
        "Edit the supplied seed rather than replacing the task interface. "
        "KernelEvo owns inputs, correctness, timing, and execution.\n"
    )


def assemble_submission(task: TaskSpec, candidate: Path, config: EvaluationConfig) -> str:
    source = candidate.read_text(encoding="utf-8")
    if EVALUATOR_MARKER in source:
        raise ValueError("candidate contains the reserved evaluator marker")
    _, evaluator = _split(task.starter_path.read_text(encoding="utf-8"), str(task.starter_path))
    return (
        source.rstrip()
        + "\n\n"
        + EVALUATOR_MARKER
        + "\n"
        + f"_CUTE_HARNESS_SEED = {config.seed}\n"
        + f"_CUTE_HARNESS_WARMUP = {config.warmup}\n"
        + f"_CUTE_HARNESS_REPEATS = {config.repeats}\n\n"
        + evaluator
    )


def evaluate(
    task: TaskSpec,
    candidate: Path,
    output_dir: Path,
    config: EvaluationConfig,
    *,
    harness_url: str = "",
) -> dict[str, Any]:
    report = check_candidate(candidate, task.policy)
    if not report.passed:
        raise ValueError("candidate policy failed: " + "; ".join(report.errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_snapshot = output_dir / "candidate.py"
    submission = output_dir / "submission.py"
    shutil.copy2(candidate, candidate_snapshot)
    submission.write_text(assemble_submission(task, candidate_snapshot, config), encoding="utf-8")

    client = HarnessClient(
        harness_url or os.environ.get("CUTE_HARNESS_URL", DEFAULT_HARNESS_URL),
        os.environ.get("CUTE_HARNESS_API_KEY", ""),
        config.timeout,
    )
    lock_path = _b300_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        response = client.run_file(submission, "pytorch")
        wall_seconds = time.monotonic() - started
    stdout = response.get("stdout") if isinstance(response.get("stdout"), str) else ""
    stderr = response.get("stderr") if isinstance(response.get("stderr"), str) else ""
    pattern_matched = re.search(str(task.validation["success_pattern"]), stdout) is not None
    kernel_time_ms = _kernel_time_ms(stdout)
    passed = (
        response.get("success") is True
        and response.get("exit_code") == 0
        and response.get("timed_out") is not True
        and pattern_matched
        and kernel_time_ms is not None
    )
    record = {
        "task_id": task.id,
        "candidate_sha256": _sha256(candidate_snapshot),
        "submission_sha256": _sha256(submission),
        "passed": passed,
        "success_pattern_matched": pattern_matched,
        "kernel_time_ms": kernel_time_ms,
        "wall_seconds": wall_seconds,
        "response": response,
    }
    (output_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    profile_id = response.get("profile_id")
    profile_path = output_dir / "profile.json"
    summary_path = output_dir / "profile_summary.md"
    # A failed run has no profile; never leave the previous attempt's summary behind.
    summary_path.unlink(missing_ok=True)
    if isinstance(profile_id, str) and profile_id:
        try:
            profile_path.write_bytes(client.download_profile(profile_id))
        except (OSError, RemoteHarnessError) as error:
            record["profile_download_error"] = str(error)
        else:
            try:
                trace = json.loads(profile_path.read_text(encoding="utf-8"))
                summary = summarize_chrome_trace(trace)
                if config.profile_timeline:
                    timeline = summarize_chrome_timeline(
                        trace,
                        candidate_kernel_symbols=_candidate_kernel_symbols(candidate_snapshot),
                    )
                    summary_text = trace_timeline_markdown(timeline, aggregate=summary)
                    record["profile_timeline"] = compact_timeline_metadata(timeline)
                else:
                    summary_text = trace_summary_markdown(summary)
                summary_path.write_text(summary_text, encoding="utf-8")
            except (OSError, ValueError) as error:
                # Author feedback is best-effort; it never fails a timed evaluation.
                record["profile_summary_error"] = f"{type(error).__name__}: {error}"
            else:
                record["profile_summary"] = summary
    (output_dir / "result.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def metrics(record: dict[str, Any], reference_ms: float) -> dict[str, Any]:
    passed = bool(record["passed"])
    runtime_ms = record.get("kernel_time_ms")
    runtime_us = float(runtime_ms) * 1000.0 if passed and isinstance(runtime_ms, (int, float)) else RUNTIME_SENTINEL_US
    speedup = reference_ms * 1000.0 / runtime_us if runtime_us < RUNTIME_SENTINEL_US else 0.0
    value = {
        "compiled": float(passed or reached_validation(record)),
        "correctness": float(passed),
        "is_valid": float(passed),
        "runtime_us": runtime_us,
        "ref_runtime_us": reference_ms * 1000.0,
        "speedup": speedup,
        "fitness": speedup,
    }
    if not passed:
        value["error"] = failure_reason(record)
    return value


def reached_validation(record: dict[str, Any]) -> bool:
    """Say whether the candidate compiled, launched, and produced numbers.

    These three metrics used to be one boolean, so a kernel that ran and merely
    missed tolerance reported `compiled: 0`. The scheduler reads that as a
    compile failure and tells the next author to rewrite the candidate, while
    its "repair the smallest reported numerical mismatch" branch -- the correct
    advice, and the only one that names the real problem -- requires
    `compiled and not correctness` and so was unreachable for every CuTe task.
    Measured over 71 arm worktrees, 44 of 224 failing turns were numeric
    near-misses steered into a rewrite this way: one arm burned all six turns
    circling a single rounding-placement error, and a Qwen turn that missed at
    0.0896 against a 0.08 threshold was told to start over.

    The generated evaluator raises `RuntimeError: validation failed: ...` from
    its appended `main()`, after compilation and after the kernels have run.
    That phrase is the shared contract of every task template here, so it is
    the signal: reaching it proves the candidate produced output.
    """
    response = record.get("response") or {}
    stderr = response.get("stderr") if isinstance(response, Mapping) else ""
    return "validation failed" in str(stderr or "")


def failure_reason(record: dict[str, Any]) -> str:
    """Say why a B300 run failed, for the next turn's feedback.

    The remote harness reports the fault only in `response.stderr`. Without
    this the island record carried an empty `error`, so the barrier loop handed
    the next author a diagnostic containing no diagnosis and it could not act
    on what went wrong. Downstream feedback truncates, so lead with the final
    exception line rather than the head of the traceback.
    """
    response = record.get("response") or {}
    if response.get("timed_out") is True:
        return "B300 run timed out"
    # The device colours some diagnostics, and CuTe raises dotted exception
    # names. `\w*` stops at the first dot, so
    # `cutlass.cute.nvgpu.common.OpError: expects the 'cta_group' Op parameter
    # to be a tcgen05.CtaGroup instance` -- the most actionable line the run
    # produces -- never matched, and the author was told only that the success
    # pattern was absent.
    stderr = ANSI.sub("", str(response.get("stderr") or ""))
    parts = []
    for line in reversed(stderr.strip().splitlines()):
        if re.match(r"\s*[\w.]*(Error|Exception)\b", line):
            parts.append(line.strip())
            break
    exit_code = response.get("exit_code")
    if exit_code not in (None, 0):
        parts.append(f"exit_code={exit_code}")
    if not record.get("success_pattern_matched", True):
        parts.append("success pattern absent from stdout")
    elif record.get("kernel_time_ms") is None:
        parts.append("no kernel_time_ms in stdout")
    summary = "; ".join(parts) if parts else ""
    # Whatever the pattern caught, hand over the tail as well. A one-line
    # exception rarely says which call raised it; the traceback beneath it does,
    # and the author cannot read the file itself -- `**/b300/**` is denied.
    tail = "\n".join(stderr.strip().splitlines()[-STDERR_TAIL_LINES:]).strip()
    if summary and tail:
        return f"{summary}\n--- stderr (last {STDERR_TAIL_LINES} lines) ---\n{tail}"
    return summary or tail or "B300 run failed without diagnostic output"


def _kernel_time_ms(stdout: str) -> float | None:
    matches = re.findall(r"(?<![A-Za-z0-9_])kernel_time_ms=([0-9]+(?:\.[0-9]+)?)", stdout)
    return float(matches[-1]) if matches and float(matches[-1]) > 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b300_lock_path() -> Path:
    return get_repo_root() / ".kernelevo" / "b300.lock"


def _candidate_kernel_symbols(path: Path) -> tuple[str, ...]:
    """Read candidate-declared ``@cute.kernel`` names for timeline attribution."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return ()
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            name = _ast_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if name in {"kernel", "cute.kernel"}:
                names.append(node.name)
                break
    return tuple(sorted(set(names)))


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class RemoteHarnessError(RuntimeError):
    pass


class HarnessClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RemoteHarnessError(f"invalid B300 harness URL: {base_url}")
        if not api_key:
            raise RemoteHarnessError("CUTE_HARNESS_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _open(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RemoteHarnessError(f"B300 harness HTTP {error.code}: {body[:1000]}") from error
        except URLError as error:
            raise RemoteHarnessError(f"cannot reach B300 harness: {error.reason}") from error

    def run_file(self, submission: Path, profiler: str) -> dict[str, Any]:
        body, boundary = _multipart(submission, profiler)
        payload = self._open(
            Request(
                f"{self.base_url}/v1/runs/file",
                data=body,
                method="POST",
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "X-API-Key": self.api_key,
                },
            )
        )
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise RemoteHarnessError("B300 harness returned a non-object response")
        return value

    def download_profile(self, profile_id: str) -> bytes:
        if "/" in profile_id or "\\" in profile_id:
            raise RemoteHarnessError("invalid profile id")
        return self._open(
            Request(
                f"{self.base_url}/v1/profiles/{profile_id}",
                headers={"X-API-Key": self.api_key},
            )
        )


def _multipart(submission: Path, profiler: str) -> tuple[bytes, str]:
    boundary = f"kernel-evo-{uuid.uuid4().hex}"
    newline = b"\r\n"
    chunks = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{submission.name}"'.encode(),
        b"Content-Type: text/x-python",
        b"",
        submission.read_bytes(),
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="profiler"',
        b"",
        profiler.encode(),
        f"--{boundary}--".encode(),
        b"",
    ]
    return newline.join(chunks), boundary
