from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from .process import run_streaming


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _bash_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if os.name == "nt" and len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/{resolved[0].lower()}/{resolved[3:]}"
    return resolved


def _headless_runner_prefix(runner: Path) -> list[str]:
    if os.name != "nt":
        return [str(runner)]
    candidates = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Git"
        / "bin"
        / "bash.exe",
        Path(r"C:\msys64\usr\bin\bash.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate), _bash_path(runner)]
    raise RuntimeError(
        "OpenCode headless experiments on Windows require Git Bash or MSYS2"
    )


@dataclass(frozen=True)
class AgentMetrics:
    requested_model: str
    reported_model: str | None
    provider: str | None
    variant: str | None
    session_id: str | None
    sessions: int | None
    input_uncached: int | None
    input_cached: int | None
    cache_write: int | None
    output: int | None
    reasoning: int | None
    wall_seconds: float
    session_wall_seconds: float | None
    exit_code: int
    timed_out: bool
    metrics_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session_id_from_events(events_path: Path) -> str | None:
    if not events_path.is_file():
        return None
    with events_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = event.get("sessionID")
            if (
                isinstance(session_id, str)
                and SESSION_ID_PATTERN.fullmatch(session_id)
            ):
                return session_id
    return None


def _query_session_metrics(
    opencode_command: str,
    session_id: str,
) -> dict[str, Any]:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise RuntimeError("OpenCode emitted an invalid session id")
    query = f"""
WITH RECURSIVE run_sessions(id) AS (
  SELECT '{session_id}'
  UNION ALL
  SELECT child.id
  FROM session AS child
  JOIN run_sessions AS parent ON child.parent_id = parent.id
)
SELECT
  COUNT(*) AS sessions,
  COALESCE(SUM(tokens_input), 0) AS input_uncached,
  COALESCE(SUM(tokens_cache_read), 0) AS input_cached,
  COALESCE(SUM(tokens_cache_write), 0) AS cache_write,
  COALESCE(SUM(tokens_output), 0) AS output,
  COALESCE(SUM(tokens_reasoning), 0) AS reasoning,
  (
    SELECT model FROM session WHERE id = '{session_id}'
  ) AS root_model,
  (
    SELECT MAX(0, time_updated - time_created)
    FROM session WHERE id = '{session_id}'
  ) AS root_wall_ms
FROM session
WHERE id IN (SELECT id FROM run_sessions)
"""
    process = subprocess.run(
        [opencode_command, "db", "--format", "json", query],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if process.returncode != 0:
        message = process.stderr.strip() or "OpenCode database query failed"
        raise RuntimeError(message)
    payload = json.loads(process.stdout)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("OpenCode database query returned no rows")
    row = payload[0]
    if not isinstance(row, dict):
        raise RuntimeError("OpenCode database query returned an invalid row")
    return row


def _reported_model(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    raw = row.get("root_model")
    if not isinstance(raw, str) or not raw:
        return None, None, None
    try:
        model = json.loads(raw)
    except json.JSONDecodeError:
        return raw, None, None
    if not isinstance(model, dict):
        return raw, None, None
    model_id = model.get("id")
    provider = model.get("providerID")
    variant = model.get("variant")
    reported = (
        f"{provider}/{model_id}"
        if isinstance(provider, str) and isinstance(model_id, str)
        else model_id if isinstance(model_id, str) else None
    )
    return (
        reported,
        provider if isinstance(provider, str) else None,
        variant if isinstance(variant, str) else None,
    )


def build_agent_prompt(
    task_id: str,
    candidate_path: Path,
    seed: int,
    gpu_timeout: float,
) -> str:
    return f"""Solve the CuTe task in the current workspace.

Read TASK.md, task.json, and submission.py. Edit only submission.py.
Before planning or delegating, load every local OpenCode skill listed in
task.json.agent_skills, then read every file in task.json.references. The skill
contains the detailed CuTe handbook; task references contain dataset-specific
design constraints.
The final candidate must implement task {task_id} and must not define main();
the evaluation harness owns input generation, reference computation, timing,
and PASS reporting.
Do not inspect previous runs, workspaces, known baselines, or evaluator source.

Tool contract:

- Use the read tool for TASK.md, task.json, submission.py, and the installed
  skill/reference files.
- Resolve every path in task.json.references relative to the current workspace,
  exactly as written. Resolve links from SKILL.md relative to the directory
  containing that SKILL.md. Never reinterpret a workspace task reference as a
  path under the skill directory.
- Use the write/edit tool only for submission.py.
- Shell access is intentionally limited to the two plain harness command forms
  below. Do not add pipes, redirects, command chaining, wrappers, or shell
  utilities such as head, ls, find, grep, or which.
- Do not use python3 -c to inspect local CUTLASS: CuTe is authoritative only on
  the remote worker. The remote harness compiler error is the API oracle.
- After an error, fix the first concrete diagnostic and rerun the same harness
  command. Do not delegate environment or API exploration before doing that.
- The local check is only an AST/policy check. Interpret messages such as
  "found 0 @cute.kernel functions" literally; do not change imports or explore
  the environment in response.
- In candidate mode, read only the task reference and the skill's two
  explicitly selected candidate-mode files before the first implementation.
  Do not enumerate or read the rest of the installed skill. Read a broader
  handbook chapter only when a concrete compiler diagnostic requires it.
- For dense GEMM, copy and adapt the compile-verified candidate template
  selected by the task. Do not reconstruct its TMA/pipeline/TMEM flow from
  memory or mix it with another API family.
- Do not repeatedly restate the task contract. Preserve any pipeline that
  already reaches execution, make one minimal diagnostic-driven edit at a
  time, and rerun promptly.
- Never rerun an identical candidate after a worker timeout or CUDA launch
  failure. If one diagnostic repeats, restore the compile-verified template
  instead of inventing another API. Stop immediately after the first PASS.

Allowed feedback commands:

python3 -m cute_harness check {task_id} {candidate_path}
python3 -m cute_harness run {task_id} {candidate_path} --seed {seed} --timeout {gpu_timeout}

Establish correctness before optimizing. Leave the best final candidate at
{candidate_path}. Finish only after a successful harness result or an exact
blocker.
"""


def build_workspace_inline_config(
    repo_root: Path,
    workspace: Path,
    existing: str | None = None,
) -> str:
    try:
        payload = json.loads(existing) if existing else {}
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"OPENCODE_CONFIG_CONTENT is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("OPENCODE_CONFIG_CONTENT must be a JSON object")

    permission = payload.setdefault("permission", {})
    if not isinstance(permission, dict):
        raise RuntimeError("inline OpenCode permission must be an object")
    read = permission.setdefault("read", {})
    edit = permission.setdefault("edit", {})
    if not isinstance(read, dict) or not isinstance(edit, dict):
        raise RuntimeError("inline OpenCode read/edit permissions must be objects")

    repo_root = repo_root.resolve()
    workspace = workspace.resolve()
    patterns = [f"{workspace.as_posix()}/**"]
    try:
        relative = workspace.relative_to(repo_root).as_posix()
    except ValueError:
        external = permission.setdefault("external_directory", {})
        if not isinstance(external, dict):
            raise RuntimeError(
                "inline external_directory permission must be an object"
            )
        external[patterns[0]] = "allow"
    else:
        patterns.append(f"{relative}/**")

    for pattern in patterns:
        read[pattern] = "allow"
    candidate = (workspace / "submission.py").as_posix()
    edit[candidate] = "allow"
    try:
        relative_candidate = (workspace / "submission.py").relative_to(
            repo_root
        )
    except ValueError:
        pass
    else:
        edit[relative_candidate.as_posix()] = "allow"
    return json.dumps(payload, separators=(",", ":"))


def run_agent(
    *,
    repo_root: Path,
    workspace: Path,
    task_id: str,
    model: str,
    agent_timeout: float,
    gpu_timeout: float,
    seed: int,
    events_path: Path,
    log_path: Path,
    opencode_command: str = "opencode",
) -> AgentMetrics:
    runner = repo_root / "opencode" / "opencode-headless.sh"
    candidate = workspace / "submission.py"
    prompt = build_agent_prompt(task_id, candidate, seed, gpu_timeout)
    command = [
        *_headless_runner_prefix(runner),
        "--foreground",
        "--dir",
        _bash_path(workspace),
        "--timeout",
        f"{agent_timeout}s",
        "--progress",
        _bash_path(events_path),
        "--agents",
        _bash_path(workspace / "AGENTS.md"),
        "--model",
        model,
        "--require-cute-key",
        "--",
        prompt,
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else os.pathsep.join((str(repo_root), existing_pythonpath))
    )
    environment["OPENCODE_CONFIG_CONTENT"] = build_workspace_inline_config(
        repo_root,
        workspace,
        environment.get("OPENCODE_CONFIG_CONTENT"),
    )

    started = time.monotonic()
    process = run_streaming(
        command,
        cwd=repo_root,
        environment=environment,
        log_path=log_path,
        timeout=agent_timeout + 45.0,
    )
    exit_code = process.exit_code
    wall_seconds = time.monotonic() - started

    session_id = _session_id_from_events(events_path)
    row: dict[str, Any] = {}
    metrics_error = None
    if session_id:
        try:
            row = _query_session_metrics(opencode_command, session_id)
        except (json.JSONDecodeError, OSError, RuntimeError) as error:
            metrics_error = str(error)
    else:
        metrics_error = "OpenCode emitted no session id"

    reported_model, provider, variant = _reported_model(row)
    root_wall_ms = row.get("root_wall_ms")
    session_wall_seconds = (
        float(root_wall_ms) / 1000.0
        if isinstance(root_wall_ms, (int, float))
        else None
    )
    return AgentMetrics(
        requested_model=model,
        reported_model=reported_model,
        provider=provider,
        variant=variant,
        session_id=session_id,
        sessions=int(row["sessions"]) if "sessions" in row else None,
        input_uncached=(
            int(row["input_uncached"]) if "input_uncached" in row else None
        ),
        input_cached=(
            int(row["input_cached"]) if "input_cached" in row else None
        ),
        cache_write=int(row["cache_write"]) if "cache_write" in row else None,
        output=int(row["output"]) if "output" in row else None,
        reasoning=int(row["reasoning"]) if "reasoning" in row else None,
        wall_seconds=wall_seconds,
        session_wall_seconds=session_wall_seconds,
        exit_code=exit_code,
        timed_out=process.timed_out or exit_code == 124,
        metrics_error=metrics_error,
    )
