"""Documentation ablation over the KernelEvo `iter` barrier loop with OpenCode authors.

Each arm is one independent KernelEvo run pinned to one documentation tier. The
runner drives `prepare -> author -> evaluate -> advance` and delegates only the
authoring turn to a bounded OpenCode subagent.

Author sessions run concurrently. B300 evaluation serializes itself through the
repository-wide `.kernelevo/b300.lock`, so every arm must be launched from this
one checkout; separate clones or worktrees would each take their own lock and
overlap timed runs on the single remote device.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import yaml

from kernel_evo import KernelEvoAgent
from kernel_evo.cute_harness.ablation import documentation_bundle
from kernel_evo.cute_harness.b300 import load_task
from kernel_evo.cute_harness.critic import (
    build_diagnostic,
    critic_task_markdown,
    parse_critic_hints,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_AGENT = "cute-fp8-author"
CRITIC_AGENT = "cute-fp8-critic"
# The write is the deliverable, so it leads. An earlier phrasing put reading
# first ("Read only X, then write ...") and sessions reliably spent the turn
# reading and planning, ending with prose like "Next Step: write submission.py"
# and no write call at all -- finish reason `stop`, well inside the step budget.
PROMPT = (
    "Read {task_file}, then implement the kernel it describes in the candidate "
    "file it names, following the loop in your instructions: write, check, "
    "evaluate, fix. Correctness first."
)
# One terse re-ask when a session ends having written nothing at all.
RETRY_PROMPT = (
    "The candidate file is unchanged. Write the complete kernel to exactly "
    "this path with a single write call, and say nothing else: {candidate}"
)
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
    "total_tokens",
)
SESSION_TOKENS_SQL = """
WITH RECURSIVE run_sessions(id) AS (
  SELECT '{session_id}'
  UNION ALL
  SELECT s.id FROM session AS s JOIN run_sessions AS r ON s.parent_id = r.id
)
SELECT
  COUNT(*) AS sessions,
  COALESCE(SUM(tokens_input), 0) AS input_uncached,
  COALESCE(SUM(tokens_cache_read), 0) AS input_cached,
  COALESCE(SUM(tokens_output), 0) AS outputs,
  COALESCE(SUM(tokens_reasoning), 0) AS reasoning
FROM session
WHERE id IN (SELECT id FROM run_sessions)
"""


def timeout_command() -> list[str]:
    """GNU timeout, which is `gtimeout` on macOS."""
    for name in ("gtimeout", "timeout"):
        path = shutil.which(name)
        if not path:
            continue
        version = subprocess.run([path, "--version"], capture_output=True, text=True)
        if "coreutils" in version.stdout.lower():
            return [path]
    raise RuntimeError("GNU timeout is required (macOS: brew install coreutils)")


def freeze_documentation(root: Path, task: Any, tier: str) -> dict[str, Any]:
    """Snapshot the tier bundle and its digests once per study."""
    output = root / "frozen_context" / tier
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    bundle = documentation_bundle(task, tier)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle.md").write_text(bundle.text, encoding="utf-8")
    manifest = {
        "tier": tier,
        "tokens_cl100k": bundle.tokens_cl100k,
        "files": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in bundle.files
        ],
        "bundle_sha256": hashlib.sha256(bundle.text.encode()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def session_tokens(session_id: str) -> dict[str, int]:
    if not session_id:
        return dict.fromkeys(TOKEN_FIELDS, 0) | {"sessions": 0}
    query = SESSION_TOKENS_SQL.format(session_id=session_id)
    result = subprocess.run(
        ["opencode", "db", "--format", "json", query],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return dict.fromkeys(TOKEN_FIELDS, 0) | {"sessions": 0}
    row = (json.loads(result.stdout) or [{}])[0]
    output_tokens = int(row.get("outputs", 0) or 0)
    reasoning = int(row.get("reasoning", 0) or 0)
    uncached = int(row.get("input_uncached", 0) or 0)
    cached = int(row.get("input_cached", 0) or 0)
    return {
        "sessions": int(row.get("sessions", 0) or 0),
        "input_tokens": uncached + cached,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "visible_output_tokens": max(0, output_tokens - reasoning),
        "total_tokens": uncached + cached + output_tokens,
    }


def author_candidate(
    task: Any,
    trace_dir: Path,
    model: str,
    session_timeout: int,
) -> dict[str, Any]:
    """Run one bounded OpenCode authoring session against a prepared packet."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    progress = trace_dir / "progress.jsonl"
    started = time.monotonic()
    result = run_session(
        AUTHOR_AGENT, model, session_timeout, authoring_prompt(task), progress
    )
    # A session that left the evaluated candidate untouched produced nothing to
    # grade; re-ask once, imperatively. This adds no task information, so it
    # does not change what a tier tells the author. When the session wrote
    # somewhere else, say where -- that is the whole error.
    candidate = Path(task.candidate_path)
    # A session can die on a transient provider error -- one arm ended every
    # turn on "The operation timed out" from a single slow request -- and the
    # turn then reports no candidate at all. Surface it rather than reading the
    # empty result as a refusal to write.
    failure = session_error(progress)
    retried = False
    if not wrote_candidate(progress, candidate):
        retried = True
        stray = [p for p in written_paths(progress) if Path(p).name.endswith(".py")]
        prompt = RETRY_PROMPT.format(candidate=candidate)
        if stray:
            prompt += (
                f" You wrote to {stray[-1]} instead, which this run does not read."
            )
        retry = run_session(AUTHOR_AGENT, model, session_timeout, prompt, progress)
        result = retry if wrote_candidate(progress, candidate) else result
    (trace_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    usage = {
        **session_tokens(session_of(result.stdout)),
        "session_id": session_of(result.stdout),
        "exit_code": result.returncode,
        "timed_out": result.returncode == 124,
        "author_seconds": time.monotonic() - started,
        "no_write_retry": retried,
        "session_error": failure,
        "session_error_after_retry": session_error(progress),
        "wrote_candidate": wrote_candidate(progress, candidate),
        "written_paths": written_paths(progress),
        # Evaluations the author spent inside its own turn, and whether any of
        # them passed. The barrier still evaluates once authoritatively after
        # this, so these count the debugging loop, not the recorded result.
        **agent_evaluations(candidate),
    }
    (trace_dir / "usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    return usage


def attempt_depth(stderr: str, passed: bool) -> int:
    """How far a candidate got, so the next turn can resume from the best one.

    Handing forward the *last* attempt loses ground whenever a turn ends worse
    than it began: one run reached a kernel that executed on the device and
    failed only on tolerance, then the next turn restarted from the skeleton and
    went back to a missing attribute.
    """
    if passed:
        return 4
    text = stderr.lower()
    if "validation failed" in text:
        return 3  # compiled, launched, ran -- only the numbers are wrong
    if "typeerror" in text:
        return 2  # found the symbol, called it wrongly
    if "attributeerror" in text or "importerror" in text:
        return 1  # wrong symbol
    return 0


def furthest_attempt(run_dir: Path, iteration: int) -> Path | None:
    """The deepest-reaching candidate from any earlier turn, else the last one."""
    best: tuple[int, int, Path] | None = None
    for previous in range(1, iteration):
        island = run_dir / f"iter_{previous:03d}" / "island_0"
        for record in sorted((island / "agent-evals").glob("eval-*/result.json")):
            candidate = record.with_name("candidate.py")
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stderr = str((payload.get("response") or {}).get("stderr") or "")
            depth = attempt_depth(stderr, bool(payload.get("passed")))
            if best is None or (depth, previous) >= (best[0], best[1]):
                best = (depth, previous, candidate)
    if best is not None:
        return best[2]
    fallback = run_dir / f"iter_{iteration - 1:03d}" / "island_0" / "candidate" / "submission.py"
    return fallback if fallback.is_file() else None


def attach_previous_attempt(run_dir: Path, iteration: int) -> bool:
    """Put the previous turn's kernel beside this turn's packet.

    Promotion stays strictly valid-only, so an invalid candidate never becomes
    an elite and the barrier still starts every turn from the skeleton. That
    left the author unable to see the code its own diagnostic refers to, and
    six turns became six independent single-shot attempts. What may be learned
    from and what may be promoted are different questions; this answers only
    the first, and touches no archive state.
    """
    if iteration < 2:
        return False
    current = run_dir / f"iter_{iteration:03d}" / "island_0"
    if not current.is_dir():
        return False
    previous = furthest_attempt(run_dir, iteration)
    if previous is None or not previous.is_file():
        return False
    seed = run_dir / "seed" / "submission.py"
    # An untouched skeleton is what this turn already starts from; showing it
    # back would be noise.
    if seed.is_file() and previous.read_bytes() == seed.read_bytes():
        return False
    target = current / "context" / "PREVIOUS_ATTEMPT.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous, target)

    result_path = run_dir / f"iter_{iteration - 1:03d}" / "island_0" / "result.json"
    error = ""
    if result_path.is_file():
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = {}
        error = str((record.get("result") or {}).get("error", "") or "")
    feedback = current / "context" / "FEEDBACK.md"
    note = (
        f"\n- Your previous attempt is `{target}`. It failed with: "
        f"{error or 'no diagnostic recorded'}. Read it, fix that error, and write the "
        "corrected kernel to the candidate path. Do not start from the empty skeleton.\n"
    )
    with feedback.open("a", encoding="utf-8") as handle:
        handle.write(note)

    # The author is told to read only what the packet lists, so an unlisted
    # file is invisible however loudly the feedback names it.
    task_file = current / "context" / "TASK.md"
    if task_file.is_file():
        text = task_file.read_text(encoding="utf-8")
        bullet = f"- `{target}`\n"
        if bullet not in text:
            marker = "\n## Contract"
            task_file.write_text(
                text.replace(marker, f"{bullet}{marker}", 1) if marker in text else text + bullet,
                encoding="utf-8",
            )
    return True


def agent_evaluations(candidate: Path) -> dict[str, Any]:
    """Summarize the B300 runs the author spent inside its own turn.

    `tools/cute-eval` records one directory per call beside the island. Reading
    them here keeps the count in the same place as the token accounting, so a
    report can say how many device runs a solve actually took.
    """
    root = candidate.resolve().parents[1] / "agent-evals"
    records = sorted(root.glob("eval-*/result.json")) if root.is_dir() else []
    passed = 0
    for path in records:
        try:
            passed += bool(json.loads(path.read_text(encoding="utf-8")).get("passed"))
        except (OSError, json.JSONDecodeError):
            continue
    return {"agent_evaluations": len(records), "agent_evaluations_passed": passed}


def session_error(progress: Path) -> str:
    """The last provider-level error in a transcript, if the session died on one.

    Distinguishes "the model declined to write" from "the request failed", which
    look identical in the candidate and led to a model being judged unable to
    author when every one of its turns had ended on a timed-out request.
    """
    if not progress.is_file():
        return ""
    latest = ""
    for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "error":
            continue
        error = event.get("error") or {}
        message = str((error.get("data") or {}).get("message") or error.get("name") or "")
        if message:
            latest = message
    return latest


def written_paths(progress: Path) -> list[str]:
    """Every path a session successfully wrote to, in order."""
    if not progress.is_file():
        return []
    paths = []
    for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        if part.get("type") != "tool" or part.get("tool") not in {"write", "edit", "patch"}:
            continue
        state = part.get("state") or {}
        if str(state.get("status", "")) != "completed":
            continue
        target = str((state.get("input") or {}).get("filePath", ""))
        if target:
            paths.append(target)
    return paths


def wrote_candidate(progress: Path, candidate: Path) -> bool:
    """True only when the session wrote the candidate this turn will evaluate.

    Checking merely that some write succeeded is not enough: sessions wrote a
    complete kernel to an invented path that the old edit glob happened to
    allow, so the write reported success while the evaluated candidate stayed
    an untouched skeleton.
    """
    resolved = str(candidate.resolve())
    return any(str(Path(path).resolve()) == resolved for path in written_paths(progress))


def authoring_prompt(task: Any) -> str:
    """Prompt delivery puts the whole tier bundle ahead of the packet instructions."""
    prompt = PROMPT.format(task_file=task.task_file)
    context = getattr(task, "prompt_context_file", None)
    if not context:
        return prompt
    # Do not claim the bundle exists in no file. It is materialized at
    # <island>/context/DOCUMENTATION.md, and an author that finds the file after
    # being told it does not exist has been given a false statement about its
    # own environment -- one read it four times having been told otherwise.
    return (
        "The task statement and the documentation for this run are reproduced "
        "in full below. You do not need to open anything to read them.\n\n"
        + Path(context).read_text(encoding="utf-8").rstrip()
        + "\n\n---\n\n"
        + prompt
    )


def run_session(
    agent: str,
    model: str,
    session_timeout: int,
    prompt: str,
    transcript: Path,
) -> subprocess.CompletedProcess[str]:
    """One bounded OpenCode session; the transcript is always written."""
    result = subprocess.run(
        [
            *timeout_command(),
            "--signal=TERM",
            "--kill-after=10s",
            str(session_timeout),
            "opencode",
            "run",
            "--format",
            "json",
            "--dir",
            str(REPO_ROOT),
            "--agent",
            agent,
            "--model",
            model,
            prompt,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(result.stdout, encoding="utf-8")
    return result


def session_of(stdout: str) -> str:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("sessionID"):
            return str(event["sessionID"])
    return ""


def critique_iteration(
    controller: Any,
    run_id: str,
    iteration: int,
    task_id: str,
    trace_dir: Path,
    model: str,
    session_timeout: int,
) -> dict[str, Any]:
    """Run one read-only critic between turns and hand its hints to the next packet."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    hints_path = trace_dir / "hints.json"
    if hints_path.exists():
        recorded = json.loads(hints_path.read_text(encoding="utf-8"))
        controller.record_critic_hints(run_id, recorded["hints"], iteration=iteration)
        return recorded

    run_dir = controller.store.run_dir(run_id)
    island_dir = run_dir / f"iter_{iteration:03d}" / "island_0"
    island = controller.store.read_state(run_id)["iterations"][str(iteration)]["islands"]["0"]
    b300_dir = island_dir / "b300"
    diagnostic = build_diagnostic(
        result=island.get("result") or {},
        stderr=read_if_present(b300_dir / "stderr.txt"),
        stdout=read_if_present(b300_dir / "stdout.txt"),
        profile_summary=str(island.get("profile_summary", "")),
    )
    task_path = trace_dir / "CRITIC.md"
    task_path.write_text(
        critic_task_markdown(
            task_id=task_id,
            iteration=iteration,
            candidate_path=str((run_dir / island["candidate_path"]).resolve()),
            diagnostic=diagnostic,
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    result = run_session(
        CRITIC_AGENT,
        model,
        session_timeout,
        f"Read only {task_path} and the candidate it names, then reply with the JSON it specifies.",
        trace_dir / "progress.jsonl",
    )
    (trace_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    hints = parse_critic_hints(result.stdout)
    session_id = session_of(result.stdout)
    recorded = {
        "iteration": iteration,
        "hints": controller.record_critic_hints(run_id, hints, iteration=iteration),
        "session_id": session_id,
        "exit_code": result.returncode,
        "timed_out": result.returncode == 124,
        "critic_seconds": time.monotonic() - started,
        "critic_tokens": session_tokens(session_id)["total_tokens"],
    }
    hints_path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    return recorded


def read_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def protocol(config: dict[str, Any]) -> dict[str, Any]:
    """The settings that make two arms incomparable if they differ."""
    settings = config.get("agent", {})
    return {
        "task": config.get("task"),
        "model": config.get("model"),
        "documentation_delivery": documentation_delivery(config),
        "profiler_feedback": bool((config.get("profiling") or {}).get("enabled", False)),
        "critic": critic_enabled(config),
        "b300_seed": settings.get("b300_seed"),
        "steps": settings.get("steps"),
        "islands": settings.get("islands"),
        "max_repairs_per_island": settings.get("max_repairs_per_island"),
        "evaluator": config.get("evaluator"),
    }


def check_protocol_matches(root: Path, config: dict[str, Any]) -> None:
    """Refuse to write a second protocol into a results root.

    Arms are keyed by tier and replication only, so re-running one root under a
    changed protocol would resume some arms, skip others, and silently mix both
    into one analysis.
    """
    existing = root / "study.yaml"
    if not existing.is_file():
        return
    previous = protocol(yaml.safe_load(existing.read_text(encoding="utf-8")) or {})
    current = protocol(config)
    if previous == current:
        return
    changed = sorted(key for key in current if previous.get(key) != current.get(key))
    raise ValueError(
        f"{existing} was written under a different protocol ({', '.join(changed)} changed); "
        "point --results at a new directory instead of mixing arms into one root"
    )


def documentation_delivery(config: dict[str, Any]) -> str:
    return str((config.get("documentation") or {}).get("delivery", "files"))


def critic_enabled(config: dict[str, Any]) -> bool:
    return bool((config.get("feedback") or {}).get("critic", False))


def agent_config(config: dict[str, Any], tier: str, task_path: Path) -> dict[str, Any]:
    settings = config["agent"]
    evaluation = config["evaluator"]
    profiling = config.get("profiling") or {}
    from_scratch = settings["b300_seed"] == "starter"
    profile_config = {"enabled": bool(profiling.get("enabled", False))}
    if "timeline" in profiling:
        profile_config["timeline"] = bool(profiling["timeline"])
    return {
        "run": {
            "name": f"iter-{tier}",
            "steps": settings["steps"],
            "islands": settings["islands"],
            # The starter is intentionally incomplete, so it cannot be preflighted.
            "seed_preflight": not from_scratch,
            "documentation_tier": tier,
            # `files` lets the author open the tier; `prompt` hands it the whole
            # bundle up front. Same bytes, different retrieval burden.
            "documentation_delivery": documentation_delivery(config),
            "b300_seed": settings["b300_seed"],
            "max_repairs_per_island": settings["max_repairs_per_island"],
        },
        "problem": {"path": str(task_path), "backend": "cute"},
        "evaluation": {
            "kind": "cute_b300",
            "precision": "fp8",
            "runtime_precision": "fp32",
            "measurement_mode": "device-time",
            "seed": evaluation["seed"],
            "warmup": evaluation["warmup"],
            "repeats": evaluation["repeats"],
            "timeout": evaluation["timeout_seconds"],
        },
        "scheduler": {
            "ideas": [
                {
                    "id": "implement-cute-fp8-kernel",
                    "summary": (
                        "Implement the task's CuTe DSL FP8 kernel so it compiles and "
                        "produces numerically correct output on B300."
                    ),
                }
                if from_scratch
                else {
                    "id": "conservative-b300-optimization",
                    "summary": (
                        "Make one conservative performance improvement to the verified "
                        "Blackwell FP8 candidate while preserving its working pipeline."
                    ),
                }
            ]
        },
        # When enabled, each turn also receives the previous candidate's B300
        # kernel breakdown as PARENT_PROFILE.md. Off is the frozen protocol.
        "profiling": profile_config,
        "cute": {"harness_enabled": True},
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [sample for sample in samples if sample["passed"]]
    best_speedup = max((float(sample["speedup"]) for sample in passed), default=0.0)
    first_pass = next((index for index, sample in enumerate(samples) if sample["passed"]), None)
    return {
        "samples": len(samples),
        "pass_count": len(passed),
        "accelerated_count": sum(float(sample["speedup"]) > 1 for sample in passed),
        "per_sample_pass_rate": len(passed) / len(samples) if samples else 0.0,
        "chains": 1,
        "chains_solved": int(bool(passed)),
        "best_speedup": best_speedup,
        "best_runtime_ms": min((float(s["runtime_ms"]) for s in passed), default=None),
        "floor_1_speedup": max(1.0, best_speedup),
        "floor_speedups": [max(1.0, best_speedup)],
        "fast_1_at_n": [
            int(any(item["passed"] and float(item["speedup"]) > 1 for item in samples[:index]))
            for index in range(1, len(samples) + 1)
        ],
        "llm_requests": sum(int(sample.get("sessions", 1) or 1) for sample in samples),
        "primary_requests": len(samples),
        "repair_requests": 0,
        "output_cap_hits": 0,
        "session_timeouts": sum(bool(sample.get("timed_out")) for sample in samples),
        "missing_jit_entrypoint": sum(
            "not decorated with jit decorator" in str(sample.get("error", "")) for sample in samples
        ),
        "no_remote_result": 0,
        "tokens_to_first_pass": (
            sum(int(s["total_tokens"]) for s in samples[: first_pass + 1]) if first_pass is not None else None
        ),
        **{field: sum(int(sample.get(field, 0)) for sample in samples) for field in TOKEN_FIELDS},
        "author_seconds": sum(float(sample["author_seconds"]) for sample in samples),
        "evaluation_seconds": sum(float(sample["evaluation_seconds"]) for sample in samples),
        "critic_calls": sum(bool(sample.get("critic_hints", 0)) for sample in samples),
        "critic_hints": sum(int(sample.get("critic_hints", 0) or 0) for sample in samples),
        "critic_seconds": sum(float(sample.get("critic_seconds", 0.0) or 0.0) for sample in samples),
        "critic_tokens": sum(int(sample.get("critic_tokens", 0) or 0) for sample in samples),
        "median_output_tokens": statistics.median(int(s.get("output_tokens", 0)) for s in samples)
        if samples
        else 0,
    }


def run_arm(
    root: Path,
    config: dict[str, Any],
    tier: str,
    replication: int,
    documentation_manifest: dict[str, Any],
) -> dict[str, Any]:
    task_id = config["task"]
    arm = root / "iter" / task_id / tier / f"r{replication:02d}"
    summary_path = arm / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    settings = config["agent"]
    task_path = REPO_ROOT / "tasks" / "cute" / "tasks" / task_id
    controller = KernelEvoAgent(arm / "kernel_evo")
    run_id = "run"
    run_dir = controller.runs_dir / run_id
    if not (run_dir / "state.json").exists():
        controller.init_run(agent_config(config, tier, task_path), run_id=run_id)

    while True:
        status = controller.status(run_id)
        phase = status["phase"]
        if phase == "complete":
            break
        iteration = int(status["current_iteration"])
        sample_path = arm / "samples" / f"iteration-{iteration:03d}.json"

        if phase in {"ready", "authoring"}:
            authoring_task = controller.prepare_iteration(run_id)[0]
            attach_previous_attempt(run_dir, iteration)
            trace_dir = authoring_task.candidate_path.parents[1] / "agent"
            usage_path = trace_dir / "usage.json"
            usage = (
                json.loads(usage_path.read_text(encoding="utf-8"))
                if usage_path.exists()
                else author_candidate(
                    authoring_task,
                    trace_dir,
                    config["model"],
                    int(settings["session_timeout_seconds"]),
                )
            )
        elif phase in {"evaluating", "evaluated"}:
            usage_path = run_dir / f"iter_{iteration:03d}" / "island_0" / "agent" / "usage.json"
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError(f"unexpected run phase: {phase}")

        started = time.monotonic()
        report = controller.evaluate_iteration(run_id)
        evaluation_seconds = time.monotonic() - started

        critique = {}
        if critic_enabled(config) and iteration < int(settings["steps"]):
            critique = critique_iteration(
                controller,
                run_id,
                iteration,
                task_id,
                run_dir / f"iter_{iteration:03d}" / "island_0" / "critic",
                config["model"],
                int(settings["session_timeout_seconds"]),
            )

        if not sample_path.exists():
            row = report["islands"][0]
            runtime_us = row.get("runtime_us")
            sample = {
                "approach": "iter",
                "task": task_id,
                "tier": tier,
                "replication": replication,
                "chain": 1,
                "turn": iteration,
                "passed": bool(row["valid"]),
                "runtime_ms": float(runtime_us) / 1000 if runtime_us else None,
                "speedup": float(row["speedup"]),
                "promoted": bool(row["promoted"]),
                "error": str(row.get("error", "")),
                "evaluation_seconds": evaluation_seconds,
                "critic_hints": len(critique.get("hints", [])),
                "critic_seconds": float(critique.get("critic_seconds", 0.0)),
                "critic_tokens": int(critique.get("critic_tokens", 0)),
                **usage,
            }
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        controller.advance_iteration(run_id)

    samples = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((arm / "samples").glob("iteration-*.json"))
    ]
    summary = summarize(samples)
    manifest = {
        "schema_version": 1,
        "approach": "iter",
        "task": task_id,
        "tier": tier,
        "replication": replication,
        "model": config["model"],
        "author": f"opencode/{AUTHOR_AGENT}",
        "chains": 1,
        "turns": settings["steps"],
        "islands": settings["islands"],
        "documentation_tokens_cl100k": documentation_manifest["tokens_cl100k"],
        "documentation_bundle_sha256": documentation_manifest["bundle_sha256"],
        "documentation_delivery": documentation_delivery(config),
        "profiler_feedback": bool((config.get("profiling") or {}).get("enabled", False)),
        "critic": f"opencode/{CRITIC_AGENT}" if critic_enabled(config) else "",
        "max_output_tokens": settings["max_output_tokens"],
        "evaluation": config["evaluator"],
    }
    arm.mkdir(parents=True, exist_ok=True)
    (arm / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"completed tier={tier} r{replication:02d}: "
        f"{summary['pass_count']}/{summary['samples']} passed",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("study-iter.yaml"))
    parser.add_argument("--model", default=os.environ.get("OPENCODE_MODEL", ""))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--tier", action="append")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, tooling, and tier bundles, then print the arm plan without "
        "starting any OpenCode session or B300 evaluation.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["model"] = args.model or config["model"]
    if args.concurrency is not None:
        config["agent"]["concurrency"] = args.concurrency
    if args.tier:
        unknown = set(args.tier) - set(config["tiers"])
        if unknown:
            raise ValueError(f"selection is outside the study config: {unknown}")
        config["tiers"] = args.tier
    if args.replications is not None:
        config["agent"]["replications"] = args.replications
    if args.steps is not None:
        config["agent"]["steps"] = args.steps

    if not args.dry_run:
        for name in ("QWEN_API_KEY", "QWEN_BASE_URL", "CUTE_HARNESS_API_KEY"):
            if not os.environ.get(name):
                raise ValueError(f"{name} is required")
    if not shutil.which("opencode"):
        raise ValueError("opencode is not available on PATH")
    timeout_command()

    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    check_protocol_matches(root, config)
    (root / "study.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment_path = root / "environment.json"
    previous = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    started_at = float(previous.get("started_at_unix", time.time()))
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    )
    environment = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "model": config["model"],
        "author": f"opencode/{AUTHOR_AGENT}",
        "python": sys.version,
        "author_concurrency": config["agent"]["concurrency"],
        "evaluation_concurrency": 1,
        "started_at_unix": started_at,
    }
    environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    task = load_task(config["task"])
    frozen = {tier: freeze_documentation(root, task, tier) for tier in config["tiers"]}
    arms = [
        (tier, replication)
        for tier in config["tiers"]
        for replication in range(1, config["agent"]["replications"] + 1)
    ]
    random.Random(config["order_seed"]).shuffle(arms)

    if args.dry_run:
        steps = int(config["agent"]["steps"])
        critic_sessions = len(arms) * max(0, steps - 1) if critic_enabled(config) else 0
        print(f"task:       {config['task']}")
        print(f"model:      {config['model']} via opencode/{AUTHOR_AGENT}")
        print(f"arms:       {len(arms)} ({len(config['tiers'])} tiers x "
              f"{config['agent']['replications']} replications)")
        print(f"delivery:   documentation by {documentation_delivery(config)}")
        print(f"feedback:   profiler={'on' if (config.get('profiling') or {}).get('enabled') else 'off'}, "
              f"critic={'on' if critic_enabled(config) else 'off'}")
        print(f"budget:     {len(arms) * steps} author sessions, "
              f"{critic_sessions} critic sessions, "
              f"{len(arms) * steps} B300 evaluations (serialized)")
        print(f"concurrency:{config['agent']['concurrency']} author streams, 1 evaluation stream")
        print("\ntier bundles:")
        for tier in config["tiers"]:
            manifest = frozen[tier]
            print(f"  {tier:<10} {manifest['tokens_cl100k']:>7,} cl100k tokens  "
                  f"{len(manifest['files'])} files  {manifest['bundle_sha256'][:12]}")
        print("\narm order:")
        for tier, replication in arms:
            print(f"  {tier}/r{replication:02d}")
        print("\nDry run: no OpenCode session or B300 evaluation was started.")
        return

    failures = []
    with ThreadPoolExecutor(max_workers=config["agent"]["concurrency"]) as pool:
        futures = {
            pool.submit(run_arm, root, config, tier, replication, frozen[tier]): (tier, replication)
            for tier, replication in arms
        }
        for future in as_completed(futures):
            tier, replication = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((tier, replication, str(error)))
                print(f"failed tier={tier} r{replication:02d}: {error}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} arm(s) failed; rerun to resume: {failures}")

    environment["completed_at_unix"] = time.time()
    environment["wall_seconds"] = environment["completed_at_unix"] - started_at
    environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("analyze.py")), str(root)],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
