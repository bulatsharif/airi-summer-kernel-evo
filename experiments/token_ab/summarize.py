from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any


TOKEN_FIELDS = {
    "input_uncached": r"^Token input uncached:\s+([\d,]+)\s*$",
    "input_cached": r"^Token input cached:\s+([\d,]+)\s*$",
    "outputs": r"^Token outputs:\s+([\d,]+)\s*$",
    "reasoning": r"^Token reasoning:\s+([\d,]+)\s*$",
}

TRIAL_COLUMNS = (
    "task_id",
    "arm",
    "trial_id",
    "agent_status",
    "local_check_passed",
    "evaluation_attempted",
    "evaluated",
    "passed",
    "logical_tokens",
    "input_uncached",
    "input_cached",
    "outputs",
    "reasoning",
    "agent_wall_seconds",
    "remote_wall_seconds",
    "device_time_ms",
    "profile_id",
    "candidate_sha256",
    "run_dir",
)

AGGREGATE_COLUMNS = (
    "task_id",
    "arm",
    "trials",
    "evaluation_attempts",
    "result_records",
    "successes",
    "success_rate",
    "median_logical_tokens_success",
    "median_logical_tokens_observed",
)


def parse_tokens(output: str) -> dict[str, int | None]:
    parsed: dict[str, int | None] = {}
    for field, pattern in TOKEN_FIELDS.items():
        match = re.search(pattern, output, flags=re.MULTILINE)
        parsed[field] = (
            int(match.group(1).replace(",", "")) if match else None
        )
    return parsed


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _agent_wall_seconds(metadata: dict[str, Any], status_path: Path) -> float | None:
    started = metadata.get("started_at")
    if not isinstance(started, str) or not status_path.is_file():
        return None
    try:
        started_at = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finished_at = datetime.fromtimestamp(
            status_path.stat().st_mtime,
            tz=started_at.tzinfo,
        )
    except (OSError, ValueError):
        return None
    return round(max(0.0, (finished_at - started_at).total_seconds()), 3)


def load_trial(metadata_path: Path) -> dict[str, Any]:
    run_dir = metadata_path.parent
    metadata = _read_json(metadata_path)
    if metadata is None:
        raise ValueError(f"metadata disappeared: {metadata_path}")

    output_path = run_dir / "output.log"
    output = (
        output_path.read_text(encoding="utf-8", errors="replace")
        if output_path.is_file()
        else ""
    )
    tokens = parse_tokens(output)
    token_values = list(tokens.values())
    logical_tokens = (
        sum(value for value in token_values if value is not None)
        if all(value is not None for value in token_values)
        else None
    )

    status_path = run_dir / "status"
    agent_status = (
        status_path.read_text(encoding="utf-8").strip()
        if status_path.is_file()
        else "missing"
    )
    result = _read_json(run_dir / "evaluation" / "result.json")
    evaluation_attempted = (run_dir / "evaluation-attempt").is_dir()
    acceptance = result.get("acceptance", {}) if result else {}
    response = result.get("response", {}) if result else {}

    return {
        "task_id": metadata.get("task_id"),
        "arm": metadata.get("arm"),
        "trial_id": metadata.get("trial_id"),
        "agent_status": agent_status,
        "local_check_passed": "check=PASS" in output,
        "evaluation_attempted": evaluation_attempted,
        "evaluated": result is not None,
        "passed": acceptance.get("passed") if result else None,
        "logical_tokens": logical_tokens,
        **tokens,
        "agent_wall_seconds": _agent_wall_seconds(metadata, status_path),
        "remote_wall_seconds": result.get("wall_seconds") if result else None,
        "device_time_ms": response.get("device_time_ms") if result else None,
        "profile_id": response.get("profile_id") if result else None,
        "candidate_sha256": result.get("candidate_sha256") if result else None,
        "run_dir": str(run_dir),
    }


def aggregate_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trial in trials:
        key = (str(trial["task_id"]), str(trial["arm"]))
        groups.setdefault(key, []).append(trial)

    rows: list[dict[str, Any]] = []
    for (task_id, arm), group in sorted(groups.items()):
        attempted = [trial for trial in group if trial["evaluation_attempted"]]
        result_records = [trial for trial in group if trial["evaluated"]]
        successes = [trial for trial in attempted if trial["passed"] is True]
        successful_tokens = [
            trial["logical_tokens"]
            for trial in successes
            if trial["logical_tokens"] is not None
        ]
        observed_tokens = [
            trial["logical_tokens"]
            for trial in group
            if trial["logical_tokens"] is not None
        ]
        rows.append(
            {
                "task_id": task_id,
                "arm": arm,
                "trials": len(group),
                "evaluation_attempts": len(attempted),
                "result_records": len(result_records),
                "successes": len(successes),
                "success_rate": (
                    round(len(successes) / len(attempted), 6)
                    if attempted
                    else None
                ),
                "median_logical_tokens_success": (
                    statistics.median(successful_tokens)
                    if successful_tokens
                    else None
                ),
                "median_logical_tokens_observed": (
                    statistics.median(observed_tokens)
                    if observed_tokens
                    else None
                ),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    metadata_paths = sorted(args.run_root.rglob("metadata.json"))
    trials = [load_trial(path) for path in metadata_paths]
    aggregates = aggregate_trials(trials)

    if args.json:
        print(
            json.dumps(
                {"schema_version": 1, "trials": trials, "aggregates": aggregates},
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.aggregate:
        _write_csv(aggregates, AGGREGATE_COLUMNS)
    else:
        _write_csv(trials, TRIAL_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
