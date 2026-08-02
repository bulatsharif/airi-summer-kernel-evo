"""Run a prepared candidate under bounded Compute Sanitizer tools in safe order."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Sequence

from kernel_evo.core.profile.contracts import run_profile_subprocess


_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def _parse_tools(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in _TOOLS]
    if invalid:
        raise ValueError(f"Unsupported sanitizer tools: {', '.join(invalid)}")
    ordered = [tool for tool in _TOOLS if tool in requested]
    if any(tool != "memcheck" for tool in ordered) and "memcheck" not in ordered:
        ordered.insert(0, "memcheck")
    return ordered or ["memcheck"]


def _error_count(output: str) -> int | None:
    matches = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+error", output, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def run_sanitizers(
    *,
    run_config: Path,
    candidate_file: Path,
    reference_file: Path,
    out_dir: Path,
    tools: Sequence[str],
    timeout: float,
) -> dict[str, object]:
    executable = shutil.which("compute-sanitizer")
    if not executable:
        return {
            "status": "unavailable",
            "passed": False,
            "reason": "compute-sanitizer executable not found",
            "tools": [],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for tool in tools:
        tool_dir = out_dir / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "--tool",
            tool,
            "--target-processes",
            "all",
            "--error-exitcode",
            "86",
            sys.executable,
            "-m",
            "kernel_evo.tools.profile_target",
            "--run-config",
            str(run_config),
            "--candidate-file",
            str(candidate_file),
            "--reference-file",
            str(reference_file),
            "--out-dir",
            str(tool_dir / "target"),
        ]
        completed = run_profile_subprocess(
            command,
            timeout=timeout,
            text=True,
            capture_output=True,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        errors = _error_count(combined)
        passed = completed.returncode == 0 and errors == 0
        result = {
            "tool": tool,
            "passed": passed,
            "returncode": completed.returncode,
            "error_count": errors,
            "summary": (
                f"{tool}: zero reported errors"
                if passed
                else f"{tool}: failed with return code {completed.returncode}, errors={errors}"
            ),
            "stdout_excerpt": completed.stdout[-2_000:],
            "stderr_excerpt": completed.stderr[-2_000:],
        }
        results.append(result)
        if tool == "memcheck" and not passed:
            break
    passed = bool(results) and all(bool(item["passed"]) for item in results)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "tools": results,
        "stopped_after_memcheck": bool(results) and results[-1]["tool"] == "memcheck" and not passed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tools", default="memcheck,synccheck")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    summary = run_sanitizers(
        run_config=Path(args.run_config).expanduser().resolve(),
        candidate_file=Path(args.candidate_file).expanduser().resolve(),
        reference_file=Path(args.reference_file).expanduser().resolve(),
        out_dir=out_dir,
        tools=_parse_tools(args.tools),
        timeout=args.timeout,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
