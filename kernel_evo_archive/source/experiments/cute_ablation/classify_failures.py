"""Sort every failed B300 evaluation in a results root into a failure taxonomy.

An arm evaluates on two surfaces and both are walked here:

* **barrier** -- ``<island>/b300/``, one per turn, the graded metric.
* **in-turn** -- ``<island>/agent-evals/eval-*/``, the agent's own debugging
  runs inside a turn, capped by ``CUTE_AGENT_EVAL_BUDGET``.

Categories are the six named in the study protocol: import/API, compile,
layout/shape, numeric mismatch, timeout, other. Two cases are counted apart
from the six because collapsing them loses the distinction that matters:

* ``out_abs=0.000000`` in a numeric mismatch means the kernel launched and
  wrote nothing at all -- a dead kernel, not a mis-scaled one.
* an in-turn eval directory with no ``result.json`` never reached the device;
  the local policy gate rejected the candidate, so there is no stderr to
  classify.

Classification reads the *final* exception line rather than the whole
traceback, because CuTe tracebacks mention "layout" and "shape" in nearly every
frame. The exception pattern is ``[\\w.]+(?:Error|Exception)`` -- dotted, so
that ``cutlass.cute.nvgpu.common.OpError`` matches.

Usage:
    python experiments/cute_ablation/classify_failures.py results/e1-l2-63 ... \\
        --output experiments/cute_ablation/results/failure_taxonomy.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

CATEGORIES = (
    "import/API",
    "compile",
    "layout/shape",
    "numeric mismatch",
    "timeout",
    "other",
)

EXCEPTION_LINE = re.compile(r"^\s*([\w.]+(?:Error|Exception))\b\s*:?(.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
VALIDATION = re.compile(
    r"validation failed:\s*full_abs=([\d.eE+-]+),\s*sample_abs=([\d.eE+-]+),\s*"
    r"out_abs=([\d.eE+-]+)"
)

IMPORT_API_TYPES = {
    "ImportError",
    "ModuleNotFoundError",
    "AttributeError",
    "NameError",
    "TypeError",
}
IMPORT_API_TEXT = (
    "missing 1 required",
    "missing 2 required",
    "unexpected keyword",
    "has no attribute",
    "not callable",
    "no module named",
    "cannot import name",
    "takes no arguments",
    "positional argument",
)
LAYOUT_TEXT = (
    "shape",
    "size mismatch",
    "not divisible",
    "divisibility",
    "incompatible",
    "stride",
    "rank",
    "out of bounds",
    "index out of range",
    "dimension",
    "layout",
    # Tile-size constraints the DSL asserts before it will build an atom, e.g.
    # "num_columns must be multiple of 32 and power of two".
    "multiple of",
    "power of two",
)
COMPILE_TYPES = {
    "SyntaxError",
    "IndentationError",
    "OpError",
    "DSLRuntimeError",
    "CompilationError",
    "NotImplementedError",
}
COMPILE_TEXT = (
    "shared memory",
    "smem",
    "ptxas",
    "nvrtc",
    "mlir",
    "failed to legalize",
    "verification failed",
    "no registered",
    "cuda error",
    "invalid device function",
    "compil",
    # The DSL raises this from the MLIR op builder when an atom or copy cannot
    # be constructed at trace time -- a build failure, not a runtime one.
    "operation creation failed",
)
TIMEOUT_TEXT = ("timeout", "timed out", "killed")


def clean(text: str) -> str:
    return ANSI.sub("", text)


def final_exception(stderr: str) -> tuple[str, str]:
    """The last ``SomeError: message`` line of a traceback, dotted names included."""
    kind, message = "", ""
    for line in clean(stderr).splitlines():
        match = EXCEPTION_LINE.match(line)
        if match:
            kind = match.group(1).split(".")[-1]
            message = match.group(2).strip()
    return kind, message


def classify(record: dict[str, Any]) -> tuple[str, str]:
    """Return (category, the signal that decided it)."""
    stderr = clean(record.get("stderr", ""))
    stdout = clean(record.get("stdout", ""))
    blob = f"{stderr}\n{stdout}".lower()

    if record.get("timed_out"):
        return "timeout", "result.json timed_out=true"
    kind, message = final_exception(stderr)
    if kind in {"TimeoutError", "TimeoutExpired"} or any(w in blob for w in TIMEOUT_TEXT):
        if not kind or kind in {"TimeoutError", "TimeoutExpired"}:
            return "timeout", kind or "timeout text in output"

    validation = VALIDATION.search(stderr) or VALIDATION.search(stdout)
    if validation:
        return "numeric mismatch", f"validation failed: out_abs={validation.group(3)}"
    if "fail" in stdout.lower() and "max_abs=" in stdout.lower() and not kind:
        return "numeric mismatch", "FAIL line with max_abs in stdout"

    lowered = f"{kind}: {message}".lower()
    if kind in COMPILE_TYPES or any(w in lowered for w in COMPILE_TEXT):
        return "compile", f"{kind}: {message[:90]}" if kind else "compile text"
    if kind in IMPORT_API_TYPES and any(w in lowered for w in IMPORT_API_TEXT):
        return "import/API", f"{kind}: {message[:90]}"
    if any(w in lowered for w in LAYOUT_TEXT):
        return "layout/shape", f"{kind}: {message[:90]}"
    if kind in IMPORT_API_TYPES:
        return "import/API", f"{kind}: {message[:90]}"
    if kind:
        return "other", f"{kind}: {message[:90]}"
    return "other", (stderr.strip().splitlines() or ["no stderr"])[-1][:90]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def evaluations(root: Path) -> list[dict[str, Any]]:
    """Every evaluation under a results root, barrier and in-turn alike."""
    found = []
    for island in sorted(root.glob("iter/*/*/r*/kernel_evo/run/iter_*/island_*")):
        tier = island.parents[4].name
        turn = int(island.parent.name.split("_")[1])
        surfaces = [("barrier", island / "b300")]
        surfaces += [
            ("in-turn", directory)
            for directory in sorted((island / "agent-evals").glob("eval-*"))
        ]
        for surface, directory in surfaces:
            if not directory.is_dir():
                continue
            result_path = directory / "result.json"
            if not result_path.is_file():
                if surface == "in-turn":
                    found.append(
                        {
                            "tier": tier,
                            "turn": turn,
                            "surface": surface,
                            "name": directory.name,
                            "passed": False,
                            "gate_rejected": True,
                        }
                    )
                continue
            result = json.loads(read(result_path) or "{}")
            found.append(
                {
                    "tier": tier,
                    "turn": turn,
                    "surface": surface,
                    "name": directory.name,
                    "passed": bool(result.get("passed")),
                    "gate_rejected": False,
                    "timed_out": bool((result.get("response") or {}).get("timed_out")),
                    "kernel_time_ms": result.get("kernel_time_ms"),
                    "stderr": read(directory / "stderr.txt"),
                    "stdout": read(directory / "stdout.txt"),
                }
            )
    return found


def summarize(root: Path) -> dict[str, Any]:
    records = evaluations(root)
    failures = []
    counts: Counter[str] = Counter()
    surface_counts: dict[str, Counter[str]] = {
        "barrier": Counter(),
        "in-turn": Counter(),
    }
    zero_output = 0
    gate_rejected = 0
    for record in records:
        if record["passed"]:
            continue
        if record.get("gate_rejected"):
            gate_rejected += 1
            continue
        category, signal = classify(record)
        counts[category] += 1
        surface_counts[record["surface"]][category] += 1
        dead = "out_abs=0.000000" in signal
        zero_output += int(dead)
        failures.append(
            {
                "tier": record["tier"],
                "turn": record["turn"],
                "surface": record["surface"],
                "eval": record["name"],
                "category": category,
                "zero_output": dead,
                "signal": signal,
            }
        )
    return {
        "root": str(root),
        # One arm per results root under the worktree layout, so the tier names
        # the row; the bare root basename repeats across the four E1 worktrees.
        "tiers": sorted({record["tier"] for record in records}),
        "evaluations": len(records),
        "passed": sum(1 for record in records if record["passed"]),
        "failed": len(failures),
        "gate_rejected_no_device_run": gate_rejected,
        "counts": {name: counts.get(name, 0) for name in CATEGORIES},
        "counts_by_surface": {
            surface: {name: table.get(name, 0) for name in CATEGORIES}
            for surface, table in surface_counts.items()
        },
        "numeric_mismatch_with_zero_output": zero_output,
        "failures": failures,
    }


def render(summaries: list[dict[str, Any]]) -> str:
    header = "| root | evals | passed | failed | " + " | ".join(CATEGORIES) + " | zero-output | gate-rejected |"
    rule = "|" + "---|" * (len(CATEGORIES) + 6)
    lines = [header, rule]
    for summary in summaries:
        cells = " | ".join(str(summary["counts"][name]) for name in CATEGORIES)
        label = f"{Path(summary['root']).name}/{'+'.join(summary['tiers'])}"
        lines.append(
            f"| {label} | {summary['evaluations']} | {summary['passed']} | "
            f"{summary['failed']} | {cells} | {summary['numeric_mismatch_with_zero_output']} | "
            f"{summary['gate_rejected_no_device_run']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [summarize(root) for root in args.roots]
    print(render(summaries))
    print()
    for summary in summaries:
        for failure in summary["failures"]:
            print(
                f"{Path(summary['root']).name:12} {failure['tier']:9} turn{failure['turn']} "
                f"{failure['surface']:8} {failure['eval']:8} {failure['category']:17} "
                f"{failure['signal']}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
