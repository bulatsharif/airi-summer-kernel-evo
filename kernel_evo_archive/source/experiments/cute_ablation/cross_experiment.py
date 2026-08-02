"""Assemble the E1-E4 arms into one table.

`analyze.py` reports a single results root -- the tier ladder inside it. The
grid spans four roots that differ by one knob each, so the cross-experiment view
has to be built across them. Every row carries the three metrics the protocol
requires of every table: pass/not-pass with turns-to-first-pass, speedup against
the arm's own measured B300 baseline, and the four-way token split.

`cached_input_tokens` is reported as measured. It is structurally 0 on an SGLang
endpoint, where it means "not reported" rather than "nothing was cached"; the
DeepSeek API does report it. The renderer prints `n/r` for a zero so the two
cases are never confused.

Usage:
    python experiments/cute_ablation/cross_experiment.py \\
        E1:results/e1-l2-63 E2:results/e2-l2-63 ... --output <path>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def baseline_ms(arm: Path) -> float | None:
    """The measured reference time this arm's speedups are divided into."""
    result = read_json(arm / "kernel_evo" / "run" / "b300" / "baseline" / "result.json")
    value = result.get("kernel_time_ms")
    return float(value) if value is not None else None


def launched_own_kernel(arm: Path, turn: int) -> bool:
    """Did this turn's candidate actually put a kernel on the device?

    A candidate that launches nothing still passes the harness: the output
    buffer it was handed already holds the reference result, so validation reads
    `full_max_abs=0.000000` and the timed region is empty. One arm scored four
    PASSes at 0.0128 ms that way -- 137 GFLOP in 12.8us is ~10.7 PFLOP/s, above
    the device's dense FP8 peak, which is what gave it away.

    The profile is the ground truth: a candidate kernel is emitted under a
    `kernel_cutlass_*` name. Everything else in the trace belongs to the harness
    (torch init, the fp32->fp8 convert, the cuBLAS reference GEMM).
    """
    result = read_json(
        arm / "kernel_evo" / "run" / f"iter_{turn:03d}" / "island_0" / "b300" / "result.json"
    )
    kernels = ((result.get("profile_summary") or {}).get("top_kernels")) or []
    return any(
        str(k.get("name", "")).startswith("kernel_cutlass_")
        and "convert" not in str(k.get("name", ""))
        for k in kernels
    )


def arms(label: str, root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.glob("iter/*/*/r*/summary.json")):
        arm = summary_path.parent
        summary = read_json(summary_path)
        manifest = read_json(arm / "manifest.json")
        samples = [
            read_json(path) for path in sorted((arm / "samples").glob("iteration-*.json"))
        ]
        # A harness PASS is necessary but not sufficient: it also has to have run
        # a kernel of its own. See launched_own_kernel().
        for sample in samples:
            sample["real"] = bool(sample.get("passed")) and launched_own_kernel(
                arm, int(sample["turn"])
            )
        passed = [sample for sample in samples if sample["real"]]
        spurious = [s for s in samples if s.get("passed") and not s["real"]]
        first = next((sample["turn"] for sample in samples if sample["real"]), None)
        best = min(passed, key=lambda s: float(s["runtime_ms"]), default=None)
        reference = baseline_ms(arm)
        rows.append(
            {
                "experiment": label,
                "tier": arm.parents[0].name,
                "root": str(root),
                "delivery": manifest.get("documentation_delivery", "files"),
                "critic": bool(manifest.get("critic")),
                "profiler": bool(manifest.get("profiler_feedback")),
                "turns": summary.get("samples", len(samples)),
                "pass_count": len(passed),
                "pass_count_harness": summary.get("pass_count", len(passed)),
                "spurious_passes": len(spurious),
                "solved": bool(passed),
                "turns_to_first_pass": first,
                # Recomputed rather than taken from summary.json, which counts a
                # spurious PASS as the first pass.
                "tokens_to_first_pass": (
                    sum(int(s.get("total_tokens", 0)) for s in samples if int(s["turn"]) <= first)
                    if first is not None
                    else None
                ),
                "best_runtime_ms": float(best["runtime_ms"]) if best else None,
                "best_speedup": float(best["speedup"]) if best else None,
                "baseline_kernel_time_ms": reference,
                "documentation_tokens_cl100k": manifest.get("documentation_tokens_cl100k"),
                **{field: summary.get(field, 0) for field in TOKEN_FIELDS},
                "critic_tokens": summary.get("critic_tokens", 0),
                "critic_hints": summary.get("critic_hints", 0),
                "agent_evaluations": sum(
                    int(sample.get("agent_evaluations", 0)) for sample in samples
                ),
                "agent_evaluations_passed": sum(
                    int(sample.get("agent_evaluations_passed", 0)) for sample in samples
                ),
                "session_errors": sum(
                    1 for sample in samples if str(sample.get("session_error", ""))
                ),
                "wrote_candidate_failures": sum(
                    1 for sample in samples if not sample.get("wrote_candidate", True)
                ),
                "author_seconds": summary.get("author_seconds"),
                "per_turn": [
                    {
                        "turn": sample.get("turn"),
                        "passed": bool(sample.get("passed")),
                        "runtime_ms": sample.get("runtime_ms"),
                        "speedup": sample.get("speedup"),
                        "agent_evaluations": sample.get("agent_evaluations", 0),
                        "input_tokens": sample.get("input_tokens", 0),
                        "output_tokens": sample.get("output_tokens", 0),
                        "reasoning_tokens": sample.get("reasoning_tokens", 0),
                        "critic_tokens": sample.get("critic_tokens", 0),
                        "session_error": sample.get("session_error", ""),
                    }
                    for sample in samples
                ],
            }
        )
    return rows


def tokens(value: Any) -> str:
    """Zero cached tokens means the provider did not report caching, not zero cache."""
    return f"{int(value):,}" if value else "n/r"


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Arm | Knob | Solved | Turns→1st pass | Best ms | Baseline ms | Speedup | "
        "Input | Cached | Output | Reasoning | Tokens→1st pass | Device evals |",
        "| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        knob = row["tier"]
        if row["experiment"] != "E1":
            extra = []
            if row["delivery"] != "files":
                extra.append(f"delivery={row['delivery']}")
            if row["critic"]:
                extra.append("critic=on")
            if row["profiler"]:
                extra.append("profiler=on")
            knob = f"{row['tier']} + {', '.join(extra)}" if extra else row["tier"]
        speedup = f"{row['best_speedup']:.2f}×" if row["best_speedup"] else "—"
        best = f"{row['best_runtime_ms']:.4f}" if row["best_runtime_ms"] else "—"
        reference = (
            f"{row['baseline_kernel_time_ms']:.4f}" if row["baseline_kernel_time_ms"] else "—"
        )
        first = row["turns_to_first_pass"] or "—"
        to_first = f"{int(row['tokens_to_first_pass']):,}" if row["tokens_to_first_pass"] else "—"
        lines.append(
            f"| {row['experiment']} | {knob} | "
            f"{row['pass_count']}/{row['turns']} | {first} | {best} | {reference} | {speedup} | "
            f"{int(row['input_tokens']):,} | {tokens(row['cached_input_tokens'])} | "
            f"{int(row['output_tokens']):,} | {int(row['reasoning_tokens']):,} | "
            f"{to_first} | {row['agent_evaluations']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        help="LABEL:path pairs, e.g. E1:results/e1-l2-63",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.roots:
        label, _, path = spec.partition(":")
        rows += arms(label, Path(path))

    print(render(rows))
    print()
    print(f"arms accounted for: {len(rows)}")
    for row in rows:
        missing = [field for field in ("input_tokens", "output_tokens") if not row[field]]
        if missing:
            print(f"  WARNING {row['experiment']}/{row['tier']}: zero {', '.join(missing)}")
        if row["spurious_passes"]:
            print(
                f"  WARNING {row['experiment']}/{row['tier']}: {row['spurious_passes']} harness PASS(es) "
                "launched no candidate kernel and are NOT counted as solves"
            )
        if row["session_errors"]:
            print(f"  WARNING {row['experiment']}/{row['tier']}: {row['session_errors']} session_error turns")
        if row["wrote_candidate_failures"]:
            print(
                f"  WARNING {row['experiment']}/{row['tier']}: "
                f"{row['wrote_candidate_failures']} turns wrote no candidate"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
