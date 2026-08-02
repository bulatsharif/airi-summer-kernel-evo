"""Frozen cumulative context levels for the CuTe documentation study."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics

import tiktoken

from kernel_evo.cute_harness.b300 import (
    EvaluationConfig,
    TaskSpec,
    baseline_candidate,
    evaluate,
)


DOCUMENTATION_TIERS = ("bare", "docs", "examples", "errors")
# How a tier reaches the author: `files` materializes it and lets the agent choose
# what to open; `prompt` injects the whole bundle into the authoring session.
DOCUMENTATION_DELIVERY = ("files", "prompt")
TIER_DIRECTORIES = (
    "tier-2-foundations",
    "tier-3-examples",
    "tier-4-errors",
)


@dataclass(frozen=True)
class DocumentationBundle:
    tier: str
    files: tuple[Path, ...]
    text: str
    tokens_cl100k: int


def documentation_bundle(task: TaskSpec, tier: str) -> DocumentationBundle:
    if tier not in DOCUMENTATION_TIERS:
        raise ValueError(f"documentation tier must be one of: {', '.join(DOCUMENTATION_TIERS)}")

    skill = task.skill_paths[0]
    level = DOCUMENTATION_TIERS.index(tier)
    files = [task.prompt_path]
    for directory in TIER_DIRECTORIES[:level]:
        files.extend(sorted((skill / "tiers" / directory).glob("*.md")))

    bundle_files = tuple(path.resolve() for path in files)
    text = "\n\n".join(f"# File: {path.name}\n\n{path.read_text(encoding='utf-8')}" for path in bundle_files)
    tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
    return DocumentationBundle(tier, bundle_files, text.rstrip() + "\n", tokens)


def materialize_bundle(task: TaskSpec, tier: str, destination: Path) -> DocumentationBundle:
    """Copy one tier's files into ``destination`` and rebase the bundle onto the copies.

    An author that reads only materialized paths cannot reach documentation above
    its tier, so the shared skill directory can be denied outright instead of
    relying on the packet listing to bound what is reachable. Copies are
    content-compared so repeated preparation of the same barrier is idempotent.
    """
    bundle = documentation_bundle(task, tier)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    used: set[str] = set()
    for source in bundle.files:
        name = source.name
        if name in used:
            name = f"{source.parent.name}-{source.name}"
        used.add(name)
        target = destination / name
        payload = source.read_bytes()
        if not target.is_file() or target.read_bytes() != payload:
            target.write_bytes(payload)
        copied.append(target.resolve())
    return DocumentationBundle(bundle.tier, tuple(copied), bundle.text, bundle.tokens_cl100k)


def timing_summary(times_ms: list[float], *, max_cv: float = 0.02) -> dict[str, float | int | bool]:
    mean = statistics.fmean(times_ms)
    std = statistics.stdev(times_ms)
    median = statistics.median(times_ms)
    mad = statistics.median(abs(value - median) for value in times_ms)
    return {
        "runs": len(times_ms),
        "mean_ms": mean,
        "std_ms": std,
        "variance_ms2": statistics.variance(times_ms),
        "cv": std / mean,
        "median_ms": median,
        "mad_ms": mad,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "ci95_half_width_ms": 1.96 * std / len(times_ms) ** 0.5,
        "max_cv": max_cv,
        "stable": std / mean <= max_cv,
    }


def run_stability(
    task: TaskSpec,
    output_dir: Path,
    *,
    runs: int,
    config: EvaluationConfig,
    candidate: Path | None = None,
    max_cv: float = 0.02,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = candidate or output_dir / "baseline.py"
    if candidate is None:
        candidate_path.write_text(baseline_candidate(task), encoding="utf-8")

    times = []
    for index in range(runs):
        trial_dir = output_dir / f"trial-{index + 1:03d}"
        record = evaluate(
            task,
            candidate_path,
            trial_dir,
            config,
        )
        if not record["passed"]:
            raise RuntimeError(f"stability trial {index + 1} failed; see {trial_dir / 'result.json'}")
        times.append(float(record["kernel_time_ms"]))

    summary: dict[str, object] = {
        "task": task.id,
        "candidate": str(candidate_path.resolve()),
        "warmup": config.warmup,
        "repeats": config.repeats,
        "times_ms": times,
        **timing_summary(times, max_cv=max_cv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_stability_markdown(summary), encoding="utf-8")
    return summary


def _stability_markdown(summary: dict[str, object]) -> str:
    verdict = "stable" if summary["stable"] else "unstable"
    return f"""# B300 evaluator stability

Task: `{summary["task"]}`

Protocol: {summary["runs"]} independent submissions, {summary["warmup"]} warmups and
{summary["repeats"]} timed CUDA-event repetitions per submission; each submission reports its median.

| Metric | Value |
| --- | ---: |
| Mean | {summary["mean_ms"]:.6f} ms |
| Standard deviation | {summary["std_ms"]:.6f} ms |
| Variance | {summary["variance_ms2"]:.9f} ms² |
| Coefficient of variation | {summary["cv"]:.3%} |
| Median | {summary["median_ms"]:.6f} ms |
| Median absolute deviation | {summary["mad_ms"]:.6f} ms |
| Range | {summary["min_ms"]:.6f}–{summary["max_ms"]:.6f} ms |
| 95% CI half-width for mean | {summary["ci95_half_width_ms"]:.6f} ms |

Verdict: **{verdict}** under the predeclared CV threshold of {summary["max_cv"]:.1%}.
"""
