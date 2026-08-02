from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import yaml


TIER_LABELS = {
    "bare": "Bare",
    "docs": "Documentation",
    "examples": "Examples",
    "errors": "Error hints",
}
TIER_ORDER = tuple(TIER_LABELS)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2))
    return center - radius / denominator, center + radius / denominator


def geomean_interval(values: list[float]) -> tuple[float, float, float]:
    logs = [math.log(value) for value in values]
    mean = statistics.fmean(logs)
    half_width = 1.96 * statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    return math.exp(mean), math.exp(mean - half_width), math.exp(mean + half_width)


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in root.rglob("summary.json"):
        if path.with_name("EXCLUDED.json").exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.with_name("manifest.json")
        if summary.get("samples") is None or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs.append({**summary, **manifest, "run_dir": str(path.parent)})
    return runs


def aggregate(runs: list[dict[str, Any]], *, by_task: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        task = run["task"] if by_task else "all"
        groups[(run["approach"], run["tier"], task)].append(run)

    rows = []
    for (approach, tier, task), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][2] if by_task else "",
            TIER_ORDER.index(item[0][1]),
        ),
    ):
        candidates = sum(int(run["samples"]) for run in group)
        samples = sum(int(run.get("primary_requests", run["samples"])) for run in group)
        passed = sum(int(run["pass_count"]) for run in group)
        accelerated = sum(int(run.get("accelerated_count", 0)) for run in group)
        independent_units = sum(int(run.get("chains", 1)) for run in group)
        solved_units = sum(int(run.get("chains_solved", float(run["best_speedup"]) > 0)) for run in group)
        floor_speedups = [
            float(value) for run in group for value in run.get("floor_speedups", [run["floor_1_speedup"]])
        ]
        floor_speedups += [1.0] * (samples - candidates)
        run_pass_rates = [
            int(run["pass_count"]) / int(run.get("primary_requests", run["samples"]))
            if int(run.get("primary_requests", run["samples"]))
            else 0.0
            for run in group
        ]
        solved_speedups = [float(run["best_speedup"]) for run in group if int(run["pass_count"])]
        geomean, speedup_low, speedup_high = geomean_interval(floor_speedups)
        rows.append(
            {
                "approach": approach,
                "tier": tier,
                "task": task,
                "runs": len(group),
                "tasks": len({run["task"] for run in group}),
                "samples": samples,
                "candidate_count": candidates,
                "pass_count": passed,
                "invalid_count": samples - passed,
                "author_failure_count": samples - candidates,
                "pass_rate": passed / samples if samples else 0.0,
                "accelerated_rate": accelerated / samples if samples else 0.0,
                "independent_units": independent_units,
                "solved_rate": solved_units / independent_units,
                "solved_rate_ci95": wilson_interval(solved_units, independent_units),
                "geomean_floor_1_speedup": geomean,
                "geomean_floor_1_speedup_ci95": [speedup_low, speedup_high],
                "mean_documentation_tokens_cl100k": statistics.fmean(
                    int(run.get("documentation_tokens_cl100k", 0)) for run in group
                ),
                "llm_requests": sum(int(run.get("llm_requests", 0)) for run in group),
                "repair_requests": sum(int(run.get("repair_requests", 0)) for run in group),
                "output_cap_hits": sum(int(run.get("output_cap_hits", 0)) for run in group),
                "missing_jit_entrypoint": sum(int(run.get("missing_jit_entrypoint", 0)) for run in group),
                "no_remote_result": sum(int(run.get("no_remote_result", 0)) for run in group),
                "best_speedup": max(float(run["best_speedup"]) for run in group),
                "run_pass_rate_mean": statistics.fmean(run_pass_rates),
                "run_pass_rate_sd": (statistics.stdev(run_pass_rates) if len(run_pass_rates) > 1 else 0.0),
                "run_best_speedup_mean": statistics.fmean(solved_speedups) if solved_speedups else 0.0,
                "run_best_speedup_sd": (statistics.stdev(solved_speedups) if len(solved_speedups) > 1 else 0.0),
                "input_tokens": sum(int(run.get("input_tokens", 0)) for run in group),
                "cached_input_tokens": sum(int(run.get("cached_input_tokens", 0)) for run in group),
                "output_tokens": sum(int(run.get("output_tokens", 0)) for run in group),
                "reasoning_tokens": sum(int(run.get("reasoning_tokens", 0)) for run in group),
                "visible_output_tokens": sum(int(run.get("visible_output_tokens", 0)) for run in group),
                "total_tokens": sum(int(run.get("total_tokens", 0)) for run in group),
                "input_tokens_per_sample": sum(int(run.get("input_tokens", 0)) for run in group) / samples
                if samples
                else 0.0,
                "output_tokens_per_sample": sum(int(run.get("output_tokens", 0)) for run in group) / samples
                if samples
                else 0.0,
                "total_tokens_per_sample": sum(int(run.get("total_tokens", 0)) for run in group) / samples
                if samples
                else 0.0,
                "llm_requests_per_sample": sum(int(run.get("llm_requests", 0)) for run in group) / samples
                if samples
                else 0.0,
            }
        )
    baselines = {(row["approach"], row["task"]): row for row in rows if row["tier"] == "bare"}
    for row in rows:
        baseline = baselines.get((row["approach"], row["task"]))
        if baseline:
            row["pass_rate_delta_vs_bare"] = row["pass_rate"] - baseline["pass_rate"]
            row["speedup_ratio_vs_bare"] = row["geomean_floor_1_speedup"] / baseline["geomean_floor_1_speedup"]
            row["output_tokens_ratio_vs_bare"] = row["output_tokens_per_sample"] / baseline["output_tokens_per_sample"]
            row["total_tokens_ratio_vs_bare"] = row["total_tokens_per_sample"] / baseline["total_tokens_per_sample"]
    return rows


def write_report(
    output_dir: Path,
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    study: dict[str, Any],
    excluded_runs: int,
) -> None:
    lines = [
        "# CuTe documentation ablation",
        "",
        "## Main results",
    ]
    for approach in dict.fromkeys(row["approach"] for row in rows):
        settings = study.get(approach, {})
        lines += [
            "",
            f"### {approach.title()}",
        ]
        if approach == "iter" and settings:
            evaluator = study["evaluator"]
            from_scratch = settings["b300_seed"] == "starter"
            lines += [
                "",
                f"- Model: `{study['model']}` driven by OpenCode.",
                f"- Matrix: 1 task × {len(study['tiers'])} cumulative tiers × "
                f"{settings['replications']} independent runs = "
                f"{len(study['tiers']) * settings['replications']} runs.",
                f"- Task: `{study['task']}`.",
                f"- Per run: {settings['islands']} island × {settings['steps']} barrier "
                f"iterations = {settings['steps']} author sessions.",
                "- Protocol: documentation delivered by "
                f"`{(study.get('documentation') or {}).get('delivery', 'files')}`; "
                "profiler feedback "
                f"{'on' if (study.get('profiling') or {}).get('enabled') else 'off'}; "
                f"critic {'on' if (study.get('feedback') or {}).get('critic') else 'off'}.",
                f"- Start: `{settings['b300_seed']}` — "
                + (
                    "the public skeleton, so the model writes the kernel. The verified "
                    "reference is timed separately as the speedup denominator and is "
                    "never shown to the model."
                    if from_scratch
                    else "the verified reference candidate, so the model only modifies it."
                ),
                f"- Parallelism: {settings['concurrency']} OpenCode author streams; "
                "one serialized B300 evaluation stream.",
                f"- Evaluation: B300 device time; seed {evaluator['seed']}; "
                f"{evaluator['warmup']} warmups; {evaluator['repeats']} timed repetitions; "
                f"{evaluator['timeout_seconds']}-second timeout.",
            ]
            wall_seconds = study.get("_environment", {}).get("wall_seconds")
            if wall_seconds is not None:
                lines.append(f"- Observed wall time: {float(wall_seconds) / 3600:.2f} hours.")
        lines += [
            "",
            "| Tier | Runs | Pass rate | Speed up (best raw) | Input tokens | Output tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in (row for row in rows if row["approach"] == approach):
            speedup = f"{row['best_speedup']:.4f}×" if row["pass_count"] else "—"
            lines.append(
                f"| {TIER_LABELS[row['tier']]} | {row['runs']} | "
                f"{row['pass_count']}/{row['samples']} ({row['pass_rate']:.1%}) | "
                f"{speedup} | {row['input_tokens']:,} | {row['output_tokens']:,} |"
            )
    lines += [
        "",
        "Pass rate is over primary author calls; responses that do not produce a candidate count "
        "as failures. Best raw speedup is reference time divided by "
        "candidate time for the fastest passing mutation; it is not a stability-confirmed result.",
        "Token columns are provider-reported totals across all runs, including repair calls.",
        "",
        "## Stability across independent runs",
        "",
        "| Approach | Tier | Runs | Pass rate/run (mean ± SD) | Solved runs | Best raw speedup/run (mean ± SD) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = (
            f"{row['run_best_speedup_mean']:.4f}× ± {row['run_best_speedup_sd']:.4f}×" if row["solved_rate"] else "—"
        )
        lines.append(
            f"| {row['approach']} | {TIER_LABELS[row['tier']]} | {row['runs']} | "
            f"{row['run_pass_rate_mean']:.1%} ± {row['run_pass_rate_sd']:.1%} | "
            f"{row['solved_rate'] * row['independent_units']:.0f}/"
            f"{row['independent_units']} | {speedup} |"
        )
    lines += [
        "",
        "## Cost",
        "",
        "| Approach | Tier | Static docs¹ | Requests/attempt | Repair calls | Cap hits | "
        "Input/attempt | Cached/attempt | Output/attempt | Total/attempt |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['approach']} | {row['tier']} | "
            f"{row['mean_documentation_tokens_cl100k']:.0f} | "
            f"{row['llm_requests_per_sample']:.2f} | {row['repair_requests']} | "
            f"{row['output_cap_hits']} | {row['input_tokens_per_sample']:.0f} | "
            f"{row['cached_input_tokens'] / row['samples']:.0f} | "
            f"{row['output_tokens_per_sample']:.0f} | {row['total_tokens_per_sample']:.0f} |"
        )
    lines += [
        "",
        "¹ Static context is a cl100k estimate; provider usage columns are authoritative.",
        "",
        "## Frequent failure",
        "",
        "| Approach | Tier | Invalid attempts | No candidate | Missing jitted entry point | No remote result |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['approach']} | {row['tier']} | {row['invalid_count']} | "
            f"{row['author_failure_count']} | "
            f"{row['missing_jit_entrypoint']} | {row['no_remote_result']} |"
        )
    lines += [
        "",
        "## Change from bare",
        "",
        "| Approach | Tier | Pass change | Speedup ratio | Output change | Total change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "pass_rate_delta_vs_bare" not in row:
            continue
        lines.append(
            f"| {row['approach']} | {row['tier']} | "
            f"{row['pass_rate_delta_vs_bare'] * 100:+.1f} pp | "
            f"{row['speedup_ratio_vs_bare']:.3f}x | "
            f"{(row['output_tokens_ratio_vs_bare'] - 1) * 100:+.1f}% | "
            f"{(row['total_tokens_ratio_vs_bare'] - 1) * 100:+.1f}% |"
        )
    lines += [
        "",
        "## By task",
        "",
        "| Approach | Tier | Task | Attempts | Pass | Best | Geomean speedup¹ | Output/attempt |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['approach']} | {row['tier']} | {row['task']} | {row['samples']} | "
            f"{row['pass_rate']:.1%} | {row['best_speedup']:.4f}x | "
            f"{row['geomean_floor_1_speedup']:.3f}x | "
            f"{row['output_tokens_per_sample']:.0f} |"
        )
    lines += [
        "",
        f"Runs included: {len(runs)}.",
    ]
    if excluded_runs:
        lines.append(f"Transport-invalid runs excluded and replaced: {excluded_runs}.")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "aggregate.json").write_text(
        json.dumps({"overall": rows, "by_task": task_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.root)
    study_path = args.root / "study.yaml"
    study = yaml.safe_load(study_path.read_text(encoding="utf-8")) if study_path.exists() else {}
    environment_path = args.root / "environment.json"
    if environment_path.exists():
        study["_environment"] = json.loads(environment_path.read_text(encoding="utf-8"))
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    write_report(
        output,
        runs,
        aggregate(runs),
        aggregate(runs, by_task=True),
        study,
        sum(1 for _ in args.root.rglob("EXCLUDED.json")),
    )


if __name__ == "__main__":
    main()
