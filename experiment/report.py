from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any, Callable


TABLE_FIELDS = (
    ("model", "Model"),
    ("task", "Task"),
    ("attempt", "Attempt"),
    ("status", "Status"),
    ("baseline_ms", "Baseline ms"),
    ("agent_ms", "Agent ms"),
    ("speedup", "Speedup"),
    ("input_uncached", "Input"),
    ("input_cached", "Cache input"),
    ("output_tokens", "Output"),
    ("agent_wall_seconds", "Agent s"),
)

SUMMARY_FIELDS = (
    ("model", "Model"),
    ("task", "Task"),
    ("attempts", "Attempts"),
    ("pass_count", "PASS"),
    ("fail_count", "FAIL"),
    ("timeout_count", "TIMEOUT"),
    ("baseline_fail_count", "Baseline fail"),
    ("pass_rate", "Pass rate"),
    ("speedup_n", "Speedup n"),
    ("speedup_mean", "Speedup mean"),
    ("speedup_std", "Speedup std"),
    ("speedup_median", "Speedup median"),
    ("input_uncached_n", "Input n"),
    ("input_uncached_mean", "Input mean"),
    ("input_uncached_std", "Input std"),
    ("input_uncached_median", "Input median"),
    ("input_cached_n", "Cache n"),
    ("input_cached_mean", "Cache mean"),
    ("input_cached_std", "Cache std"),
    ("input_cached_median", "Cache median"),
    ("output_tokens_n", "Output n"),
    ("output_tokens_mean", "Output mean"),
    ("output_tokens_std", "Output std"),
    ("output_tokens_median", "Output median"),
    ("agent_wall_seconds_n", "Agent s n"),
    ("agent_wall_seconds_mean", "Agent s mean"),
    ("agent_wall_seconds_std", "Agent s std"),
    ("agent_wall_seconds_median", "Agent s median"),
)

SUMMARY_METRICS = (
    "speedup",
    "input_uncached",
    "input_cached",
    "output_tokens",
    "agent_wall_seconds",
)


def _display(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field in {"baseline_ms", "agent_ms"}:
        return f"{float(value):.4f}"
    if field == "speedup":
        return f"{float(value):.3f}x"
    if field == "agent_wall_seconds":
        return f"{float(value):.1f}"
    if field in {"input_uncached", "input_cached", "output_tokens"}:
        return f"{int(value):,}"
    return str(value)


def _summary_display(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field.endswith("_n"):
        return str(int(value))
    if field == "pass_rate":
        return f"{float(value):.1%}"
    if field.startswith("speedup_"):
        return f"{float(value):.3f}x"
    if field.startswith(("input_", "output_tokens_")):
        return f"{float(value):,.1f}"
    if field.startswith("agent_wall_seconds_"):
        return f"{float(value):.1f}"
    return str(value)


def _render_table(
    rows: list[dict[str, Any]],
    fields: tuple[tuple[str, str], ...],
    display: Callable[[str, Any], str],
) -> str:
    rendered = [
        [display(field, row.get(field)) for field, _ in fields]
        for row in rows
    ]
    headers = [header for _, header in fields]
    widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in rendered),
        )
        for index in range(len(headers))
    ]
    header_line = " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )
        for row in rendered
    ]
    return "\n".join((header_line, separator, *body))


def render_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(rows, TABLE_FIELDS, _display)


def render_summary_table(rows: list[dict[str, Any]]) -> str:
    return _render_table(rows, SUMMARY_FIELDS, _summary_display)


def _numeric_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _metric_summary(values: list[float]) -> dict[str, int | float | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) >= 2 else None,
        "median": statistics.median(values) if values else None,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("model"), row.get("task"))
        groups.setdefault(key, []).append(row)

    summaries = []
    for (model, task), group in groups.items():
        attempts = [row for row in group if row.get("attempt") is not None]
        pass_count = sum(row.get("status") == "PASS" for row in attempts)
        summary: dict[str, Any] = {
            "model": model,
            "task": task,
            "attempts": len(attempts),
            "pass_count": pass_count,
            "fail_count": sum(
                row.get("status") == "FAIL" for row in attempts
            ),
            "timeout_count": sum(
                row.get("status") == "TIMEOUT" for row in attempts
            ),
            "baseline_fail_count": sum(
                row.get("status") == "BASELINE_FAIL" for row in group
            ),
            "pass_rate": (
                pass_count / len(attempts) if attempts else None
            ),
        }
        successful = [
            row for row in attempts if row.get("status") == "PASS"
        ]
        for metric in SUMMARY_METRICS:
            metric_rows = successful if metric == "speedup" else attempts
            metric_summary = _metric_summary(
                _numeric_values(metric_rows, metric)
            )
            for statistic, value in metric_summary.items():
                summary[f"{metric}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[tuple[str, str], ...],
) -> None:
    fieldnames = [field for field, _ in fields]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", rows, TABLE_FIELDS)
    (output_dir / "results.txt").write_text(
        render_table(rows) + "\n",
        encoding="utf-8",
    )

    summaries = summarize_rows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    (output_dir / "summary.txt").write_text(
        render_summary_table(summaries) + "\n",
        encoding="utf-8",
    )
