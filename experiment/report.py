from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


TABLE_FIELDS = (
    ("model", "Model"),
    ("task", "Task"),
    ("status", "Status"),
    ("baseline_ms", "Baseline ms"),
    ("agent_ms", "Agent ms"),
    ("speedup", "Speedup"),
    ("input_uncached", "Input"),
    ("input_cached", "Cache input"),
    ("output_tokens", "Output"),
    ("agent_wall_seconds", "Agent s"),
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


def render_table(rows: list[dict[str, Any]]) -> str:
    rendered = [
        [_display(field, row.get(field)) for field, _ in TABLE_FIELDS]
        for row in rows
    ]
    headers = [header for _, header in TABLE_FIELDS]
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


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [field for field, _ in TABLE_FIELDS]
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    (output_dir / "results.txt").write_text(
        render_table(rows) + "\n",
        encoding="utf-8",
    )
