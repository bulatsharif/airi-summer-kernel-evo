from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubmissionSpec:
    index: int
    task_id: str
    path: Path

    @property
    def name(self) -> str:
        return f"{self.index:03d}:{self.path.name}"


TABLE_FIELDS = (
    ("candidate", "Candidate"),
    ("task_id", "Task"),
    ("status", "Status"),
    ("baseline_ms", "Baseline ms"),
    ("candidate_ms", "Candidate ms"),
    ("speedup", "Speedup"),
)


def parse_submission_specs(values: list[str]) -> list[SubmissionSpec]:
    specs: list[SubmissionSpec] = []
    for index, value in enumerate(values, start=1):
        task_id, separator, raw_path = value.partition("=")
        task_id = task_id.strip()
        raw_path = raw_path.strip()
        if not separator or not task_id or not raw_path:
            raise ValueError(
                "submission must use TASK_ID=PATH format, "
                f"got: {value!r}"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"submission file does not exist: {path}")
        specs.append(SubmissionSpec(index, task_id, path))
    return specs


def comparison_row(
    spec: SubmissionSpec,
    *,
    status: str,
    baseline_ms: float | None,
    candidate_ms: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    speedup = None
    if (
        status == "PASS"
        and baseline_ms is not None
        and candidate_ms is not None
        and candidate_ms > 0
    ):
        speedup = baseline_ms / candidate_ms
    return {
        "candidate": spec.name,
        "task_id": spec.task_id,
        "submission": str(spec.path),
        "status": status,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": speedup,
        "error": error,
    }


def _display(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field in {"baseline_ms", "candidate_ms"}:
        return f"{float(value):.6f}"
    if field == "speedup":
        return f"{float(value):.3f}x"
    return str(value)


def render_comparison_table(rows: list[dict[str, Any]]) -> str:
    headers = [header for _, header in TABLE_FIELDS]
    rendered = [
        [_display(field, row.get(field)) for field, _ in TABLE_FIELDS]
        for row in rows
    ]
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


def write_comparison_reports(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = dict(metadata)
    payload["rows"] = rows
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "candidate",
        "task_id",
        "submission",
        "status",
        "baseline_ms",
        "candidate_ms",
        "speedup",
        "error",
    ]
    with (output_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    (output_dir / "comparison.txt").write_text(
        render_comparison_table(rows) + "\n",
        encoding="utf-8",
    )
