"""Deterministic summaries of the B300 harness PyTorch Chrome trace.

The remote harness returns a PyTorch profiler trace per run. Only the GPU-side
slices are actionable for an author. The legacy summary aggregates them by
kernel name; the optional timeline retains every GPU activity and every idle
hole while omitting CPU/Python profiler bookkeeping. `ts` and `dur` are
microseconds even though `displayTimeUnit` is "ms".
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
NAME_WIDTH = 88


def summarize_chrome_trace(trace: dict, *, top_k: int = 10) -> dict[str, Any]:
    """Aggregate GPU-side trace events by kernel name, hottest first."""
    if not isinstance(trace, Mapping):
        raise ValueError("Chrome trace must be a JSON object")
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace has no traceEvents list")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    category_ms = dict.fromkeys(GPU_CATEGORIES, 0.0)
    category_calls = dict.fromkeys(GPU_CATEGORIES, 0)
    per_kernel: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping) or event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category not in category_ms:
            continue
        duration = event.get("dur")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        milliseconds = float(duration) / 1000.0
        category_ms[category] += milliseconds
        category_calls[category] += 1
        if category != "kernel":
            continue
        entry = per_kernel.setdefault(
            str(event.get("name", "")), {"total_ms": 0.0, "calls": 0}
        )
        entry["total_ms"] += milliseconds
        entry["calls"] += 1

    gpu_time_ms = sum(category_ms.values())
    ranked = sorted(per_kernel.items(), key=lambda item: (-item[1]["total_ms"], item[0]))
    return {
        "device": _device(trace),
        "gpu_time_ms": round(gpu_time_ms, 6),
        "kernel_time_ms": round(category_ms["kernel"], 6),
        "kernel_calls": category_calls["kernel"],
        "memcpy_time_ms": round(category_ms["gpu_memcpy"], 6),
        "memcpy_calls": category_calls["gpu_memcpy"],
        "memset_time_ms": round(category_ms["gpu_memset"], 6),
        "memset_calls": category_calls["gpu_memset"],
        "distinct_kernels": len(per_kernel),
        "top_kernels": [
            {
                "name": name,
                "total_ms": round(entry["total_ms"], 6),
                "calls": entry["calls"],
                "pct_gpu_time": (
                    round(100.0 * entry["total_ms"] / gpu_time_ms, 2) if gpu_time_ms else 0.0
                ),
            }
            for name, entry in ranked[:top_k]
        ],
    }


def summarize_chrome_timeline(
    trace: dict,
    *,
    candidate_kernel_symbols: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Retain every ordered GPU activity and measure device-wide idle holes.

    Candidate attribution is deliberately a hint rather than a claim: a GPU
    kernel is marked as candidate-authored only when its generated name contains
    one of the candidate source's ``@cute.kernel`` function names.
    """
    if not isinstance(trace, Mapping):
        raise ValueError("Chrome trace must be a JSON object")
    raw_events = trace.get("traceEvents")
    if not isinstance(raw_events, list):
        raise ValueError("Chrome trace has no traceEvents list")

    symbols = tuple(sorted({str(item) for item in candidate_kernel_symbols if str(item)}))
    ordered: list[tuple[float, int, Mapping[str, Any]]] = []
    for ordinal, event in enumerate(raw_events):
        if (
            not isinstance(event, Mapping)
            or event.get("ph") != "X"
            or event.get("cat") not in GPU_CATEGORIES
        ):
            continue
        timestamp = event.get("ts")
        duration = event.get("dur")
        if (
            isinstance(timestamp, bool)
            or isinstance(duration, bool)
            or not isinstance(timestamp, (int, float))
            or not isinstance(duration, (int, float))
            or duration < 0
        ):
            continue
        ordered.append((float(timestamp), ordinal, event))
    ordered.sort(key=lambda item: (item[0], item[1]))

    if not ordered:
        return {
            "device": _device(trace),
            "event_count": 0,
            "timeline_span_ms": 0.0,
            "activity_time_ms": 0.0,
            "busy_time_ms": 0.0,
            "idle_time_ms": 0.0,
            "idle_pct": 0.0,
            "largest_gap_ms": 0.0,
            "candidate_kernel_symbols": list(symbols),
            "observed_candidate_symbols": [],
            "unobserved_candidate_symbols": list(symbols),
            "kernels": [],
            "events": [],
        }

    origin_us = ordered[0][0]
    busy_until_us = origin_us
    idle_time_us = 0.0
    largest_gap_us = 0.0
    activity_time_us = 0.0
    aliases: dict[tuple[Any, ...], dict[str, Any]] = {}
    observed_symbols: set[str] = set()
    timeline_events: list[dict[str, Any]] = []

    for index, (timestamp, _ordinal, event) in enumerate(ordered, start=1):
        duration_us = float(event["dur"])
        gap_us = max(0.0, timestamp - busy_until_us)
        idle_time_us += gap_us
        largest_gap_us = max(largest_gap_us, gap_us)
        busy_until_us = max(busy_until_us, timestamp + duration_us)
        activity_time_us += duration_us

        category = str(event["cat"])
        name = str(event.get("name", ""))
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        stream = args.get("stream", event.get("tid"))
        device = args.get("device")
        item: dict[str, Any] = {
            "index": index,
            "start_ms": round((timestamp - origin_us) / 1000.0, 6),
            "gap_before_us": round(gap_us, 6),
            "duration_us": round(duration_us, 6),
            "kind": _kind(category),
            "device": device,
            "stream": stream,
        }

        if category == "kernel":
            symbol = _matching_symbol(name, symbols)
            if symbol:
                observed_symbols.add(symbol)
            source_hint = (
                f"candidate:{symbol}"
                if symbol
                else "CuTe helper/setup"
                if name.startswith("kernel_cutlass_")
                else "framework/validation"
            )
            launch = _launch(args)
            alias_key = (
                name,
                source_hint,
                tuple(launch.get("grid", ())),
                tuple(launch.get("block", ())),
                launch.get("shared_memory_bytes"),
                launch.get("registers_per_thread"),
            )
            alias = aliases.get(alias_key)
            if alias is None:
                alias = {
                    "id": f"K{len(aliases) + 1}",
                    "name": name,
                    "source_hint": source_hint,
                    "calls": 0,
                    "total_us": 0.0,
                    **launch,
                }
                aliases[alias_key] = alias
            alias["calls"] += 1
            alias["total_us"] += duration_us
            item["activity"] = alias["id"]
        else:
            item["activity"] = name
            byte_count = args.get("bytes")
            if isinstance(byte_count, (int, float)) and not isinstance(byte_count, bool):
                item["bytes"] = int(byte_count)
        timeline_events.append(item)

    span_us = max(0.0, busy_until_us - origin_us)
    busy_time_us = max(0.0, span_us - idle_time_us)
    kernels = []
    for alias in aliases.values():
        kernels.append(
            {
                **alias,
                "total_us": round(float(alias["total_us"]), 6),
            }
        )
    observed = sorted(observed_symbols)
    return {
        "device": _device(trace),
        "event_count": len(timeline_events),
        "timeline_span_ms": round(span_us / 1000.0, 6),
        "activity_time_ms": round(activity_time_us / 1000.0, 6),
        "busy_time_ms": round(busy_time_us / 1000.0, 6),
        "idle_time_ms": round(idle_time_us / 1000.0, 6),
        "idle_pct": round(100.0 * idle_time_us / span_us, 2) if span_us else 0.0,
        "largest_gap_ms": round(largest_gap_us / 1000.0, 6),
        "candidate_kernel_symbols": list(symbols),
        "observed_candidate_symbols": observed,
        "unobserved_candidate_symbols": sorted(set(symbols) - observed_symbols),
        "kernels": kernels,
        "events": timeline_events,
        "iteration": _iteration_timeline(timeline_events),
    }


def _iteration_timeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the whole-run trace to one representative iteration.

    The trace spans everything the harness did: input generation, JIT compile,
    the reference, warmup and every timed repeat. Its whole-run idle figure is
    therefore dominated by setup -- on a real capture, 4894 ms idle across a
    4922 ms span, 99.4%, against 0.5% idle inside the block itself. Reporting
    only the whole-run number tells an author its GPU is starved when the
    measured block is back to back, so reduce to the repeating part and give
    the two figures separately.

    Segmentation anchors on an activity name, not on a gap threshold: gap size
    is what is being measured here and so cannot also be the premise. A kernel
    launched once per repeat appears exactly `repeats` times, so each of its
    occurrences opens an iteration. The representative is the median iteration
    by wall span -- the early ones are warmup and the extremes are scheduling
    noise, so a median row set reproduces across reruns where a first or a
    fastest one does not.
    """
    launches = [
        item
        for item in events
        if item.get("kind") == "kernel"
        and isinstance(item.get("start_ms"), (int, float))
        and isinstance(item.get("duration_us"), (int, float))
    ]
    if len(launches) < 2:
        return {}
    counts = Counter(str(item.get("activity", "")) for item in launches)
    # How many times the timed loop ran. Kernels inside the block share one
    # launch count and together account for more launches than anything else.
    # Group by launch count and take the group covering the most launches --
    # not the most common count, which one-off setup kernels win: a real
    # capture has eleven distinct names launched once against seven launched
    # fifty-five times, so "most common" selects 1 and finds no loop at all.
    coverage: Counter[int] = Counter()
    for count in counts.values():
        # A kernel launched once cannot belong to a repeating block, and there
        # are enough of them in a real capture to outweigh the loop on volume
        # alone: forty distinct setup kernels would beat five repeats of three.
        if count > 1:
            coverage[count] += count
    if not coverage:
        return {}
    per_iteration = max(coverage.items(), key=lambda item: (item[1], item[0]))[0]
    recurring = {name for name, count in counts.items() if count >= per_iteration}
    ordered = [item for item in launches if str(item.get("activity", "")) in recurring]
    anchor = next(
        (
            str(item.get("activity", ""))
            for item in ordered
            if counts[str(item.get("activity", ""))] == per_iteration
        ),
        "",
    )
    if not anchor:
        return {}

    blocks: list[list[dict[str, Any]]] = []
    for item in ordered:
        if str(item.get("activity", "")) == anchor:
            blocks.append([])
        if blocks:
            blocks[-1].append(item)
    width = Counter(len(block) for block in blocks).most_common(1)[0][0]
    whole = [block for block in blocks if len(block) == width]
    if len(whole) < 2:
        return {}

    def start_us(item: Mapping[str, Any]) -> float:
        return float(item["start_ms"]) * 1000.0

    def end_us(item: Mapping[str, Any]) -> float:
        return start_us(item) + float(item["duration_us"])

    chosen = sorted(
        whole, key=lambda block: (end_us(block[-1]) - start_us(block[0]), start_us(block[0]))
    )[len(whole) // 2]
    origin = start_us(chosen[0])

    rows: list[dict[str, Any]] = []
    previous_end: float | None = None
    for item in chosen:
        begin = start_us(item)
        rows.append(
            {
                "activity": str(item.get("activity", "")),
                "start_us": round(begin - origin, 3),
                "end_us": round(end_us(item) - origin, 3),
                "duration_us": round(float(item["duration_us"]), 3),
                "gap_before_us": (
                    None if previous_end is None else round(begin - previous_end, 3)
                ),
            }
        )
        previous_end = end_us(item)

    busy = sum(row["duration_us"] for row in rows)
    span = rows[-1]["end_us"]
    inner = [row for row in rows if row["gap_before_us"] is not None]
    worst = max(inner, key=lambda row: row["gap_before_us"], default=None)
    # The gap in front of the block is host dispatch between repeats, not a
    # bubble an author can close by fusing. Keep it out of the in-block total
    # so the two can never be added together by mistake.
    position = whole.index(chosen)
    between = (
        round(origin - end_us(whole[position - 1][-1]), 3) if position > 0 else None
    )
    return {
        "iterations": len(whole),
        "kernels_per_iteration": width,
        "span_us": round(span, 3),
        "busy_us": round(busy, 3),
        "gap_us": round(span - busy, 3),
        "gap_pct": round(100.0 * (span - busy) / span, 2) if span > 0 else 0.0,
        "largest_gap_us": worst["gap_before_us"] if worst else 0.0,
        "largest_gap_before": worst["activity"] if worst else "",
        "between_iterations_us": between,
        "rows": rows,
    }


def trace_summary_markdown(summary: dict, *, max_chars: int = 2000) -> str:
    """Render one bounded markdown table; identical input yields identical text."""
    if not isinstance(summary, Mapping):
        raise ValueError("trace summary must be a JSON object")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    device = summary.get("device", {})
    lines = ["# B300 device profile of your last candidate", ""]
    if isinstance(device, Mapping) and device.get("name"):
        details = [str(device["name"])]
        if device.get("num_sms") is not None:
            details.append(f"{device['num_sms']} SMs")
        if device.get("shared_mem_per_block_optin") is not None:
            details.append(f"{device['shared_mem_per_block_optin']} B opt-in shared memory/block")
        lines.append(f"- device: {', '.join(details)}")
    lines.append(
        f"- GPU busy: {_ms(summary.get('gpu_time_ms'))} ms total; kernels "
        f"{_ms(summary.get('kernel_time_ms'))} ms over {_count(summary.get('kernel_calls'))} "
        f"launches of {_count(summary.get('distinct_kernels'))} distinct names"
    )
    lines.append(
        f"- memcpy: {_ms(summary.get('memcpy_time_ms'))} ms over "
        f"{_count(summary.get('memcpy_calls'))} calls; memset: "
        f"{_ms(summary.get('memset_time_ms'))} ms over {_count(summary.get('memset_calls'))} calls"
    )
    lines.extend(["", "| kernel | ms | calls | % GPU |", "| --- | ---: | ---: | ---: |"])
    text = "\n".join(lines) + "\n"

    kernels = summary.get("top_kernels", [])
    kernels = [item for item in kernels if isinstance(item, Mapping)] if isinstance(kernels, list) else []
    shown = 0
    for kernel in kernels:
        row = (
            f"| {_name(kernel.get('name', ''))} | {_ms(kernel.get('total_ms'))} "
            f"| {_count(kernel.get('calls'))} | {_pct(kernel.get('pct_gpu_time'))} |\n"
        )
        if len(text) + len(row) + len(_omitted(len(kernels) - shown - 1)) > max_chars:
            break
        text += row
        shown += 1
    return (text + _omitted(len(kernels) - shown))[:max_chars]


def trace_timeline_markdown(timeline: dict, *, aggregate: dict | None = None) -> str:
    """Render the complete GPU timeline without dropping or sampling activities."""
    if not isinstance(timeline, Mapping):
        raise ValueError("trace timeline must be a JSON object")
    events = timeline.get("events")
    kernels = timeline.get("kernels")
    if not isinstance(events, list) or not isinstance(kernels, list):
        raise ValueError("trace timeline has no events/kernels list")

    lines = [
        "# B300 complete GPU timeline of your last candidate",
        "",
        (
            f"- complete GPU-side trace: {_count(timeline.get('event_count'))} activities; "
            "CPU/Python profiler records are intentionally omitted"
        ),
        (
            f"- whole capture: span {_ms(timeline.get('timeline_span_ms'))} ms; active union "
            f"{_ms(timeline.get('busy_time_ms'))} ms; idle holes "
            f"{_ms(timeline.get('idle_time_ms'))} ms "
            f"({_pct(timeline.get('idle_pct'))}%); largest hole "
            f"{_ms(timeline.get('largest_gap_ms'))} ms"
        ),
        (
            "- that idle figure covers input generation, JIT compilation and the reference "
            "as well as your block, so it is not a measure of your kernels; the per-iteration "
            "section below is"
        ),
        (
            f"- summed activity time: {_ms(timeline.get('activity_time_ms'))} ms; this may "
            "exceed active union when streams overlap"
        ),
        (
            "- source labels are hints: `candidate:<symbol>` is matched against this candidate's "
            "`@cute.kernel` definitions; other labels describe likely harness/runtime work"
        ),
    ]
    _append_symbols(lines, "candidate `@cute.kernel` symbols", timeline.get("candidate_kernel_symbols"))
    _append_symbols(lines, "observed candidate symbols", timeline.get("observed_candidate_symbols"))
    _append_symbols(
        lines,
        "candidate symbols not observed in this trace",
        timeline.get("unobserved_candidate_symbols"),
    )

    lines.extend(_iteration_lines(timeline.get("iteration"), kernels))

    if isinstance(aggregate, Mapping):
        hot = aggregate.get("top_kernels")
        if isinstance(hot, list) and hot:
            lines.extend(
                [
                    "",
                    "## Hottest GPU kernels (legacy aggregate)",
                    "",
                    "| kernel | ms | calls | % GPU |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for kernel in hot:
                if not isinstance(kernel, Mapping):
                    continue
                lines.append(
                    f"| {_name(kernel.get('name', ''))} | {_ms(kernel.get('total_ms'))} | "
                    f"{_count(kernel.get('calls'))} | {_pct(kernel.get('pct_gpu_time'))} |"
                )

    lines.extend(
        [
            "",
            "## Kernel legend",
            "",
            "| id | source hint | kernel | calls | total us | launch |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for kernel in kernels:
        if not isinstance(kernel, Mapping):
            continue
        lines.append(
            f"| {_cell(kernel.get('id', ''))} | {_cell(kernel.get('source_hint', ''))} | "
            f"{_name(kernel.get('name', ''))} | {_count(kernel.get('calls'))} | "
            f"{_us(kernel.get('total_us'))} | {_launch_text(kernel)} |"
        )
    if not kernels:
        lines.append("| - | - | no GPU kernels observed | 0 | 0.000 | - |")

    lines.extend(
        [
            "",
            "## Ordered GPU activities",
            "",
            "`gap before` is device-wide idle time since all earlier GPU activity ended; "
            "zero means overlap or no measurable hole.",
            "",
            "| # | start ms | gap before us | duration us | lane | activity |",
            "| ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for event in events:
        if not isinstance(event, Mapping):
            continue
        lines.append(
            f"| {_count(event.get('index'))} | {_ms(event.get('start_ms'))} | "
            f"{_us(event.get('gap_before_us'))} | {_us(event.get('duration_us'))} | "
            f"{_lane(event)} | {_activity(event)} |"
        )
    if not events:
        lines.append("| - | - | - | - | - | no GPU activities observed |")
    return "\n".join(lines) + "\n"


def _iteration_lines(iteration: Any, kernels: Any = None) -> list[str]:
    """Render the representative iteration: start, end and idle per launch.

    Rows carry the legend's alias id. Resolve it back to the kernel's own name
    here: the legend does hold the mapping, but a table of `K24`..`K30` makes
    an author scroll a 500-row document to learn which of its kernels the gap
    sits in front of, and the point of this section is to be readable alone.
    """
    if not isinstance(iteration, Mapping):
        return []
    rows = iteration.get("rows")
    if not isinstance(rows, list) or not rows:
        return []
    names: dict[str, str] = {}
    if isinstance(kernels, list):
        for kernel in kernels:
            if isinstance(kernel, Mapping):
                symbol = str(kernel.get("source_hint", ""))
                names[str(kernel.get("id", ""))] = (
                    symbol.split("candidate:", 1)[1]
                    if symbol.startswith("candidate:")
                    else str(kernel.get("name", ""))
                )

    lines = [
        "",
        "## One iteration of your block, launch by launch",
        "",
        (
            f"Median of {_count(iteration.get('iterations'))} timed iterations, "
            f"{_count(iteration.get('kernels_per_iteration'))} launches each. "
            "Microseconds from the start of the iteration."
        ),
        "",
        (
            f"- wall span {_us(iteration.get('span_us'))}; kernels busy "
            f"{_us(iteration.get('busy_us'))}; idle between your kernels "
            f"{_us(iteration.get('gap_us'))} ({_pct(iteration.get('gap_pct'))}% of the span)"
        ),
    ]
    worst = str(iteration.get("largest_gap_before", ""))
    if worst:
        lines.append(
            f"- largest single gap {_us(iteration.get('largest_gap_us'))} immediately before "
            f"`{_name(names.get(worst, worst))}`"
        )
    if isinstance(iteration.get("between_iterations_us"), (int, float)):
        lines.append(
            f"- {_us(iteration.get('between_iterations_us'))} elapses between one iteration "
            "and the next; that is host-side dispatch outside your block, excluded from the "
            "idle figure above, and fusing cannot remove it"
        )
    lines.extend(
        [
            "",
            "| # | kernel | id | start | end | dur | idle before |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        gap = row.get("gap_before_us")
        alias = str(row.get("activity", ""))
        lines.append(
            f"| {position} | {_name(names.get(alias, alias))} | {_cell(alias)} "
            f"| {_us(row.get('start_us'))} | {_us(row.get('end_us'))} "
            f"| {_us(row.get('duration_us'))} | {'—' if gap is None else _us(gap)} |"
        )
    return lines


def compact_timeline_metadata(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Keep timeline diagnostics in result.json without duplicating every trace event."""
    return {
        str(key): value
        for key, value in timeline.items()
        if key not in {"events", "kernels", "device"}
    }


def _device(trace: Mapping[str, Any]) -> dict[str, Any]:
    properties = trace.get("deviceProperties")
    first = properties[0] if isinstance(properties, list) and properties else None
    if not isinstance(first, Mapping):
        return {}
    fields = (
        ("name", "name"),
        ("numSms", "num_sms"),
        ("sharedMemPerBlockOptin", "shared_mem_per_block_optin"),
    )
    return {target: first[source] for source, target in fields if first.get(source) is not None}


def _kind(category: str) -> str:
    return {"gpu_memcpy": "memcpy", "gpu_memset": "memset"}.get(category, category)


def _matching_symbol(name: str, symbols: tuple[str, ...]) -> str:
    matches = [
        symbol
        for symbol in symbols
        if name == f"kernel_cutlass_{symbol}" or name.startswith(f"kernel_cutlass_{symbol}_")
    ]
    return max(matches, key=len) if matches else ""


def _launch(args: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, target in (
        ("grid", "grid"),
        ("block", "block"),
        ("shared memory", "shared_memory_bytes"),
        ("registers per thread", "registers_per_thread"),
    ):
        value = args.get(source)
        if source in {"grid", "block"}:
            if isinstance(value, (list, tuple)):
                result[target] = list(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[target] = int(value)
    return result


def _append_symbols(lines: list[str], label: str, value: Any) -> None:
    if not isinstance(value, list):
        return
    rendered = ", ".join(f"`{str(item).replace('`', '')}`" for item in value) or "none"
    lines.append(f"- {label}: {rendered}")


def _launch_text(kernel: Mapping[str, Any]) -> str:
    parts = []
    for label, key in (("grid", "grid"), ("block", "block")):
        value = kernel.get(key)
        if isinstance(value, list):
            parts.append(f"{label}=" + "x".join(str(item) for item in value))
    for label, key in (("smem", "shared_memory_bytes"), ("regs", "registers_per_thread")):
        value = kernel.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{label}={int(value)}")
    return _cell(", ".join(parts) or "-")


def _lane(event: Mapping[str, Any]) -> str:
    device = event.get("device")
    stream = event.get("stream")
    parts = []
    if device is not None:
        parts.append(f"d{device}")
    if stream is not None:
        parts.append(f"s{stream}")
    return _cell("/".join(parts) or "-")


def _activity(event: Mapping[str, Any]) -> str:
    activity = _name(event.get("activity", ""))
    byte_count = event.get("bytes")
    if isinstance(byte_count, (int, float)) and not isinstance(byte_count, bool):
        activity += f" ({int(byte_count)} B)"
    return activity


def _omitted(count: int) -> str:
    return f"\n{count} lower-cost kernel(s) omitted.\n" if count > 0 else ""


def _name(value: Any) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= NAME_WIDTH else text[: NAME_WIDTH - 1].rstrip() + "…"


def _cell(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _ms(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def _us(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def _pct(value: Any) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def _count(value: Any) -> str:
    return str(int(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"
