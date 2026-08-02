"""One bounded LLM critique between authoring turns.

The author sees only its own packet, so a failure it cannot diagnose repeats.
After each evaluation a read-only critic reads the candidate and the harness
diagnostic and returns a few short hints, which the next turn receives at the
top of its compressed feedback.

The critic writes nothing: hints are parsed out of its session transcript, so
it needs no write permission anywhere near a candidate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


HINT_LIMIT = 3
HINT_MAX_CHARS = 240
DIAGNOSTIC_MAX_CHARS = 4_000
_TEXT_KEYS = ("text", "content", "message", "summary")
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def critic_task_markdown(
    *,
    task_id: str,
    iteration: int,
    candidate_path: str,
    diagnostic: str,
    hint_limit: int = HINT_LIMIT,
) -> str:
    """The critic's whole prompt: one candidate, one diagnostic, one JSON reply."""
    return f"""# CuTe FP8 failure critique

Task `{task_id}`, authoring turn {iteration} has been evaluated. Read the candidate
at `{candidate_path}` and the diagnostic below, then explain what to change.

## Harness diagnostic

```
{_clip(diagnostic, DIAGNOSTIC_MAX_CHARS)}
```

## Reply

Return **only** this JSON object, at most {hint_limit} hints, each one sentence:

```json
{{"hints": ["..."]}}
```

Each hint names a concrete, checkable change to the next candidate: the construct
to fix, the shape or layout that disagrees, the API spelling that does not exist.
Write no code and no general advice. If the diagnostic does not identify a cause,
say so in one hint rather than guessing.
"""


def build_diagnostic(
    *,
    result: Mapping[str, Any],
    stderr: str = "",
    stdout: str = "",
    profile_summary: str = "",
) -> str:
    """Assemble what the harness knows about one evaluated candidate."""
    status = "valid" if result.get("valid") else "invalid"
    speedup = float(result.get("speedup", 0.0) or 0.0)
    lines = [f"status: {status} (compiled={bool(result.get('compiled'))}, speedup={speedup:.3f}x)"]
    for label, value in (
        ("error", str(result.get("error", "") or "")),
        ("stderr (tail)", _tail(stderr)),
        ("stdout (tail)", _tail(stdout)),
        ("profile", profile_summary),
    ):
        if value.strip():
            lines.append(f"\n--- {label} ---\n{value.strip()}")
    return "\n".join(lines)


def parse_critic_hints(transcript: str, *, limit: int = HINT_LIMIT) -> list[str]:
    """Recover hints from an OpenCode JSON-event transcript, tolerantly.

    The event schema is the CLI's, not ours, so treat every string in it as
    possible model output and take the last well-formed hint payload.
    """
    for candidate in reversed(_json_payloads(_transcript_text(transcript))):
        hints = _hints_from(candidate)
        if hints:
            return _normalize(hints, limit)
    return []


def _transcript_text(transcript: str) -> str:
    chunks: list[str] = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            chunks.append(line)
            continue
        # Keep the envelope too: a line may be the reply rather than wrap it.
        # Nested text comes after, so a real message still outranks its envelope.
        chunks.append(line)
        chunks.extend(_strings(event))
    return "\n".join(chunks)


def _strings(value: Any, depth: int = 0) -> list[str]:
    if depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, item in value.items():
            if isinstance(item, str):
                if str(key) in _TEXT_KEYS:
                    found.append(item)
            else:
                found.extend(_strings(item, depth + 1))
        return found
    if isinstance(value, list):
        return [text for item in value for text in _strings(item, depth + 1)]
    return []


def _json_payloads(text: str) -> list[Any]:
    """Every JSON value in the text, in document order, so the last one wins."""
    found: list[tuple[int, Any]] = []
    for match in _FENCE.finditer(text):
        try:
            found.append((match.start(), json.loads(match.group(1))))
        except ValueError:
            continue
    found.extend(_scan_objects(text))
    return [payload for _, payload in sorted(found, key=lambda item: item[0])]


def _scan_objects(text: str) -> list[tuple[int, Any]]:
    """Find balanced `{...}`/`[...]` spans that parse, without a real parser."""
    payloads: list[tuple[int, Any]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        starts = [index for index, char in enumerate(text) if char == opener]
        for start in starts[:200]:
            depth = 0
            for index in range(start, min(len(text), start + 20_000)):
                if text[index] == opener:
                    depth += 1
                elif text[index] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            payloads.append((start, json.loads(text[start : index + 1])))
                        except ValueError:
                            pass
                        break
    return payloads


def _hints_from(payload: Any) -> list[str]:
    if isinstance(payload, Mapping):
        for key in ("hints", "hint", "feedback"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
        return []
    if isinstance(payload, list) and payload and all(isinstance(item, str) for item in payload):
        return [item for item in payload if item.strip()]
    return []


def _normalize(hints: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for hint in hints:
        text = _clip(" ".join(str(hint).split()), HINT_MAX_CHARS)
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= max(1, limit):
            break
    return seen


def _tail(text: str, limit: int = 1_500) -> str:
    text = str(text or "").rstrip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
