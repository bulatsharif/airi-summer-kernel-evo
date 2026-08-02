#!/usr/bin/env python3
"""Check whether an OpenAI-compatible endpoint returns structured tool calls.

OpenCode drives the author agent entirely through the `tool_calls` field. A
server that emits the model's tool syntax as plain `content` instead leaves the
agent unable to read, write, or run anything, and every turn silently produces
no candidate -- which is what a Gemma deployment did while Qwen on the same host
worked, so this is a serving-flag difference, not a model limitation.

Usage:
    GEMMA_BASE_URL=... GEMMA_API_KEY=... python3 tools/probe-toolcalls.py
    python3 tools/probe-toolcalls.py http://host:30001/v1 google/gemma-4-31B-it

Keys come from the environment; none are written here.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

ENDPOINTS = [
    ("qwen", os.environ.get("QWEN_BASE_URL", ""), "Qwen/Qwen3.6-35B-A3B", "QWEN_API_KEY"),
    ("gemma", os.environ.get("GEMMA_BASE_URL", ""), "google/gemma-4-31B-it", "GEMMA_API_KEY"),
]


def probe(base_url: str, model: str, api_key: str) -> tuple[bool, str]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Read /etc/hosts using the read_file tool."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 150,
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        return False, f"unreachable: {error}"
    message = payload["choices"][0]["message"]
    calls = message.get("tool_calls")
    if calls:
        function = calls[0].get("function", {})
        return True, f"structured: {function.get('name')}({function.get('arguments')})"
    return False, f"UNPARSED, arrived as text: {str(message.get('content'))[:110]}"


def main() -> int:
    targets = list(ENDPOINTS)
    if len(sys.argv) >= 3:
        targets = [("custom", sys.argv[1], sys.argv[2], "GEMMA_API_KEY")]

    worst = 0
    for label, base_url, model, key_var in targets:
        if not base_url:
            print(f"{label:6} SKIP  ({key_var.replace('_API_KEY', '_BASE_URL')} not set)")
            continue
        key = os.environ.get(key_var, "")
        if not key:
            print(f"{label:6} SKIP  ({key_var} not set)")
            continue
        ok, detail = probe(base_url, model, key)
        print(f"{label:6} {'OK  ' if ok else 'FAIL'}  {detail}")
        worst = worst or (0 if ok else 1)
    if worst:
        print(
            "\nA FAIL means the server needs a tool-call parser for that model's format.\n"
            "Check the parsers your build supports:\n"
            "    python -m sglang.launch_server --help | grep -A5 tool-call-parser\n"
            "and mirror the flags used by the endpoint that reports OK."
        )
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
