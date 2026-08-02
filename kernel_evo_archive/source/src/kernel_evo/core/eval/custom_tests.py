"""Small, task-owned correctness suites for kernel evaluation.

Custom suites export ``run_tests(context)``.  They may use ordinary assertions
and may return ``None``, a boolean, one result mapping, or a sequence of result
mappings.  KernelEvo deliberately does not prescribe elementwise comparison:
the task may check norms, invariants, mutation, or any other semantic contract.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def run_custom_test_file(path: str | Path, context: Any) -> dict[str, Any]:
    """Load and execute one custom test file, returning JSON-safe evidence."""

    test_path = Path(path).expanduser().resolve()
    if not test_path.is_file():
        return {
            "passed": False,
            "tests": [],
            "error": f"Custom test file not found: {test_path}",
        }
    try:
        spec = importlib.util.spec_from_file_location(
            f"kernelevo_custom_tests_{abs(hash(test_path))}", test_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load custom tests from {test_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        runner = getattr(module, "run_tests", None)
        if not callable(runner):
            raise AttributeError("Custom test file must export run_tests(context)")
        raw = runner(context)
        tests = _normalize_results(raw)
        passed = all(bool(item.get("passed", False)) for item in tests)
        return {"passed": passed, "tests": tests, "error": "" if passed else "custom assertion failed"}
    except Exception as exc:
        return {
            "passed": False,
            "tests": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _normalize_results(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return [{"name": "custom_tests", "passed": True}]
    if isinstance(value, bool):
        return [{"name": "custom_tests", "passed": value}]
    if isinstance(value, Mapping):
        return [_normalize_result(value, 0)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_result(item if isinstance(item, Mapping) else {"passed": bool(item)}, index)
            for index, item in enumerate(value)
        ]
    raise TypeError("run_tests(context) must return None, bool, mapping, or sequence")


def _normalize_result(value: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = {str(key): _json_value(item) for key, item in value.items()}
    result.setdefault("name", f"custom_test_{index + 1}")
    result.setdefault("passed", True)
    result["passed"] = bool(result["passed"])
    return result


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)
