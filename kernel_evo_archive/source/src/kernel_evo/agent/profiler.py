"""Optional harness-owned profiling with compact author feedback."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from kernel_evo.agent.models import EvaluationContext, EvaluationResult
from kernel_evo.core.profile.contracts import run_profile_subprocess
from kernel_evo.core.stages.profile.summary_compaction import summarize_profiler_for_llm


@dataclass(slots=True)
class ProfileResult:
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CandidateProfiler(Protocol):
    def profile(self, context: EvaluationContext, result: EvaluationResult) -> ProfileResult | str: ...


class KernelBenchProfiler:
    """Run harness-selected profilers; selection policy belongs to the controller."""

    def profile(self, context: EvaluationContext, result: EvaluationResult) -> ProfileResult:
        if context.problem_dir is None or not (
            result.valid or (result.compiled and result.correctness)
        ):
            return ProfileResult()
        runners = tuple(str(item) for item in context.config.get("profile_runners", ["torch"]))
        root = context.island_dir / "profile"
        root.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, Any] = {}
        ref_arch_src = str(context.run_config.get("ref_arch_src", ""))

        if "torch" in runners:
            summaries["torch"] = self._run_torch(
                context, root / "torch", ref_arch_src, result
            )
        if "nsys" in runners:
            summaries["nsys"] = self._run_nsys(context, root / "nsys", ref_arch_src)
        if "ncu" in runners:
            summaries["ncu"] = self._run_ncu(context, root / "ncu", ref_arch_src)
        if "sanitizer" in runners and str(context.config.get("backend", "")) == "cute":
            summaries["sanitizer"] = self._run_sanitizer(
                context,
                root / "sanitizer",
                ref_arch_src,
            )

        compact = {
            runner: summarize_profiler_for_llm(profiler_name=runner, summary=summary)
            for runner, summary in summaries.items()
            if isinstance(summary, dict)
        }
        return ProfileResult(summary=_compact_markdown(compact), data=compact)

    @staticmethod
    def _run_torch(
        context: EvaluationContext,
        out_dir: Path,
        ref_arch_src: str,
        result: EvaluationResult,
    ) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        reference = out_dir / "reference.py"
        reference.write_text(ref_arch_src, encoding="utf-8")
        run_config_path = context.problem_dir / "run_config.json"  # type: ignore[operator]
        command = [
            sys.executable,
            "-m",
            "kernel_evo.tools.profile_target",
            "--run-config",
            str(run_config_path),
            "--candidate-file",
            str(context.candidate_path),
            "--reference-file",
            str(reference),
            "--out-dir",
            str(out_dir),
            "--full-profile",
        ]
        if result.runtime_us is not None:
            command.extend(["--evaluator-runtime-us", str(result.runtime_us)])
        completed = run_profile_subprocess(
            command,
            timeout=float(context.run_config.get("profile_subprocess_timeout", 600.0)),
            text=True,
            capture_output=True,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "reason": (completed.stderr or completed.stdout)[-2_000:],
        }

    @staticmethod
    def _run_nsys(context: EvaluationContext, out_dir: Path, ref_arch_src: str) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        reference = out_dir / "reference.py"
        reference.write_text(ref_arch_src, encoding="utf-8")
        run_config_path = context.problem_dir / "run_config.json"  # type: ignore[operator]
        command = [
            sys.executable,
            "-m",
            "kernel_evo.tools.profile_nsys",
            "--run-config",
            str(run_config_path),
            "--candidate-file",
            str(context.candidate_path),
            "--reference-file",
            str(reference),
            "--out-dir",
            str(out_dir),
        ]
        completed = run_profile_subprocess(
            command,
            timeout=float(context.run_config.get("profile_subprocess_timeout", 600.0)),
            text=True,
            capture_output=True,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "reason": (completed.stderr or completed.stdout)[-2_000:],
        }

    @staticmethod
    def _run_ncu(context: EvaluationContext, out_dir: Path, ref_arch_src: str) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        reference = out_dir / "reference.py"
        reference.write_text(ref_arch_src, encoding="utf-8")
        run_config_path = context.problem_dir / "run_config.json"  # type: ignore[operator]
        command = [
            sys.executable,
            "-m",
            "kernel_evo.tools.profile_ncu",
            "--run-config",
            str(run_config_path),
            "--candidate-file",
            str(context.candidate_path),
            "--reference-file",
            str(reference),
            "--out-dir",
            str(out_dir),
        ]
        completed = run_profile_subprocess(
            command,
            timeout=float(context.run_config.get("profile_subprocess_timeout", 600.0)),
            text=True,
            capture_output=True,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "reason": (completed.stderr or completed.stdout)[-2_000:],
        }

    @staticmethod
    def _run_sanitizer(
        context: EvaluationContext,
        out_dir: Path,
        ref_arch_src: str,
    ) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        reference = out_dir / "reference.py"
        reference.write_text(ref_arch_src, encoding="utf-8")
        run_config_path = context.problem_dir / "run_config.json"  # type: ignore[operator]
        tools_value = context.config.get("cute_sanitizer_tools", ["memcheck", "synccheck"])
        if isinstance(tools_value, str):
            tools = tools_value
        else:
            tools = ",".join(str(item) for item in tools_value)
        command = [
            sys.executable,
            "-m",
            "kernel_evo.tools.sanitize_candidate",
            "--run-config",
            str(run_config_path),
            "--candidate-file",
            str(context.candidate_path),
            "--reference-file",
            str(reference),
            "--out-dir",
            str(out_dir),
            "--tools",
            tools,
        ]
        completed = run_profile_subprocess(
            command,
            timeout=float(context.run_config.get("profile_subprocess_timeout", 600.0)),
            text=True,
            capture_output=True,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "reason": (completed.stderr or completed.stdout)[-2_000:],
        }


def coerce_profile_result(value: ProfileResult | str | Mapping[str, Any] | None) -> ProfileResult:
    if value is None:
        return ProfileResult()
    if isinstance(value, ProfileResult):
        return value
    if isinstance(value, str):
        return ProfileResult(summary=value)
    if isinstance(value, Mapping):
        data = dict(value)
        return ProfileResult(summary=_compact_markdown(data), data=data)
    raise TypeError(f"Profiler returned unsupported value: {type(value).__name__}")


def profile_result_status(profile: ProfileResult) -> str:
    """Summarize nested runner outcomes for controller transitions."""
    if not profile.data:
        return "completed" if profile.summary.strip() else "failed"

    statuses: list[str] = []
    top_level = str(profile.data.get("status", "")).strip().lower()
    if top_level:
        statuses.append(top_level)
    statuses.extend(
        str(value.get("status", "")).strip().lower()
        for value in profile.data.values()
        if isinstance(value, Mapping) and str(value.get("status", "")).strip()
    )
    if not statuses:
        return "completed"
    if any(status in {"completed", "passed", "success", "ok"} for status in statuses):
        return "completed"
    if all(status in {"skipped", "not_selected"} for status in statuses):
        return "skipped"
    return "failed"


def _compact_markdown(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    lines = ["Profiler summary:"]
    encoded = json.dumps(value, ensure_ascii=False, indent=2, default=str).splitlines()
    lines.extend(encoded)
    return "\n".join(lines)
