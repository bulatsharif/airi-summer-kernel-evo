from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"


class TaskError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    directory: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def title(self) -> str:
        return str(self.data["title"])

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.data["policy"])

    @property
    def validation(self) -> dict[str, Any]:
        return dict(self.data["validation"])

    def member_path(self, field: str) -> Path:
        value = self.data.get(field)
        if not isinstance(value, str) or not value:
            raise TaskError(f"{self.id}: field {field!r} must be a path string")
        path = (self.directory / value).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise TaskError(
                f"{self.id}: {field!r} escapes the repository: {value}"
            ) from error
        return path

    @property
    def prompt_path(self) -> Path:
        return self.member_path("prompt")

    @property
    def starter_path(self) -> Path:
        return self.member_path("starter")

    @property
    def baseline_path(self) -> Path:
        return self.member_path("baseline")

    @property
    def reference_paths(self) -> tuple[Path, ...]:
        values = self.data.get("references", [])
        if not isinstance(values, list):
            raise TaskError(f"{self.id}: 'references' must be a path list")
        paths: list[Path] = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                raise TaskError(
                    f"{self.id}: references[{index}] must be a path string"
                )
            path = (self.directory / value).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError as error:
                raise TaskError(
                    f"{self.id}: reference escapes the repository: {value}"
                ) from error
            paths.append(path)
        return tuple(paths)

    @property
    def agent_skill_paths(self) -> tuple[Path, ...]:
        values = self.data.get("agent_skills", [])
        if not isinstance(values, list):
            raise TaskError(f"{self.id}: 'agent_skills' must be a path list")
        paths: list[Path] = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                raise TaskError(
                    f"{self.id}: agent_skills[{index}] must be a path string"
                )
            path = (self.directory / value).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError as error:
                raise TaskError(
                    f"{self.id}: agent skill escapes the repository: {value}"
                ) from error
            paths.append(path)
        return tuple(paths)

    def public_manifest(self) -> dict[str, Any]:
        public = dict(self.data)
        public.pop("baseline", None)
        return public


def _require_mapping(
    data: dict[str, Any],
    field: str,
    manifest_path: Path,
) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise TaskError(f"{manifest_path}: {field!r} must be an object")
    return value


def _validate_manifest(data: Any, manifest_path: Path) -> TaskSpec:
    if not isinstance(data, dict):
        raise TaskError(f"{manifest_path}: manifest root must be an object")

    required_strings = ("id", "title", "prompt", "starter", "baseline")
    for field in required_strings:
        if not isinstance(data.get(field), str) or not data[field]:
            raise TaskError(
                f"{manifest_path}: {field!r} must be a non-empty string"
            )

    if data.get("schema_version") != 1:
        raise TaskError(
            f"{manifest_path}: unsupported schema_version "
            f"{data.get('schema_version')!r}"
        )

    task_id = data["id"]
    if task_id != manifest_path.parent.name:
        raise TaskError(
            f"{manifest_path}: id {task_id!r} must equal directory name "
            f"{manifest_path.parent.name!r}"
        )

    policy = _require_mapping(data, "policy", manifest_path)
    validation = _require_mapping(data, "validation", manifest_path)

    for field in ("minimum_cute_kernels", "minimum_cute_jit_functions"):
        value = policy.get(field)
        if not isinstance(value, int) or value < 1:
            raise TaskError(
                f"{manifest_path}: policy.{field} must be a positive integer"
            )

    required_calls = policy.get("required_calls")
    if (
        not isinstance(required_calls, list)
        or not required_calls
        or not all(isinstance(item, str) and item for item in required_calls)
    ):
        raise TaskError(
            f"{manifest_path}: policy.required_calls must be a non-empty "
            "string list"
        )

    pattern = validation.get("success_pattern")
    if not isinstance(pattern, str) or not pattern:
        raise TaskError(
            f"{manifest_path}: validation.success_pattern is required"
        )
    try:
        re.compile(pattern)
    except re.error as error:
        raise TaskError(
            f"{manifest_path}: invalid success_pattern: {error}"
        ) from error

    spec = TaskSpec(manifest_path.parent.resolve(), data)
    for field in ("prompt", "starter", "baseline"):
        path = spec.member_path(field)
        if not path.is_file():
            raise TaskError(f"{manifest_path}: missing {field} file: {path}")
    reference_names: set[str] = set()
    for path in spec.reference_paths:
        if not path.is_file():
            raise TaskError(f"{manifest_path}: missing reference file: {path}")
        if path.name in reference_names:
            raise TaskError(
                f"{manifest_path}: duplicate reference basename: {path.name}"
            )
        reference_names.add(path.name)
    skill_names: set[str] = set()
    for path in spec.agent_skill_paths:
        if not path.is_dir() or not (path / "SKILL.md").is_file():
            raise TaskError(
                f"{manifest_path}: invalid agent skill directory: {path}"
            )
        if path.name in skill_names:
            raise TaskError(
                f"{manifest_path}: duplicate agent skill name: {path.name}"
            )
        skill_names.add(path.name)
    return spec


def discover_tasks(tasks_root: Path = TASKS_ROOT) -> dict[str, TaskSpec]:
    if not tasks_root.is_dir():
        raise TaskError(f"tasks directory does not exist: {tasks_root}")

    tasks: dict[str, TaskSpec] = {}
    for manifest_path in sorted(tasks_root.glob("*/task.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TaskError(f"cannot read {manifest_path}: {error}") from error
        spec = _validate_manifest(data, manifest_path)
        if spec.id in tasks:
            raise TaskError(f"duplicate task id: {spec.id}")
        tasks[spec.id] = spec

    if not tasks:
        raise TaskError(f"no task manifests found under {tasks_root}")
    return tasks


def load_task(task_id: str) -> TaskSpec:
    tasks = discover_tasks()
    try:
        return tasks[task_id]
    except KeyError as error:
        known = ", ".join(tasks)
        raise TaskError(f"unknown task {task_id!r}; known tasks: {known}") from error
