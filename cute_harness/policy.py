from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import TaskSpec


MAX_SUBMISSION_BYTES = 512 * 1024
ALLOWED_IMPORT_ROOTS = {
    "cutlass",
    "math",
    "torch",
    "typing",
}
FORBIDDEN_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "runpy",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
def qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = qualified_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


@dataclass(frozen=True)
class CheckReport:
    path: Path
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    observed_calls: frozenset[str]
    cute_kernel_count: int
    cute_jit_count: int
    has_direct_main_call: bool

    @property
    def passed(self) -> bool:
        return not self.errors


class _SubmissionVisitor(ast.NodeVisitor):
    def __init__(self, candidate_mode: bool) -> None:
        self.candidate_mode = candidate_mode
        self.errors: list[str] = []
        self.calls: set[str] = set()
        self.cute_kernel_count = 0
        self.cute_jit_count = 0

    def _error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"line {line}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                self._error(node, f"forbidden import: {alias.name}")
            elif root not in ALLOWED_IMPORT_ROOTS:
                self._error(node, f"import is not allowlisted: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if node.level:
            self._error(node, "relative imports are forbidden")
        elif root in FORBIDDEN_IMPORT_ROOTS:
            self._error(node, f"forbidden import: {module}")
        elif root not in ALLOWED_IMPORT_ROOTS:
            self._error(node, f"import is not allowlisted: {module}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._error(node, f"dunder attribute access is forbidden: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = qualified_name(node.func)
        if name:
            self.calls.add(name)
            if name in FORBIDDEN_CALLS or name in {
                f"builtins.{call}" for call in FORBIDDEN_CALLS
            }:
                self._error(node, f"forbidden call: {name}")
            candidate_forbidden = (
                name == "print"
                or name == "torch"
                or name.startswith("torch.")
                or name == "F"
                or name.startswith("F.")
            )
            if self.candidate_mode and candidate_forbidden:
                self._error(
                    node,
                    f"call is forbidden in candidate code: {name}",
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if any(
            isinstance(child, ast.Name) and child.id == "__name__"
            for child in ast.walk(node.test)
        ):
            self._error(
                node,
                "if __name__ guard is forbidden; call main() directly",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.candidate_mode and node.name == "main":
            self._error(node, "candidate must not define main(); harness owns it")
        decorators = {
            name
            for decorator in node.decorator_list
            if (name := qualified_name(decorator)) is not None
        }
        if "cute.kernel" in decorators:
            self.cute_kernel_count += 1
        if "cute.jit" in decorators:
            self.cute_jit_count += 1
        self.generic_visit(node)


def _has_direct_main_call(module: ast.Module) -> bool:
    for statement in module.body:
        if not isinstance(statement, ast.Expr):
            continue
        expression = statement.value
        if (
            isinstance(expression, ast.Call)
            and qualified_name(expression.func) == "main"
        ):
            return True
    return False


def check_submission(
    task: TaskSpec,
    path: Path,
    *,
    candidate_mode: bool = True,
) -> CheckReport:
    path = path.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return CheckReport(
            path,
            (f"submission does not exist: {path}",),
            (),
            frozenset(),
            0,
            0,
            False,
        )

    size = path.stat().st_size
    if size > MAX_SUBMISSION_BYTES:
        errors.append(
            f"submission is {size} bytes; maximum is {MAX_SUBMISSION_BYTES}"
        )

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return CheckReport(
            path,
            (f"cannot read UTF-8 submission: {error}",),
            (),
            frozenset(),
            0,
            0,
            False,
        )

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno else "unknown line"
        return CheckReport(
            path,
            (f"syntax error at {location}: {error.msg}",),
            (),
            frozenset(),
            0,
            0,
            False,
        )

    visitor = _SubmissionVisitor(candidate_mode)
    visitor.visit(module)
    errors.extend(visitor.errors)

    policy: dict[str, Any] = task.policy
    minimum_kernels = int(policy["minimum_cute_kernels"])
    minimum_jit = int(policy["minimum_cute_jit_functions"])
    if visitor.cute_kernel_count < minimum_kernels:
        errors.append(
            f"found {visitor.cute_kernel_count} @cute.kernel functions; "
            f"task requires at least {minimum_kernels}"
        )
    if visitor.cute_jit_count < minimum_jit:
        errors.append(
            f"found {visitor.cute_jit_count} @cute.jit functions; "
            f"task requires at least {minimum_jit}"
        )

    for required_call in policy["required_calls"]:
        if required_call not in visitor.calls:
            errors.append(f"required CuTe call not found: {required_call}")

    direct_main = _has_direct_main_call(module)
    if candidate_mode and direct_main:
        errors.append("candidate must not call main(); harness owns it")
    if not candidate_mode and not direct_main:
        errors.append("submission must call main() directly at module scope")

    return CheckReport(
        path,
        tuple(errors),
        tuple(warnings),
        frozenset(visitor.calls),
        visitor.cute_kernel_count,
        visitor.cute_jit_count,
        direct_main,
    )
