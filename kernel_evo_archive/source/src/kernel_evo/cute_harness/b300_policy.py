"""Static integrity checks for candidate-only B300 CuTe submissions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ALLOWED_IMPORTS = {"cutlass", "math", "torch", "typing"}
FORBIDDEN_IMPORTS = {
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
FORBIDDEN_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
MAX_SUBMISSION_BYTES = 512 * 1024


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


@dataclass(frozen=True)
class CheckReport:
    errors: tuple[str, ...]
    observed_calls: frozenset[str]
    cute_kernels: int
    cute_jit_functions: int
    has_model_new: bool

    @property
    def passed(self) -> bool:
        return not self.errors


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.calls: set[str] = set()
        self.kernels = 0
        self.jit_functions = 0
        self.jit_function_names: set[str] = set()
        self.has_model_new = False

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.error(node, "relative imports are forbidden")
        else:
            self._check_import(node, node.module or "")
        self.generic_visit(node)

    def _check_import(self, node: ast.AST, module: str) -> None:
        root = module.split(".", 1)[0]
        if root in FORBIDDEN_IMPORTS:
            self.error(node, f"forbidden import: {module}")
        elif root not in ALLOWED_IMPORTS:
            self.error(node, f"import is not allowlisted: {module}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.error(node, f"dunder attribute access is forbidden: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _name(node.func)
        if name:
            self.calls.add(name)
            if name == "main":
                self.error(node, "candidate must not call main(); the evaluator owns it")
            if name in FORBIDDEN_CALLS or name in {f"builtins.{value}" for value in FORBIDDEN_CALLS}:
                self.error(node, f"forbidden call: {name}")
            if name == "print" or name == "torch" or name.startswith("torch.") or name == "F" or name.startswith("F."):
                self.error(node, f"call is forbidden in candidate code: {name}")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if any(isinstance(child, ast.Name) and child.id == "__name__" for child in ast.walk(node.test)):
            self.error(node, "if __name__ guard is forbidden; the evaluator owns execution")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "main":
            self.error(node, "candidate must not define main(); the evaluator owns it")
        decorators = {_name(value) for value in node.decorator_list}
        self.kernels += "cute.kernel" in decorators
        self.jit_functions += "cute.jit" in decorators
        if "cute.jit" in decorators:
            self.jit_function_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "ModelNew":
            self.has_model_new |= any(
                isinstance(item, ast.FunctionDef)
                and item.name == "forward"
                or isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "forward" for target in item.targets)
                or isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == "forward"
                for item in node.body
            )
        self.generic_visit(node)


def check_candidate(path: Path, policy: dict) -> CheckReport:
    errors: list[str] = []
    if not path.is_file():
        return CheckReport((f"candidate does not exist: {path}",), frozenset(), 0, 0, False)
    if path.stat().st_size > MAX_SUBMISSION_BYTES:
        errors.append(f"candidate exceeds {MAX_SUBMISSION_BYTES} bytes")
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        return CheckReport((f"cannot parse candidate: {error}",), frozenset(), 0, 0, False)

    visitor = _Visitor()
    visitor.visit(module)
    errors.extend(visitor.errors)
    if not visitor.has_model_new:
        errors.append("candidate must define the common ModelNew.forward interface")
    minimum_kernels = int(policy["minimum_cute_kernels"])
    minimum_jit = int(policy["minimum_cute_jit_functions"])
    if visitor.kernels < minimum_kernels:
        errors.append(f"found {visitor.kernels} @cute.kernel functions; task requires {minimum_kernels}")
    if visitor.jit_functions < minimum_jit:
        errors.append(f"found {visitor.jit_functions} @cute.jit functions; task requires {minimum_jit}")
    entrypoint = str(policy["entrypoint_jit"])
    if entrypoint not in visitor.jit_function_names:
        errors.append(f"required entry point must remain `@cute.jit def {entrypoint}(...)`")
    for required_call in policy["required_calls"]:
        if required_call not in visitor.calls:
            errors.append(f"required CuTe call not found: {required_call}")
    return CheckReport(
        tuple(errors),
        frozenset(visitor.calls),
        visitor.kernels,
        visitor.jit_functions,
        visitor.has_model_new,
    )
