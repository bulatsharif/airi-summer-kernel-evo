from __future__ import annotations

import ast
from dataclasses import dataclass


class UnsafeSourceError(ValueError):
    pass


@dataclass(frozen=True)
class Violation:
    line: int
    message: str


_BANNED_MODULES = {
    "commands",
    "ctypes",
    "importlib",
    "marshal",
    "multiprocessing",
    "pickle",
    "pty",
    "subprocess",
}
_BANNED_BUILTIN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "setattr",
    "vars",
}
_BANNED_QUALIFIED_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.fork",
    "os.forkpty",
    "os.popen",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
}


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.module_aliases: dict[str, str] = {}
        self.call_aliases: dict[str, str] = {}
        self.violations: list[Violation] = []

    def reject(self, node: ast.AST, message: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 1), message))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _BANNED_MODULES:
                self.reject(node, f"import of {root!r} is not allowed")
            self.module_aliases[alias.asname or root] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in _BANNED_MODULES:
            self.reject(node, f"import from {root!r} is not allowed")
        for alias in node.names:
            local_name = alias.asname or alias.name
            qualified = f"{module}.{alias.name}".strip(".")
            self.call_aliases[local_name] = qualified
            if qualified in _BANNED_QUALIFIED_CALLS:
                self.reject(node, f"import of process/shell API {qualified!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.reject(node, "dunder attribute access is not allowed")
        qualified = self._qualified_name(node)
        if qualified in _BANNED_QUALIFIED_CALLS:
            self.reject(node, f"process/shell API {qualified!r} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "__builtins__":
            self.reject(node, "access to __builtins__ is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        qualified = self._qualified_name(node.func)
        if qualified in _BANNED_BUILTIN_CALLS:
            self.reject(node, f"call to {qualified!r} is not allowed")
        if qualified in _BANNED_QUALIFIED_CALLS:
            self.reject(node, f"process/shell call {qualified!r} is not allowed")
        self.generic_visit(node)

    def _qualified_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.call_aliases.get(node.id, node.id)
        if not isinstance(node, ast.Attribute):
            return None
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        base = self.module_aliases.get(current.id, current.id)
        return ".".join([base, *reversed(parts)])


def validate_source(source: str) -> None:
    try:
        tree = ast.parse(source, filename="submission.py")
    except SyntaxError as exc:
        raise UnsafeSourceError(
            f"source is not valid Python at line {exc.lineno}: {exc.msg}"
        ) from exc

    visitor = _PolicyVisitor()
    visitor.visit(tree)
    if visitor.violations:
        details = "; ".join(
            f"line {item.line}: {item.message}" for item in visitor.violations
        )
        raise UnsafeSourceError(details)
