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
UNAVAILABLE_CUTE_APIS = {
    "cutlass.float32": "use the case-sensitive cutlass.Float32 scalar type",
    "cute._cute": "private compatibility modules are not exposed",
    "cute.GemmMode": "no GemmMode enum is exposed in CUTLASS 4.6.1",
    "cute.arch.blockDim": "use tuple-valued cute.arch.block_dim()",
    "cute.arch.thread_id": "use tuple-valued cute.arch.thread_idx()",
    "cute.arch.grid_dim_x": "use tuple-valued cute.arch.grid_dim()",
    "cute.arch.make_pipeline_state": "import cutlass.pipeline as pipeline and use pipeline.make_pipeline_state(...) instead",
    "cute.cdiv": "use static integer arithmetic such as (x + y - 1) // y",
    "cute.div": "use static integer arithmetic or the appropriate layout op",
    "cute.block_dim": "use tuple-valued cute.arch.block_dim()",
    "cute.fill": "use the exposed cute.full or cute.full_like constructor with an explicit dtype/template",
    "cute.float32": "use cutlass.Float32 for the scalar type",
    "cute.launch_config": "launch the bound @cute.kernel object with .launch()",
    "cute.launch_kernel": "launch the bound @cute.kernel object with .launch()",
    "cute.LayoutEnum": "use utils.LayoutEnum.from_tensor(tensor).mma_major_mode()",
    "cute.make_fake_compact_tensor": "this helper is not exposed",
    "cute.make_shape": "use a plain tuple for static shapes",
    "cute.make_stride": "pass a plain stride tuple to cute.make_layout(..., stride=...)",
    "cute.make_smem_tensor": "use utils.SmemAllocator().allocate_tensor(element_type, layout, byte_alignment=...)",
    "cute.partition_D": "this top-level helper is not exposed; use only the verified cute.local_partition API where its semantics match",
    "cute.partition_S": "this top-level helper is not exposed; use only the verified cute.local_partition API where its semantics match",
    "cute.pointer": "use CuTe tensor tiling, slicing, and partition APIs instead of inventing pointer extraction",
    "cute.raw_pointer_as_ptr": "this pointer-conversion helper is not exposed; preserve and transform CuTe tensor views",
    "cute.thread_id": "use tuple-valued cute.arch.thread_idx()",
    "cute.PipelineTmaUmma": "import cutlass.pipeline as pipeline and use pipeline.PipelineTmaUmma.create(...) instead",
    "cute.arch.PipelineTmaUmma": "import cutlass.pipeline as pipeline and use pipeline.PipelineTmaUmma.create(...) instead",
    "cute.pipeline": "import cutlass.pipeline as pipeline and use that module directly",
    "cute._range": "use cutlass.range for CuTe DSL loops",
    "sm100_utils.make_smem_tensor_A": "this helper is not exposed",
    "sm100_utils.make_smem_tensor_B": "this helper is not exposed",
    "sm100_utils.make_trivial_pipeline": "use pipeline.PipelineTmaUmma.create(...) with the verified keyword-only signature",
    "sm100_utils.stage_input_A": "this helper is not exposed by blackwell_helpers",
    "sm100_utils.stage_input_B": "this helper is not exposed by blackwell_helpers",
    "sm100_utils.wait_pipeline": "use methods on a verified pipeline object",
    "sm100_utils.commit_pipeline": "use methods on a verified pipeline object",
    "sm100_utils.epilog_tmem_copy_and_partition": "import cutlass.utils as utils and use utils.epilog_tmem_copy_and_partition(...) instead",
    "sm100_utils.epilog_smem_copy_and_partition": "import cutlass.utils as utils and use utils.epilog_smem_copy_and_partition(...) instead",
    "sm100_utils.epilog_gmem_copy_and_partition": "import cutlass.utils as utils and use utils.epilog_gmem_copy_and_partition(...) instead",
    "sm100_utils.make_tiled_tma_atom_A": "use cute.nvgpu.make_tiled_tma_atom_A(...) instead",
    "sm100_utils.make_tiled_tma_atom_B": "use cute.nvgpu.make_tiled_tma_atom_B(...) instead",
    "sm100_utils._get_major_mode": "private helper is unavailable; use utils.LayoutEnum.from_tensor(tensor).mma_major_mode()",
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
    def __init__(
        self,
        candidate_mode: bool,
        kernel_names: set[str],
        kernel_required_args: dict[str, int],
        launched_kernel_calls: set[int],
    ) -> None:
        self.candidate_mode = candidate_mode
        self.kernel_names = kernel_names
        self.kernel_required_args = kernel_required_args
        self.launched_kernel_calls = launched_kernel_calls
        self.errors: list[str] = []
        self.calls: set[str] = set()
        self.tma_copy_nodes: list[ast.Call] = []
        self.function_kinds: list[str] = []
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
        if (
            self.candidate_mode
            and module.startswith("cutlass.cute.nvgpu")
        ):
            invalid = sorted(
                alias.name
                for alias in node.names
                if alias.name in {"OperandMajorMode", "OperandSource"}
            )
            if invalid:
                self._error(
                    node,
                    "release-incompatible nvgpu import: "
                    + ", ".join(invalid)
                    + "; derive major modes with utils.LayoutEnum and use "
                    "the verified make_trivial_tiled_mma form",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._error(node, f"dunder attribute access is forbidden: {node.attr}")
        name = qualified_name(node)
        if self.candidate_mode and name:
            for unavailable, replacement in UNAVAILABLE_CUTE_APIS.items():
                if name == unavailable or name.startswith(f"{unavailable}."):
                    self._error(
                        node,
                        f"API is unavailable on the B300 CUTLASS baseline: "
                        f"{unavailable}; {replacement}",
                    )
                    break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = qualified_name(node.func)
        if self.candidate_mode and name == "cute.struct.MemRange":
            self._error(
                node,
                "cute.struct.MemRange[...] is a @cute.struct field "
                "annotation, not a constructor; instantiate the enclosing "
                "struct with utils.SmemAllocator inside @cute.kernel",
            )
        if (
            self.candidate_mode
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get_thread_slice", "get_coord"}
        ):
            self._error(
                node,
                f"unverified CuTe method {node.func.attr}(); for dense "
                "TiledMma use tiled_mma.get_slice(0) followed by "
                "ThrMma.partition_A/B/C, and derive CTA coordinates from "
                "cute.arch.block_idx()",
            )
        if (
            self.candidate_mode
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_slice"
            and not node.args
            and not node.keywords
        ):
            self._error(
                node,
                "zero-argument get_slice() is not part of the verified B300 "
                "recipes; use tiled_mma.get_slice(0) for the collective MMA "
                "or tiled_copy.get_slice(thread_idx) for a per-thread copy",
            )
        if (
            self.candidate_mode
            and name == "cute.arch.alloc_smem"
            and (not self.function_kinds or self.function_kinds[-1] != "kernel")
        ):
            self._error(
                node,
                "cute.arch.alloc_smem is a device operation and must be "
                "called inside @cute.kernel, not from @cute.jit host code",
            )
        if (
            self.candidate_mode
            and name in self.kernel_required_args
            and id(node) in self.launched_kernel_calls
        ):
            provided = len(node.args) + sum(
                keyword.arg is not None for keyword in node.keywords
            )
            required = self.kernel_required_args[name]
            if provided < required:
                self._error(
                    node,
                    f"@cute.kernel {name} requires {required} bound "
                    f"arguments before .launch(...); got {provided}",
                )
        if (
            self.candidate_mode
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "launch"
            and qualified_name(node.func.value) in self.kernel_names
        ):
            kernel_name = qualified_name(node.func.value)
            self._error(
                node,
                f"@cute.kernel {kernel_name} must be bound to its arguments "
                f"before launch; use {kernel_name}(...).launch(...) rather "
                f"than {kernel_name}.launch(...)",
            )
        if (
            self.candidate_mode
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "launch"
            and isinstance(node.func.value, ast.Call)
            and qualified_name(node.func.value.func) in self.kernel_names
        ):
            launch_keywords = {
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
            }
            missing_launch = sorted({"grid", "block"} - launch_keywords)
            if missing_launch:
                self._error(
                    node,
                    "bound @cute.kernel launch requires explicit grid= and "
                    "block=; missing: " + ", ".join(missing_launch),
                )
        if name:
            self.calls.add(name)
            if (
                self.candidate_mode
                and name == "cute.copy"
                and any(
                    keyword.arg == "tma_bar_ptr" for keyword in node.keywords
                )
            ):
                self.tma_copy_nodes.append(node)
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
            if self.candidate_mode and name == "cute.gemm":
                required = ("atom", "d", "a", "b", "c")
                positional = min(len(node.args), len(required))
                provided_keywords = {
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                missing = [
                    argument
                    for argument in required[positional:]
                    if argument not in provided_keywords
                ]
                if missing:
                    self._error(
                        node,
                        "cute.gemm requires (atom, d, a, b, c); missing: "
                        + ", ".join(missing),
                    )
                unexpected_keywords = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                    and keyword.arg not in required
                )
                if len(node.args) > len(required) or unexpected_keywords:
                    details: list[str] = []
                    if len(node.args) > len(required):
                        details.append(
                            f"{len(node.args)} positional arguments"
                        )
                    if unexpected_keywords:
                        details.append(
                            "unexpected keywords: "
                            + ", ".join(unexpected_keywords)
                        )
                    self._error(
                        node,
                        "cute.gemm accepts exactly (atom, d, a, b, c); "
                        + "; ".join(details),
                    )
            if self.candidate_mode and name == "cute.make_layout":
                if len(node.args) != 1:
                    self._error(
                        node,
                        "cute.make_layout requires exactly one positional "
                        "shape argument; pass stride= as a keyword",
                    )
            if self.candidate_mode and name == "cute.make_tensor":
                if len(node.args) != 2 or node.keywords:
                    self._error(
                        node,
                        "cute.make_tensor requires exactly (pointer, layout); "
                        "it does not allocate storage from a layout and dtype",
                    )
                elif node.args:
                    pointer_arg = node.args[0]
                    layout_arg = node.args[1]
                    pointer_call = (
                        qualified_name(pointer_arg.func)
                        if isinstance(pointer_arg, ast.Call)
                        else None
                    )
                    layout_name = qualified_name(layout_arg)
                    if pointer_call == "cute.make_layout":
                        self._error(
                            node,
                            "cute.make_tensor first argument must be a real "
                            "Pointer; cute.make_layout(...) is not backing "
                            "storage",
                        )
                    if layout_name and (
                        layout_name.startswith("cutlass.Float")
                        or layout_name.startswith("cutlass.Int")
                    ):
                        self._error(
                            node,
                            "cute.make_tensor second argument must be a "
                            "Layout, not a cutlass scalar dtype",
                        )
            if (
                self.candidate_mode
                and name == "sm100_utils.make_trivial_tiled_mma"
            ):
                if len(node.args) < 6:
                    self._error(
                        node,
                        "sm100_utils.make_trivial_tiled_mma requires six "
                        "positional arguments: (a_dtype, a_major, b_major, "
                        "accumulator_dtype, cta_group, mma_tiler_mn)",
                    )
                unsupported = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg not in {None, "loc", "ip"}
                )
                if unsupported:
                    self._error(
                        node,
                        "sm100_utils.make_trivial_tiled_mma has only been "
                        "verified with six positional arguments; unsupported "
                        "keywords: " + ", ".join(unsupported),
                    )
            if self.candidate_mode and name == "cute.full":
                required = ("shape", "fill_value", "dtype")
                positional = min(len(node.args), len(required))
                provided_keywords = {
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                missing = [
                    argument
                    for argument in required[positional:]
                    if argument not in provided_keywords
                ]
                if missing:
                    self._error(
                        node,
                        "cute.full requires (shape, fill_value, dtype); "
                        "missing: " + ", ".join(missing),
                    )
            if self.candidate_mode and name == "pipeline.CooperativeGroup":
                agent_arg = node.args[0] if node.args else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "agent"
                    ),
                    None,
                )
                agent_name = qualified_name(agent_arg) if agent_arg else None
                if not agent_name or not agent_name.startswith("pipeline.Agent."):
                    self._error(
                        node,
                        "Pipeline CooperativeGroup first argument must be a "
                        "pipeline.Agent enum such as pipeline.Agent.Thread; "
                        "pass the thread count as size=, never as agent=",
                    )
            if (
                self.candidate_mode
                and name == "pipeline.PipelineTmaUmma.create"
            ):
                barrier_keyword = next(
                    (
                        keyword
                        for keyword in node.keywords
                        if keyword.arg == "barrier_storage"
                    ),
                    None,
                )
                if barrier_keyword is None or (
                    isinstance(barrier_keyword.value, ast.Constant)
                    and barrier_keyword.value.value is None
                ):
                    self._error(
                        node,
                        "PipelineTmaUmma.create requires an explicit non-None "
                        "shared-memory barrier_storage pointer on the B300 "
                        "baseline",
                    )
            if self.candidate_mode and name in {"cute.Shape", "cute.Tile"}:
                self._error(
                    node,
                    f"{name} is a typing union and cannot be instantiated; "
                    "use a plain tuple for static shapes",
                )
            if self.candidate_mode and name in self.kernel_names:
                tensor_ssa_args = [
                    keyword.arg or "<positional>"
                    for keyword in node.keywords
                    if isinstance(keyword.value, ast.Call)
                    and qualified_name(keyword.value.func)
                    in {"cute.full", "cute.full_like"}
                ]
                if tensor_ssa_args:
                    self._error(
                        node,
                        "cute.full/full_like returns TensorSSA and cannot be "
                        "bound to a cute.Tensor kernel argument; affected "
                        "arguments: " + ", ".join(tensor_ssa_args),
                    )
                invalid_keywords = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg == "compile_only"
                )
                if invalid_keywords:
                    self._error(
                        node,
                        f"@cute.kernel {name} does not accept compile_only; "
                        "bind only declared kernel arguments before .launch()",
                    )
            if self.candidate_mode and name.endswith(".make_smem_A"):
                self._error(
                    node,
                    "TiledMma.make_smem_A is unavailable on the B300 baseline",
                )
            if self.candidate_mode and name.endswith(".numel"):
                self._error(
                    node,
                    "Tensor.numel() is unavailable; use cute.size(tensor)",
                )
            if (
                self.candidate_mode
                and name.endswith(".load")
                and (node.args or node.keywords)
            ):
                self._error(
                    node,
                    "CuTe tensor load() takes no value argument; use "
                    "tensor.load() and transform the returned value",
                )
            if self.candidate_mode and any(
                marker in name
                for marker in (
                    ".partition_A.store",
                    ".partition_B.store",
                    ".partition_C.store",
                )
            ):
                self._error(
                    node,
                    "partition_A/B/C are methods that return tensor views; "
                    "they do not expose a .store method. Store through a "
                    "real destination tensor or a verified copy partition",
                )
            if (
                self.candidate_mode
                and name != "cute.cosize"
                and name.endswith(".cosize")
            ):
                self._error(
                    node,
                    "layout.cosize() is unavailable; use cute.cosize(layout)",
                )
            if (
                self.candidate_mode
                and name in self.kernel_names
                and id(node) not in self.launched_kernel_calls
            ):
                self._error(
                    node,
                    f"@cute.kernel {name} is bound but not launched; use "
                    f"{name}(...).launch(grid=..., block=..., smem=...)",
                )
            if (
                self.candidate_mode
                and name.endswith(".make_trivial_tiled_mma")
            ):
                invalid_keywords = sorted(
                    {
                        keyword.arg
                        for keyword in node.keywords
                        if keyword.arg in {"m", "n", "k"}
                    }
                )
                if invalid_keywords:
                    self._error(
                        node,
                        "make_trivial_tiled_mma does not accept tile-size "
                        "keywords on CUTLASS 4.6.1: "
                        + ", ".join(invalid_keywords),
                    )
            if (
                self.candidate_mode
                and name
                in {
                    "cute.arch.block_dim",
                    "cute.arch.block_idx",
                    "cute.arch.grid_dim",
                    "cute.arch.thread_idx",
                }
                and node.args
            ):
                self._error(
                    node,
                    f"{name} takes no positional arguments and returns an "
                    "(x, y, z) tuple",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.candidate_mode and isinstance(node.value, ast.Call):
            name = qualified_name(node.value.func)
            if name in {
                "cute.nvgpu.make_tiled_tma_atom_A",
                "cute.nvgpu.make_tiled_tma_atom_B",
            } and any(
                isinstance(target, (ast.Tuple, ast.List))
                for target in node.targets
            ):
                self._error(
                    node,
                    f"{name} returns one TmaInfo object on CUTLASS 4.6.1; "
                    "do not unpack it, use .atom and .tma_tensor",
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            self.candidate_mode
            and self.function_kinds
            and self.function_kinds[-1] == "jit"
        ):
            self._error(
                node,
                "do not construct Python storage classes inside @cute.jit; "
                "define a @cute.struct at module scope and allocate it with "
                "utils.SmemAllocator inside @cute.kernel",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        name = qualified_name(node.value)
        if self.candidate_mode and name in self.kernel_names:
            self._error(
                node,
                f"CUDA-style {name}[grid](...) launch syntax is invalid for "
                f"CuTe DSL; use {name}(...).launch(grid=..., block=...)",
            )
        if self.candidate_mode and name == "cute.Shape":
            self._error(
                node,
                "cute.Shape is a typing union, not a generic constructor; "
                "use a plain tuple for grid/block shapes",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            self.candidate_mode
            and node.id == "_"
            and isinstance(node.ctx, ast.Load)
        ):
            self._error(
                node,
                "reading bare '_' is rejected by the CUTLASS DSL preprocessor; "
                "use an explicit CuTe coordinate/slice expression",
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if (
            self.candidate_mode
            and isinstance(node.test, ast.Call)
            and qualified_name(node.test.func) == "cute.arch.elect_one"
        ):
            self._error(
                node,
                "cute.arch.elect_one() is a context manager; use "
                "'with cute.arch.elect_one():' rather than an if condition",
            )
        if any(
            isinstance(child, ast.Name) and child.id == "__name__"
            for child in ast.walk(node.test)
        ):
            self._error(
                node,
                "if __name__ guard is forbidden; call main() directly",
            )
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        elects_one = any(
            isinstance(item.context_expr, ast.Call)
            and qualified_name(item.context_expr.func) == "cute.arch.elect_one"
            for item in node.items
        )
        if self.candidate_mode and elects_one:
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if qualified_name(child.func) != "cute.copy":
                    continue
                if any(keyword.arg == "tma_bar_ptr" for keyword in child.keywords):
                    self._error(
                        child,
                        "TMA cute.copy(..., tma_bar_ptr=...) already elects "
                        "one issuing thread; keep the copy outside "
                        "'with cute.arch.elect_one():' to avoid deadlock",
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
            if self.candidate_mode:
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        self._error(
                            child,
                            "early return is forbidden inside @cute.kernel; "
                            "predicate work instead",
                        )
        if "cute.jit" in decorators:
            self.cute_jit_count += 1
        kind = (
            "kernel"
            if "cute.kernel" in decorators
            else "jit"
            if "cute.jit" in decorators
            else "function"
        )
        self.function_kinds.append(kind)
        try:
            self.generic_visit(node)
        finally:
            self.function_kinds.pop()


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


def _cute_kernel_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = {
            name
            for decorator in node.decorator_list
            if (name := qualified_name(decorator)) is not None
        }
        if "cute.kernel" in decorators:
            names.add(node.name)
    return names


def _cute_kernel_required_args(module: ast.Module) -> dict[str, int]:
    required: dict[str, int] = {}
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = {
            name
            for decorator in node.decorator_list
            if (name := qualified_name(decorator)) is not None
        }
        if "cute.kernel" not in decorators:
            continue
        positional = len(node.args.posonlyargs) + len(node.args.args)
        required[node.name] = positional - len(node.args.defaults)
    return required


def _decorated_function_names(module: ast.Module, decorator_name: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = {
            name
            for decorator in node.decorator_list
            if (name := qualified_name(decorator)) is not None
        }
        if decorator_name in decorators:
            names.add(node.name)
    return names


def _launched_kernel_calls(module: ast.Module) -> set[int]:
    launched: set[int] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "launch"
            and isinstance(function.value, ast.Call)
        ):
            launched.add(id(function.value))
    return launched


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

    visitor = _SubmissionVisitor(
        candidate_mode,
        _cute_kernel_names(module),
        _cute_kernel_required_args(module),
        _launched_kernel_calls(module),
    )
    visitor.visit(module)
    errors.extend(visitor.errors)
    if visitor.tma_copy_nodes and not any(
        name.endswith(".tma_partition") for name in visitor.calls
    ):
        warnings.append(
            "TMA cute.copy(..., tma_bar_ptr=...) was found without a "
            "tma_partition call; equal-shape manual views can compile but "
            "raise CUDA illegal instruction at launch"
        )

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

    entrypoint_jit = policy.get("entrypoint_jit")
    if isinstance(entrypoint_jit, str):
        jit_names = _decorated_function_names(module, "cute.jit")
        if entrypoint_jit not in jit_names:
            errors.append(
                f"evaluator entry point {entrypoint_jit} must be defined and "
                "decorated with @cute.jit"
            )

    for required_call in policy["required_calls"]:
        observed = required_call in visitor.calls
        if required_call == "cpasync.tma_partition":
            observed = observed or (
                "cute.nvgpu.cpasync.tma_partition" in visitor.calls
            )
        if not observed:
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
