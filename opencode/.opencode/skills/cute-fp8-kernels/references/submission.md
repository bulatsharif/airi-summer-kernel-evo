# Single-file submission contract

Use one self-contained Python file for every remote correctness or performance
run. The file is the reproducible unit of work: it contains the CuTe DSL kernel,
host launch code, input construction, reference, checks, and any bounded timing.
No project-specific Python harness is required.

## Contents

- [Required shape](#required-shape)
- [Imports and environment](#imports-and-environment)
- [Kernel and host boundaries](#kernel-and-host-boundaries)
- [Torch and CuTe interoperation](#torch-and-cute-interoperation)
- [Compilation and execution](#compilation-and-execution)
- [Correctness output](#correctness-output)
- [Timing output](#timing-output)
- [Remote submission](#remote-submission)
- [What must fail the process](#what-must-fail-the-process)
- [Minimal structural template](#minimal-structural-template)
- [Final checklist](#final-checklist)

## Required shape

The submitted file must:

1. import every dependency it uses
2. define the CuTe DSL implementation in the same file
3. define `main()`
4. call `main()` under the standard module guard
5. produce a nonzero exit code when compilation, execution, or validation fails

The remote service receives a file, not the surrounding repository. Do not
depend on adjacent modules, generated artifacts, a current working directory,
or an earlier submission.

The file may contain several classes and functions. “Single file” does not mean
“single function.”

## Imports and environment

Keep imports explicit and near the top. A typical Blackwell submission needs a
subset of:

```python
import math
import statistics
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.runtime import make_ptr
from cutlass.cute import pipeline
from cutlass.cute.nvgpu import tcgen05
```

Exact import paths vary with the installed CUTLASS release. Inspect the
installed package or its shipped examples when an import is uncertain. Do not
add package installation commands to the submission.

Set a deterministic Torch seed in `main()`. Print the device, compute
capability, Torch version, CUDA version, and CUTLASS version only when diagnosing
compatibility; routine benchmark submissions should keep output compact.

## Kernel and host boundaries

Use the boundary that matches CuTe DSL:

- `@cute.kernel` contains device code and is launched with explicit grid, block,
  optional cluster, dynamic shared-memory size when required, and stream.
- `@cute.jit` contains compiled host setup or reusable device helpers.
- normal Python in `main()` creates Torch inputs, constructs runtime arguments,
  invokes `cute.compile`, checks results, and measures repeated execution.
- values that specialize code are `cutlass.Constexpr` arguments or fields
  captured in a JIT-compatible kernel object.

Keep the kernel configuration visible. A reader should be able to find:

- CTA and MMA tile
- cluster shape and one-CTA/two-CTA instruction mode
- pipeline stage count
- input, accumulator, and output types
- alignment and tail-shape assumptions
- dense or block-scaled path

Before launching, expose a `can_implement(...)` check when the implementation
has shape, alignment, layout, or datatype restrictions. Its failure should
explain the violated contract.

## Torch and CuTe interoperation

Torch owns the allocation and can provide the correctness reference. CuTe owns
the submitted implementation.

Two common boundary styles are:

1. pass a Torch tensor through the supported DLPack conversion at the JIT
   boundary
2. create a CuTe tensor from its data pointer and an explicit layout

Prefer explicit conversion when physical layout, assumed alignment, or static
shape affects code generation. Preserve the Torch tensor object for the entire
kernel invocation so its storage remains alive.

A CUDA stream must refer to the same context as the Torch allocations. In
releases using CUDA Python driver handles, the conversion commonly follows this
shape:

```python
from cuda import cuda

stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
```

Confirm the exact stream type in the installed examples. Do not silently launch
on an unrelated stream and then read the result from Torch without
synchronization.

## Compilation and execution

Compile after representative tensors, layouts, configuration, and stream have
been created:

```python
executor = cute.compile(jit_entry, a, b, c, stream)
executor(a, b, c, stream)
torch.cuda.synchronize()
```

`cute.compile` is part of setup, not kernel latency. Reuse the executor for:

- the first functional launch
- warmups
- correctness cases compatible with the compiled signature
- timed iterations

If a shape or configuration is static, compile each distinct specialization
deliberately and label its result. Do not hide recompilation inside the timing
loop.

Keep allocations and reference computation outside a kernel-only timing
interval. End-to-end timing is allowed only when the task explicitly asks for
it, and it must be labeled as such.

## Correctness output

At minimum, print one compact machine-readable line per test case:

```text
case M=1024 N=1024 K=1024 L=1 passed=true max_abs=... max_rel=...
```

For a scaled operation, also identify the format, scale convention, and
granularity. If non-finite values are forbidden, test and report their count.

Use `torch.testing.assert_close` or explicit assertions after reporting metrics.
An assertion makes the remote process fail, which is preferable to printing
`passed=false` and returning exit code zero.

The reference must use the actual quantized operands. Follow `correctness.md`;
do not compare only with a matmul over the original unquantized data.

## Timing output

After correctness passes:

1. warm up the compiled executor
2. synchronize
3. run a bounded number of timed repetitions
4. synchronize at the correct boundaries
5. print the timing method, iteration count, and a robust statistic

Example result format:

```text
timing method=cuda_event warmup=10 iterations=50 median_ms=... min_ms=...
throughput tflops=...
```

Do not emit a large line for every iteration. Keep enough summary data to
compare candidates without overwhelming the remote response.

## Remote submission

The parent `AGENTS.md` defines the current service URL, authentication header,
form fields, response schema, and profile download command. It is the authority
for transport details.

Submit the file with the API key read from `CUTE_HARNESS_API_KEY`. Never print,
copy into the Python file, or commit the key.

The normal sequence is:

1. save or update one submission file
2. run local syntax checks that do not require a GPU
3. submit with the requested profiler
4. inspect `success`, `exit_code`, `stdout`, `stderr`, and `timed_out`
5. inspect `profile_error` independently of process correctness
6. download the profile only if it answers a concrete performance question

A profiler failure does not prove kernel failure, and kernel success does not
prove the profiler result is meaningful. Treat the two channels separately.

## What must fail the process

Let these conditions raise or explicitly exit nonzero:

- missing required CUDA/CUTLASS capability
- unsupported dtype, shape, layout, or alignment
- JIT or PTX assembly failure
- kernel launch failure
- illegal memory access
- non-finite output when not allowed
- correctness tolerance failure
- absent native-path evidence when the task explicitly requires it

Do not use broad `try/except Exception` blocks that turn these failures into
informational prints. A narrow exception may add context, but it must re-raise.

Do not fall back from a failed CuTe kernel to `torch.matmul`. The reference is
not a substitute implementation.

## Minimal structural template

This template establishes the submission shape. It is intentionally not a
complete GEMM and should not be submitted without replacing the placeholders.

```python
import torch
import cutlass
import cutlass.cute as cute


class Kernel:
    def __init__(
        self,
        tile_mn: tuple[int, int],
        stages: int,
    ):
        self.tile_mn = tile_mn
        self.stages = stages

    def can_implement(self, a, b, c) -> tuple[bool, str]:
        # Check the actual dtype/layout/alignment/shape contract.
        return True, ""

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        # Partition tensors, run TMA/MMA pipelines, and store the result.
        ...

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
        self.kernel(a, b, c).launch(
            grid=(1, 1, 1),       # derive from problem and tile
            block=(128, 1, 1),    # derive from implementation
            stream=stream,
        )


def make_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Use actual quantized values and the operation's exact scale semantics.
    return a.float() @ b.float()


def main() -> None:
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    # Allocate deterministic inputs and output.
    a = ...
    b = ...
    c = ...
    stream = ...

    operation = Kernel(tile_mn=(128, 128), stages=3)
    supported, reason = operation.can_implement(a, b, c)
    if not supported:
        raise RuntimeError(reason)

    executor = cute.compile(operation, a, b, c, stream)
    executor(a, b, c, stream)
    torch.cuda.synchronize()

    reference = make_reference(a, b)
    torch.testing.assert_close(c, reference, atol=..., rtol=...)
    print("correctness passed=true")


if __name__ == "__main__":
    main()
```

## Final checklist

- [ ] The file is self-contained.
- [ ] `main()` and the module guard are present.
- [ ] The implementation is CuTe DSL Python.
- [ ] PyTorch is confined to allocation, input preparation, reference, and
      measurement.
- [ ] Configuration and unsupported cases are explicit.
- [ ] The executor is compiled once per specialization and reused.
- [ ] Compilation and allocation are outside kernel-only timing.
- [ ] Correctness failures produce a nonzero process exit.
- [ ] No API key, secret, machine-specific path, or downloaded artifact is in
      the file.
- [ ] Output states what was tested and how it was measured.
