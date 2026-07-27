# CuTe DSL working model

CuTe DSL is a statically typed compiled Python subset. Python traces and
specializes IR; DSL values represent generated runtime code.

## Contents

[Boundaries](#boundaries) · [Static versus runtime](#static-versus-runtime) ·
[Tensors and framework boundary](#tensors-and-framework-boundary) ·
[Blackwell flow](#blackwell-flow)

## Boundaries

```python
@cute.kernel
def kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    ...


@cute.jit
def launch(a, b, c, stream):
    kernel(a, b, c).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads, 1, 1),
        cluster=(cluster_x, cluster_y, 1),
        stream=stream,
    )


executor = cute.compile(launch, a, b, c, stream)
executor(a, b, c, stream)
```

- `@cute.kernel`: device entry and explicit resources.
- `@cute.jit`: compiled host/device helper.
- `cutlass.Constexpr`: tile, stages, flags, layouts, or other static
  configuration.
- runtime arguments: addresses and values compatible with the compiled
  signature.

`cute.compile` compiles for example arguments and returns a reusable executor.
Keep compilation outside timing. Implicit JIT/file caches exist, but correctness
must not depend on a warm cache. Controls include `CUTE_DSL_CACHE_DIR` and
`CUTE_DSL_DISABLE_FILE_CACHING`; confirm installed spelling.

## Static versus runtime

Compile-time values construct layouts, tiled MMA/copy, storage, unrolled loops,
and feature branches. Runtime values index tensors and control supported IR
loops/branches.

| Need | Construct |
|---|---|
| fully unrolled fixed loop | `cutlass.range_constexpr` |
| runtime loop with hints | `cutlass.range` |
| ordinary runtime loop | `range` under JIT |
| compile-time branch | `cutlass.const_expr` |
| runtime predicate | `if` over DSL value |

Important limitations:

- runtime code cannot change the type/shape of Python tuples, lists, objects, or
  layouts
- a runtime DSL value generally cannot index a Python container
- dependent types are unsupported
- dynamic regions cannot use arbitrary early `return`, `break`, or `continue`
- `global`/`nonlocal`, reflection, generators, coroutines, and arbitrary object
  mutation are outside the kernel model
- layout coordinate algebra is 32-bit

All participants in copy/MMA/barrier collectives must take compatible runtime
control flow.

Kernel classes are useful for immutable compile-time configuration:

```python
class Gemm:
    def __init__(self, tile_mn, cluster_mn, stages, use_2cta):
        self.tile_mn = tile_mn
        self.cluster_mn = cluster_mn
        self.stages = stages
        self.use_2cta = use_2cta
```

Validate fields through `can_implement`; do not mutate them during compiled
execution.

## Tensors and framework boundary

A CuTe tensor is storage/pointer plus a layout mapping logical coordinates to
addresses. Shape alone does not imply row/column major. Preserve hierarchical
modes; mathematical B `[K,N]` is often passed physically as `[N,K,L]`.

Common operations:

| API | Role |
|---|---|
| `make_layout`, `make_tensor` | layout and tensor construction |
| `local_tile`, `slice_`, `tiled_divide` | tile/stage selection |
| `size`, `cosize`, `size_in_bytes` | logical/storage/byte extent |
| `make_copy_atom`, `make_tiled_copy` | copy definition |
| `partition_S`, `partition_D` | per-thread source/destination |

Torch storage can cross through implicit DLPack conversion or an explicit CuTe
view with layout/assumed alignment. Explicit conversion is still zero-copy and
is preferable when TMA, physical order, alignment, or specialization matters.
Keep the Torch owner alive and use the same CUDA stream. Mark dimensions
runtime-dynamic deliberately rather than baking an example size into a layout.

## Blackwell flow

Use `cutlass.cute.nvgpu.tcgen05` and installed helpers:

```text
tiled MMA + staged SMEM layouts
-> TMA atoms/descriptors
-> partition A/B/C
-> GMEM-to-SMEM producer pipeline
-> tcgen05 MMA into TMEM
-> TMEM-to-register epilogue
-> convert/store output
```

Do not reconstruct instruction descriptors from memory. Preserve ownership,
transaction bytes, barriers, and phase transitions as one protocol.

During debugging, Python `print` shows trace-time static values; `cute.printf`
shows device values. Debug/line-info flags alter generated code and timing.
Exact release-specific APIs come from installed examples.
