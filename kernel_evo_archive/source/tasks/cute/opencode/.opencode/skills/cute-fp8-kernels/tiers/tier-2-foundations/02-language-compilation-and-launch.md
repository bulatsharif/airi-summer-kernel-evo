# CuTe DSL language, compilation, and launch API

CuTe DSL is a compiled Python subset. Decorated Python functions are
preprocessed and traced into an intermediate representation, lowered toward
PTX and device code, then loaded and launched. Python remains the
metaprogramming language; DSL values represent generated runtime computation.

## The three compilation stages

1. **AST preprocessing.** The compiler rewrites supported loops, branches, and
   decorated function boundaries into structured regions.
2. **Tracing and partial evaluation.** Python executes with proxy arguments.
   Tensor operations emit IR; Python-static values are evaluated immediately.
3. **Lowering and execution.** The IR is optimized, lowered to target code,
   assembled, loaded, and launched.

This explains why a single function can contain both ordinary Python objects
and runtime DSL values. It also explains why an untaken Python branch may
vanish when preprocessing is disabled.

## Standard imports

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
```

Architecture-specific modules are documented in Tier II. A candidate should
use only imports already permitted by its harness policy.

## `@cute.jit`

`@cute.jit` declares a compiled host function or an inline compiled helper.

Decorator form:

```text
@cute.jit
@cute.jit(preprocess=True)
@cute.jit(preprocess=False)
```

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `preprocess` | `bool`, true | Rewrite supported Python control flow before tracing |

Use the installed `preprocess` spelling.

Call behavior:

- Python may call a `@cute.jit` function.
- A `@cute.jit` or `@cute.kernel` function may call a `@cute.jit` helper; the
  helper is inlined at compile time.
- Static arguments specialize generated code.
- Runtime arguments become parameters of the generated function.
- A top-level JIT call may use `no_cache=True` to force recompilation. Normal
  authoring should leave caching enabled.

## `@cute.kernel`

`@cute.kernel` declares a GPU kernel. It cannot be called directly from
ordinary Python and cannot invoke another `@cute.kernel`.

Decorator form:

```text
@cute.kernel
@cute.kernel(preprocess=True)
@cute.kernel(preprocess=False)
```

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `preprocess` | `bool`, true | Preserve supported dynamic loops and branches as structured IR |

A JIT function first binds all declared kernel arguments:

```python
bound_kernel = device_kernel(argument_0, argument_1)
```

The resulting bound object is launched:

```python
bound_kernel.launch(grid=grid, block=block)
```

Binding and launching are separate. Calling `device_kernel.launch(...)`
without binding declared arguments does not satisfy the kernel signature.

## Kernel launch parameters

```text
bound_kernel.launch(
    *,
    grid,
    block,
    cluster=None,
    smem=None,
    stream=None,
    fallback_cluster=None,
    max_number_threads=(0, 0, 0),
    min_blocks_per_mp=0,
    use_pdl=False,
    cooperative=False,
    smem_merge_branch_allocs=False,
    preferred_smem_carveout=None,
)
```

The exact bound method may expose a subset of these options. Supply only
options required by the design.

| Parameter | Meaning |
| --- | --- |
| `grid` | Three-dimensional number of CTAs to launch |
| `block` | Three-dimensional thread count for each CTA |
| `cluster` | Preferred three-dimensional CTA-cluster shape |
| `smem` | Dynamic shared-memory bytes; `None` permits allocator-based calculation |
| `stream` | CUDA stream on which the kernel is launched |
| `fallback_cluster` | Minimum cluster shape when the preferred shape cannot be scheduled |
| `max_number_threads` | Maximum thread dimensions emitted as a launch bound |
| `min_blocks_per_mp` | Minimum resident blocks per multiprocessor hint |
| `use_pdl` | Enable programmatic dependent launch |
| `cooperative` | Request a cooperative launch |
| `smem_merge_branch_allocs` | Permit mutually exclusive branches to reuse shared storage |
| `preferred_smem_carveout` | Shared-memory versus L1 carveout preference |

`grid`, `block`, and cluster shapes are plain three-element tuples or lists.
Their values belong to the implementation, so this documentation does not
provide defaults.

## Calling convention matrix

| Caller | Callee | Allowed | Result |
| --- | --- | --- | --- |
| Python | `@cute.jit` | yes | JIT compilation or cache lookup, then execution |
| Python | `@cute.kernel` | no | kernel must be launched through JIT code |
| `@cute.jit` | `@cute.jit` | yes | compile-time inline call |
| `@cute.jit` | Python helper | yes | helper executes while tracing |
| `@cute.jit` | `@cute.kernel` | yes | bind, then launch |
| `@cute.kernel` | `@cute.jit` | yes | compile-time inline helper |
| `@cute.kernel` | Python helper | yes | helper executes while tracing |
| `@cute.kernel` | `@cute.kernel` | no | dynamic parallelism is not this calling model |

## Arguments and specialization

Common annotation classes are:

```text
cutlass.Constexpr     compile-time Python value
cutlass.Int32         runtime signed integer
cutlass.Int64         runtime signed integer
cutlass.Float32       runtime floating-point scalar
cute.Tensor           typed tensor view
cute.Pointer          typed pointer
```

Use `Constexpr` for facts that construct types, layouts, instruction atoms, or
unrolled structure. Use runtime scalar annotations for coordinates, dynamic
dimensions, and values that may vary without recompilation.

The JIT cache key depends on the compiled function and the static/type
specialization of its arguments. Converting a genuinely dynamic dimension to a
Python integer may create one compiled specialization per observed value.

## Candidate and harness boundary

When the surrounding harness owns compilation and evaluation:

- preserve the required JIT entry point and its signature;
- keep device work in CuTe DSL;
- do not add input generation, an oracle, timing, or a `main()` function;
- do not call `cute.compile` inside the candidate;
- let the harness bind external framework objects to the candidate entry point.

These are interface rules, not an algorithm. The task statement specifies the
actual arguments and semantics.
