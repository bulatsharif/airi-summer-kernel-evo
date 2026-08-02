# API, binding, and tracing errors

## Required entry point is not JIT-decorated

Symptoms:

```text
Function ... is not decorated with jit decorator
required entry point must remain @cute.jit
```

Inspect:

- exact function name from the task;
- `@cute.jit` remains immediately attached;
- signature and argument order are preserved;
- `ModelNew.forward` points to that function;
- no later definition shadows it.

Fix the interface before inspecting kernel internals.

## Failed to bind arguments

Symptoms name a decorated function and expected signature.

Compare:

| Expected | Actual |
| --- | --- |
| positional argument count | bound arguments |
| keyword names | supplied keywords |
| `Constexpr` versus runtime | argument objects |
| pointer/tensor annotation | runtime adapter result |
| kernel binding | all declared kernel arguments |

Do not remove an argument merely to silence binding when the harness contract
requires it.

## Unknown attribute or namespace

Common observed hallucinations:

| Invented or misplaced spelling | Local API fact |
| --- | --- |
| `blackwell_helpers.SmemAllocator` | allocator lives under `cutlass.utils` |
| `cute.arch.num_threads` | use `block_dim()` and derive the needed count |
| `TiledMma.get_slice_in_stage` | public slice method is `get_slice` |
| `cute.convert`, `cute.cast` | use public scalar type constructors |
| `cute.maximum`, `cute.fmax` | express compatible scalar comparisons/selection |
| `cute.constexpr` | compile-time helpers live under `cutlass` |
| `cute.pipeline` | import `cutlass.pipeline` |
| tensor `.partition_D` | partition method belongs to a ThrCopy/ThrMma object |
| `utils.SharedStorage` | define a `@cute.struct` storage type explicitly |

Do not blindly accept a compiler's similarly named suggestion. Verify that its
parameter and return-object roles match the call site.

## TMA namespace confusion

There are two related surfaces:

```text
cpasync.make_tiled_tma_atom          generic factory
cute.nvgpu.make_tiled_tma_atom_A/B  MMA-aware operand factories
```

The specialized A/B factories do not live under `cpasync`; the generic factory
does. Their construction roles and tiling transformations differ. Choose from
the required abstraction, not from name similarity.

## Static value used as a runtime object

Symptoms:

- Python `int` unexpectedly has `.value`;
- branch or operation expects a DSL scalar;
- a static tuple is treated like an IR aggregate.

Identify where the value is created. Use a DSL scalar constructor only when
generated arithmetic is required. Do not wrap every constant blindly; Python
needs static values to construct layouts and types.

## Runtime proxy used as a Python integer

Symptoms:

```text
ArithValue object cannot be interpreted as an integer
runtime value cannot index tuple/list
range_constexpr requires constexpr
```

Trace the value's first use in Python structure:

- tuple/list index;
- static loop bound;
- shape rank;
- class/type selection;
- Python allocation length.

Keep structure static or move the operation into generated runtime control
flow. Casting one runtime integer type to another does not make it Python
static.

## Dynamic control-flow rejection

Symptoms:

```text
early exit is not allowed
continue not properly in loop
value unavailable outside dynamic region
branch changes variable type
```

Repair principles:

- replace early exit with a predicate carried through the remaining body;
- initialize compatible values before the region;
- keep both branches' result types equal;
- avoid mutation of Python containers in runtime regions.

Do not convert a runtime condition to `const_expr` unless it is truly static.

## Wrong object kind

Frequent confusions:

```text
Layout versus Tensor
Pointer versus Tensor
TensorSSA versus pointer-backed Tensor
TmaInfo versus tuple
TiledCopy versus ThrCopy
TiledMma versus ThrMma
pipeline object versus participant/token
```

At the failure, write:

```text
expected kind:
actual type:
constructor that produced it:
method that should own the next operation:
```

Repair the first wrong constructor or method owner.

## DLPack direction confusion

`from_dlpack` consumes a framework object implementing the producer protocol
and returns a CuTe runtime tensor. The returned internal `_Tensor` is not
necessarily a framework DLPack producer. Do not try to reconvert it through
`.__dlpack__()` unless the API explicitly supports that direction.

## Symbol does not exist on the module you reached for

Symptoms:

```text
AttributeError: module 'cutlass' has no attribute ...
AttributeError: module 'cutlass.cute' has no attribute ...
AttributeError: module 'cutlass.cute.nvgpu.tcgen05' has no attribute ...
```

This is the most frequent failure in this DSL, and it is almost never a missing
import: the module imported cleanly and the attribute simply is not part of the
Python API. It usually arrives by analogy — a helper that exists in CUTLASS C++,
or a name that would be natural in another framework, recalled from memory
rather than read.

Inspect:

- whether the packet's documentation names this construct at all;
- whether the operation belongs on the module you used, on the tiled MMA
  object, or on a helper namespace;
- whether you invented a convenience wrapper for arithmetic the task does not
  need — dimensions that divide evenly need no rounding helper, plain integer
  division suffices.

Prefer a spelling the documentation shows over one you remember. A second guess
from the same memory is rarely better than the first.

## Callable used as a subscript, or called with the wrong arity

Symptoms:

```text
TypeError: 'function' object is not subscriptable
TypeError: ... missing N required positional arguments
```

The symbol was found, so this is progress from the previous class: the remaining
error is shape of the call, not its existence. The first form indexes something
that must be called; the second supplies too few operands to an operation whose
signature is fixed.

Inspect:

- brackets where parentheses belong, and the reverse;
- every operand the operation requires, in order, including accumulator and
  destination arguments that are easy to omit;
- whether a factory must be constructed before the object it returns is used.

The documentation shows the full call form. Copy its argument order rather than
inferring one.

## Attribute errors on a module or enum

Symptoms:

```text
AttributeError: module 'cutlass.cute.nvgpu.tcgen05' has no attribute ...
AttributeError: type object 'LayoutEnum' has no attribute ...
```

Almost always a name carried over from CUTLASS C++ rather than an absent
feature, and almost always a case convention. Cycling through capitalisations
costs one evaluation per guess and rarely converges; the API sections of the
documentation list the exact members, so look the name up instead of varying it.

## Errors this device has actually produced

Every line below is a verbatim diagnostic from a real run on this hardware,
with the number of times it occurred. They are reproduced exactly, not
paraphrased, so a message can be matched against this list character for
character when it appears.

```text
[31x] Error Code: MmaF8F6F4Op error
[13x] cutlass.base_dsl.common.DSLOperationBuildError:
[11x] RuntimeError: validation failed: full_abs=8.928038, sample_abs=2.592082
[9x] AttributeError: type object 'LayoutEnum' has no attribute 'RowMajor'
[8x] TypeError: copy() missing 1 required positional argument: 'dst'
[5x] AttributeError: type object 'OperandMajorMode' has no attribute 'RowMajor'
[5x] AttributeError: module 'cutlass.cute' has no attribute 'launch'
[4x] AttributeError: module 'cutlass.cute' has no attribute 'make_smem_tensor'. Did you mean: 'make_rmem_tensor'?
[3x] AttributeError: module 'cutlass.cute' has no attribute 'LayoutEnum'
[3x] TypeError: gemm() missing 1 required positional argument: 'c'
[3x] TypeError: 'function' object is not subscriptable
[3x] AttributeError: 'function' object has no attribute 'launch'
[2x] AttributeError: type object 'LayoutEnum' has no attribute 'kRowMajor'
[2x] ValueError: tensor<ptr<f32, tmem, align<1>> o ((128,128),1,1):((65536,1),0,0)> doesn't support load and store
[2x] AttributeError: type object 'OperandMajorMode' has no attribute 'kRowMajor'
[2x] AttributeError: 'NoneType' object has no attribute 'shape'
[2x] TypeError: make_tensor() got an unexpected keyword argument 'dtype'
[2x] AttributeError: module 'cutlass.cute' has no attribute 'shared_array'
[2x] AttributeError: module 'cutlass.cute.nvgpu.tcgen05' has no attribute 'CTA_GROUP_ONE'
[2x] cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: Failed to bind arguments to function `fp8_gemm_kernel` with signature `(matrix_a: cutlass.cute.typing.Tensor, matrix_b_nk: cutlass.cute.typing.Tensor, output: cutlass.cute.typing.Tensor)`
[2x] AttributeError: module 'cutlass.cute.nvgpu.tcgen05' has no attribute 'CTA_GROUP'
[2x] AttributeError: module 'cutlass.cute.nvgpu.tcgen05' has no attribute 'CTA_Group'. Did you mean: 'CtaGroup'?
[1x] ValueError: unable to convert (128, 128) in type <class 'tuple'> to Numeric
[1x] ValueError: Expected source and destination tensors to have the same rank, but got 2 and 3
[1x] AttributeError: type object 'OperandMajorMode' has no attribute 'ROW'
[1x] ValueError: Operation creation failed
[1x] cutlass.base_dsl.common.DSLAstPreprocessorError: DSLAstPreprocessorError: Early exit (break) is not allowed in `fp8_gemm_kernel`
[1x] AttributeError: module 'cutlass.cute' has no attribute 'fragment'
[1x] TypeError: Tensor() takes no arguments
[1x] TypeError: gemm() missing 2 required positional arguments: 'b' and 'c'
[1x] AttributeError: module 'cutlass.cute' has no attribute 'Smem'
[1x] AttributeError: type object 'OperandMajorMode' has no attribute 'ROW_MAJOR'
[1x] AttributeError: type object 'CtaGroup' has no attribute 'OneCta'
[1x] AttributeError: type object 'CtaGroup' has no attribute 'OneCTA'
```

Read the list before inventing a spelling. A name that appears here has already
been tried on this device and does not resolve; trying it again costs an
evaluation and returns the same line. Several entries are the same mistake in
different clothes -- six spellings of a major mode, four of a CTA-group member --
and the API sections carry the forms that do resolve.

Three of these are not name problems and no respelling will move them: a rank
mismatch between the two sides of a copy, a Python tuple handed to something
that wanted a layout or a numeric type, and an early `break` inside a traced
kernel. Those require changing what was built, not what it was called.

`RuntimeError: validation failed: full_abs=..., sample_abs=..., out_abs=...`
means the kernel compiled, launched and ran, and only the numbers are wrong --
the furthest any failure gets. When `out_abs` is `0.000000` the output buffer was
never written, so look for a kernel that launched but stored nothing rather than
for an arithmetic error.
