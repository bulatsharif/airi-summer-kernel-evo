# Control flow, assumptions, and diagnostic output

CuTe's default AST preprocessor converts supported Python control flow into
structured IR. Whether a condition or bound is Python-static or DSL-runtime
determines where it executes.

## Loop forms

| Form | Evaluation | Parameters |
| --- | --- | --- |
| `cutlass.range_constexpr(...)` | Python compile time, fully unrolled | static start/stop/step |
| `range(...)` | generated runtime loop under preprocessing | runtime-compatible start/stop/step |
| `cutlass.range(...)` | generated runtime loop | start/stop/step plus compiler controls |

`cutlass.range` may accept controls such as:

| Parameter | Meaning |
| --- | --- |
| `unroll` | requested unroll factor or policy |
| `unroll_full` | request complete unrolling when legal |
| `prefetch_stages` | request compiler software pipelining |

Software-pipelining controls alter generated scheduling; they do not create
hardware transaction barriers or repair a manually staged asynchronous
pipeline.

`range_constexpr` requires every bound to be known to Python during tracing.
Use it when each iteration constructs static code or types. Use a runtime range
when the trip count is a DSL value.

## Branches

```text
if cutlass.const_expr(static_predicate):
    ...  # Python compile-time branch

if runtime_predicate:
    ...  # structured runtime branch
```

`cutlass.const_expr(value)` requires a compile-time value. It removes untaken
code from the specialization. An ordinary `if` over a DSL Boolean emits both
structured regions.

The same distinction applies to `while`: a `const_expr` condition executes
during tracing; a DSL condition emits runtime control flow.

## Dynamic-region restrictions

Inside generated dynamic loops and branches:

- `break`, `continue`, early `return`, and exception raising are unsupported;
- a value first created inside a region is not automatically available after
  the region;
- a variable cannot change its Python/DSL type across branches or iterations;
- Python container shape and tuple nesting cannot change at runtime;
- a runtime proxy cannot drive reflection, imports, generators, or dynamic
  dispatch.

Keep Python metaprogramming outside dynamic regions. Express runtime selection
with values and compatible tensor operations rather than changing object
structure.

## Compile-time assumptions

```text
cute.assume(value, divby=None) -> value
```

`divby` is a positive Python integer stating that a runtime integer is
divisible by that value. The function returns the constrained value. The
assumption must be true for every launch; it is not a runtime check. This
release does not expose lower- or upper-bound keywords on `cute.assume`.

Use explicit task validation or predication when a condition is not guaranteed.
False assumptions may enable invalid addressing or code generation.

## Static inspection

Python output runs during tracing:

```python
print(layout)
print(tensor.shape, tensor.stride)
print(cute.pretty_str(fragment))
```

It can inspect static object kinds, layouts, shapes, strides, and
specialization choices. A runtime scalar prints as a proxy representation
rather than its device value.

## Device output

```text
cute.printf(format_or_value, *values)
```

Supported styles include:

```python
cute.printf(value)
cute.printf("index={}, value={}", index, value)
cute.printf("value=%.3f", value)
```

Device printing affects timing and can produce large output. Guard it to a
small participant subset and remove it before performance measurement.

## Compilation and runtime stages

Classify a failure by its earliest stage:

1. Python parse/import;
2. decorator argument binding;
3. AST preprocessing;
4. tracing and object/type construction;
5. IR verification/lowering;
6. PTX assembly;
7. kernel launch and resources;
8. asynchronous execution or memory access;
9. numerical validation;
10. performance measurement.

An error reported during a later synchronization can originate from an earlier
asynchronous launch. The earliest deterministic diagnostic is normally the
most useful evidence.

## API-use discipline

- Treat free functions, object methods, layouts, tensors, atoms, and
  participant slices as different kinds.
- Preserve the exact namespace and argument roles documented locally.
- Do not translate a CUDA C++, Triton, or framework idiom by guessing a CuTe
  spelling.
- Do not build further logic on an unverified return object.
- Compilation validates syntax and types; a real launch validates resources,
  address spaces, and synchronization.
