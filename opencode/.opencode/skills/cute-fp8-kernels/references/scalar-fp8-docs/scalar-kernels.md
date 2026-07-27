# Verified scalar CuTe kernel mechanics

## Imports and scalar types

```python
import cutlass
import cutlass.cute as cute
```

Use `cutlass.Float8E4M3FN` for FP8 E4M3FN storage and `cutlass.Float32` for
accumulation/output. Convert a loaded FP8 element before arithmetic:

```python
value = input_tensor[row, column].to(cutlass.Float32)
```

Store through tensor indexing:

```python
output_tensor[row, column] = value
```

Do not use `.ptr`, `.data_ptr()`, PyTorch, or manually extracted pointers for
ordinary global tensor elements.

## Device coordinates

These helpers live under `cute.arch`, accept no arguments, and return
`(x, y, z)` tuples:

```python
thread_x, _, _ = cute.arch.thread_idx()
block_x, _, _ = cute.arch.block_idx()
block_x_size, _, _ = cute.arch.block_dim()
linear = block_x * block_x_size + thread_x
```

There is no verified `cute.thread_id`, `cute.block_dim`, or axis-argument form
such as `cute.arch.thread_idx(0)`.

## Static loops and predicates

Use `cutlass.range_constexpr(STATIC_BOUND)` for small compile-time loops. Use
ordinary integer `+`, `-`, `*`, `//`, and `%` to decode a dynamic linear index.
CuTe DSL lowers device-dependent `if` predicates inside `@cute.kernel`.

Keep reductions in a local FP32 scalar and write the result once. When several
conditions guard a load, nested `if` statements are easier for the DSL frontend
than guessed helper functions.

## Kernel and launch syntax

```python
@cute.kernel
def scalar_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    thread_x, _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_idx()
    block_x_size, _, _ = cute.arch.block_dim()
    linear = block_x * block_x_size + thread_x
    if linear < TOTAL_ELEMENTS:
        row = linear // ROW_SIZE
        column = linear % ROW_SIZE
        value = input_tensor[row, column].to(cutlass.Float32)
        output_tensor[row, column] = value * SCALE


@cute.jit
def scalar_entry(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    scalar_kernel(input_tensor, output_tensor).launch(
        grid=((TOTAL_ELEMENTS + 127) // 128, 1, 1),
        block=(128, 1, 1),
    )
```

Bind every declared kernel argument before `.launch(...)`, and always provide
explicit `grid=` and `block=` keywords.

## Submission boundary

Candidate code must contain only reusable constants, CuTe structs/helpers,
`@cute.kernel` functions, and `@cute.jit` entrypoints. Do not define `main()`,
create tensors, call torch, or print validation markers; the evaluator appends
those parts after the candidate passes the local policy check.
