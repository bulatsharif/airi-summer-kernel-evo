# Adapting the Level 2 GEMM + BiasAdd + ReLU task

This page maps the public task contract to the documented Blackwell flow. It
does not provide a complete candidate.

## Shapes and physical layout

```text
matrix_a:    [M,K] = [1024,8192]
matrix_b_nk: [N,K] = [8192,8192]
bias:        [N]   = [8192]
output:      [M,N] = [1024,8192]
```

The evaluator already stores the right operand as N-by-K. Do not transpose it
inside the CuTe candidate.

With a correctness-first `(BM,BN,BK) = (128,128,64)` tiler:

```text
M tiles = 1024 / 128 = 8
N tiles = 8192 / 128 = 64
K tiles = 8192 / 64 = 128
```

All three task dimensions are divisible by these tile dimensions, so the GEMM
grid does not need partial-tile predicates for this exact task.

## Precision contract

- A and B storage: `cutlass.Float8E4M3FN`.
- accumulator: `cutlass.Float32` in TMEM.
- bias and output: Float32.
- task result: `relu((FP8_GEMM * SCALE_A * SCALE_B) + bias)`.

Do not pass `cute.full(...)` as a fake tensor scale argument. `SCALE_A` and
`SCALE_B` are compile-time Python constants already present in the starter.
Convert their product to a Float32 scalar where the device epilogue uses it.

## Correctness-first decomposition

Two CuTe launches are allowed and easier to debug:

1. Dense FP8 GEMM writes the raw FP32 accumulator result to `output`.
2. An elementwise CuTe kernel rewrites every output element as:

   ```text
   max(output[row, column] * (SCALE_A * SCALE_B) + bias[column], 0)
   ```

The second kernel must cover all `M*N` elements and must be launched only after
the GEMM kernel in the same JIT execution order.

## GEMM host/JIT responsibilities

The `@cute.jit` entrypoint should:

- derive A/B major modes;
- construct the tiled MMA;
- construct staged A/B SMEM layouts;
- construct A/B `TmaInfo` descriptors;
- bind all kernel arguments;
- launch the GEMM grid with 128 threads;
- launch the elementwise grid.

It must not allocate SMEM/TMEM, create Python storage classes, compute a torch
reference, or define evaluator inputs.

## GEMM kernel responsibilities

The `@cute.kernel` GEMM body should:

- allocate `SharedStorage`, sA, and sB;
- allocate TMEM and create both pipelines;
- derive block coordinates and GMEM tiles;
- form MMA and TMA partitions;
- loop over all 128 K tiles;
- issue every hardware K block per loaded stage;
- wait for the accumulator;
- store the complete FP32 tile to GMEM;
- free TMEM.

## Elementwise kernel responsibilities

Use global indexing derived from tuple-valued `thread_idx`, `block_idx`, and
`block_dim`. Each valid linear index maps to:

```text
row = linear // N
column = linear % N
```

Load `output[row,column]` and `bias[column]`, apply scale, add, and ReLU, then
write the Float32 value back to the same output tensor.

## Recommended attempt sequence

1. Make the local policy check pass.
2. First remote goal: complete compilation.
3. Second goal: complete launch without barrier/TMEM faults.
4. Third goal: inspect numerical diagnostics.
5. Change only the failing layer; do not replace the verified dataflow after a
   later-stage error.
