# Candidate error atlas: CUTLASS CuTe DSL 4.6.1

Use this file only after a concrete local or remote failure. It summarizes
recurring errors observed in agent runs against the B300 worker.

## Dense FP8 GEMM

Start from `candidate-dense-gemm-template.py`. Do not reconstruct its TMA,
pipeline, TMEM, or epilogue code from memory.

| Diagnostic | Cause | Correction |
|---|---|---|
| `cute has no attribute constexpr` | Triton-like spelling | Use a normal trace-time Python constant; use `cutlass.Constexpr` only for an annotation. |
| `Expected a TensorSSA or Numeric(Float), but got ... ArithValue` while chaining scalar math | A CuTe math intrinsic returned an arithmetic SSA value that the next intrinsic will not accept directly | Follow `candidate-math-api.md` and materialize every nested intrinsic boundary with `cutlass.Float32(...)`. Do not invent a temporary tensor or layout for a scalar. |
| `SharedStorage` missing from `cute.struct` or `cutlass.utils` | Invented base class | Declare storage with `@cute.struct` and `cute.struct.MemRange`. |
| `make_tiled_tma_atom_A/B` missing from `cpasync` | Helper is in the parent namespace | Call `cute.nvgpu.make_tiled_tma_atom_A/B`. |
| `TmaOperandMajorMode` missing | Obsolete API family | Derive major modes with `utils.LayoutEnum.from_tensor(...).mma_major_mode()`. |
| Cannot unpack the TMA result | The helper returns one descriptor object | Keep `tma_a`/`tma_b`; pass `.atom` and `.tma_tensor`. |
| `producer_get_barrier` missing | Invented manual barrier protocol | Use the token from `acquire_and_advance()` and its `.barrier`. |
| `partition_D` missing on `ThrMma`, `TiledMma`, or tensor | Wrong owner | Call `partition_D` on `tmem_copy.get_slice(thread_idx)`. |
| Weakly congruent coordinate/layout error | Extra K/stage coordinate was added | Preserve the template's exact `tma_global_*[(None, empty_ab.count)]` indexing. |
| Invalid vector LHS in `autovec_copy` | Passing a loaded vector instead of the register tensor | Transform with `register_tensor.store(register_tensor.load() * scale)`, then copy the register tensor. |
| `CUDA unspecified launch failure` after template changes | Broken synchronization, TMEM lifetime, or indexing | Restore the exact template core. Change only task constants, names, output scale, and a separate elementwise kernel. |
| Validation error larger by orders of magnitude | Missing/doubled FP8 scale or bad output indexing | Apply `SCALE_A * SCALE_B` exactly once before adding FP32 bias; keep output two-dimensional. |
| Output looks zero/uninitialized although elementwise code runs | The GEMM device kernel was called without `.launch()`, or host-side construction was wrapped in another `@cute.kernel` | Keep `dense_fp8_gemm` as `@cute.jit`; call it from the public entrypoint. It alone constructs objects and launches `dense_fp8_gemm_kernel`. |

## Scalar and reduction kernels

| Symptom | Cause | Correction |
|---|---|---|
| Every lane sums the same values | Loop index ignores `thread_idx`/lane | Use `column = iteration * THREADS + thread_idx`. |
| A second warp reduction is 32 times too large | Every lane re-sums every SMEM partial before shuffling | Either give each lane at most one partial or use the verified one-warp reduction. |
| Correct scale gives a large error, removing it gives a huge error | Indexing bug misdiagnosed as dequantization | Keep the declared scale and audit row/column/channel indices. |
| `continue not properly in loop` | Unsupported control flow in lowered CuTe loop | Replace `continue` with nested predicates. |
| Remote execution timeout | Compile-time unrolling or a pathological kernel | Use `cutlass.range` for large loops and never retry the identical candidate. |
| `cutlass.relu`, `cute.fmax`, or `cute.maximum` missing | Invented activation helper | Use `summed * (summed > 0.0)` for ReLU. |

## Retry discipline

1. Preserve any candidate that reached remote execution.
2. Fix one diagnostic at a time.
3. Never replace a compiling dense GEMM core to fix a separate elementwise
   epilogue.
4. Do not retry an identical candidate after a launch failure or worker
   timeout.
5. If the same diagnostic appears twice, restore the compile-verified template
   and reapply only task constants and the separate epilogue.
6. Stop immediately after a harness `PASS`; do not tune in a correctness run.
