# Hopper GEMM construction map

Read this map before opening `kernel.py`. Open the full implementation only for the section required by the current hypothesis.

## Which section answers which question

| Decision | Read in `kernel.py` | Why |
|---|---|---|
| dtype, WGMMA atom, tile, cluster | `HopperWgmmaGemmKernel._setup_attributes` | Couples operand major mode, instruction shape, atom layout, cluster layout, and epilogue tile. These choices are not independent. |
| host ABI and static specialization | `HopperWgmmaGemmKernel.__call__` | Builds tensors, TMA descriptors, grid, dynamic SMEM, and the launch. Cache all properties that remain static here. |
| TMA/WGMMA pipeline | `HopperWgmmaGemmKernel.kernel` | Contains barrier initialization, TMA partitioning, producer/consumer stage state, WGMMA fence/commit/wait, and stage release. Borrow this protocol as a unit. |
| stage count and SMEM | `_compute_stages` and `_make_smem_layouts` | Stage count changes both shared-memory capacity and the lifetime assumed by the mainloop. |
| cluster multicast | `_compute_grid` and `_make_tma_atoms_and_tensors` | Multicast legality depends on the CTA layout, cluster coordinate, and operand reuse direction. |
| epilogue/TMA store | `_make_tma_store_atoms_and_tensors` plus the kernel epilogue | Accumulators convert in registers, move through a compatible SMEM tile, then use a TMA store pipeline. |
| compile, reference, timing | `run` | Separates `cute.compile`, validates decoded FP8 values, and benchmarks the reused executor. |

## Dataflow

```text
BF16 or FP8 GMEM A/B
  -> TMA descriptor + cluster partition
  -> staged WGMMA-compatible SMEM
  -> producer transaction barrier
  -> WGMMA fence / issue / commit / bounded wait
  -> Float32 accumulator registers
  -> fused conversion/epilogue
  -> swizzled SMEM C tile
  -> TMA store pipeline
  -> output GMEM
```

## Synchronization invariant

For each mainloop stage, the producer acquires the stage, issues both operand TMA transfers with the exact transaction-byte count, and commits it. The consumer waits for that phase, fences WGMMA operands, issues and waits for every read of the stage, then releases it. Moving a release earlier can produce silent corruption or a hang even when tensor shapes still look correct.

## Evidence expected after the barrier

- correctness for the configured dtype and representative/tail shapes;
- WGMMA, TMA, and mbarrier instruction families;
- zero local loads/stores caused by spills;
- exact registers and dynamic/static shared memory;
- memcheck before synccheck after movement or barrier changes;
- timing from the reused non-debug executor, including required FP8 conversion or prepacking.
