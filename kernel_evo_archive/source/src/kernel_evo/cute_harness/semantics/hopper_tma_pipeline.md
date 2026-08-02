# Hopper TMA, barriers, WGMMA pipeline, and clusters

The validated dataflow is:

```text
BF16/FP8 GMEM tile
  -> TMA G2S (optionally cluster multicast)
  -> staged swizzled SMEM
  -> WGMMA with Float32 RMEM accumulators
  -> fused register epilogue
  -> swizzled SMEM output tile
  -> TMA S2G
```

Create TMA atoms with the 4.2.x `cutlass.cute.nvgpu.cpasync` API:

```python
op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
    op, gmem_tensor, smem_layout, smem_tile, num_multicast=1
)
```

For a staged `PipelineTmaAsync`, `tx_count` is the exact number of bytes arriving at the full barrier for one stage. With A and B in one stage:

```python
tx_count = cute.size_in_bytes(a_dtype, a_stage_layout) + cute.size_in_bytes(b_dtype, b_stage_layout)
```

State-machine invariant:

```text
producer_acquire(stage)
TMA A/B issued against producer_get_barrier(stage)
producer_commit(stage)  # TMA pipeline commit may be a no-op
consumer_try_wait / consumer_wait(stage)
wgmma fence -> gemm -> commit_group -> wait_group
consumer_release(stage)
```

Do not release the stage while an outstanding WGMMA can still read it. Do not let only a divergent subset of participating threads execute a required barrier.

Clusters and multicast:

- Cluster dimensions multiply to at most the architecture/launch limit used by the example (the validated search set is 1x1, 2x1, 1x2, 2x2).
- Use a CTA layout and `make_layout_image_mask` to construct multicast masks.
- Initialize barriers, then complete `cluster_arrive`/`cluster_wait` before remote shared-memory use.
- Multicast reduces duplicate L2 traffic only when adjacent CTAs actually reuse the same A or B tile; otherwise it adds coordination.

Stage count must fit opt-in shared memory together with barriers and epilogue storage. More stages can reduce occupancy or spill registers, and are an experiment rather than a monotonic optimization.

