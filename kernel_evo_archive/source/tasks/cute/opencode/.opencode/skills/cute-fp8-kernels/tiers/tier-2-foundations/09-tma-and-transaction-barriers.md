# TMA and transaction-barrier foundations

The Tensor Memory Accelerator performs multidimensional transfers between
global and shared memory. A legal TMA transfer is a contract among a descriptor,
tensor layouts, a shared-memory tile, participants, and a transaction barrier.

## TMA operation families

Representative copy-operation objects under
`cutlass.cute.nvgpu.cpasync` include:

```text
CopyBulkTensorTileG2SOp
CopyBulkTensorTileG2SMulticastOp
CopyBulkTensorTileS2GOp
CopyReduceBulkTensorTileS2GOp
```

The operation selects direction and semantics. It is passed to a TMA atom
factory; it is not itself a tensor or descriptor.

## Generic TMA atom factory

```text
cpasync.make_tiled_tma_atom(
    op,
    gmem_tensor,
    smem_layout,
    cta_tiler,
    num_multicast=1,
    *,
    internal_type=None,
) -> TmaInfo
```

| Parameter | Meaning |
| --- | --- |
| `op` | TMA copy operation |
| `gmem_tensor` | global tensor and physical layout |
| `smem_layout` | destination/source shared-memory layout, possibly staged |
| `cta_tiler` | CTA-level tile definition |
| `num_multicast` | number of multicast recipients |
| `internal_type` | optional descriptor element type when storage type is unsupported directly |

The returned TMA tensor maps logical global coordinates to descriptor
coordinates. It is not ordinary data storage.

## MMA-aware A/B factories

```text
cute.nvgpu.make_tiled_tma_atom_A(
    op,
    gmem_tensor,
    smem_layout,
    mma_tiler_mnk,
    tiled_mma,
    cluster_shape_vmnk=None,
    *,
    internal_type=None,
) -> TmaInfo

cute.nvgpu.make_tiled_tma_atom_B(
    op,
    gmem_tensor,
    smem_layout,
    mma_tiler_mnk,
    tiled_mma,
    cluster_shape_vmnk=None,
    *,
    internal_type=None,
) -> TmaInfo
```

The specialized factories account for the M/K or N/K projection expected by
the tiled MMA and for multicast across the complementary cluster mode.

`TmaInfo` exposes:

```text
.atom
.smem_layout
.tma_tensor
```

Treat it as an object with named fields, not as an assumed tuple.
Two-item unpacking into `(atom, tma_tensor)` remains supported for backward
compatibility; `smem_layout` is available through the named property.

## TMA partitioning

```text
cpasync.tma_partition(
    atom,
    cta_coord,
    cta_layout,
    smem_tensor,
    gmem_tensor,
) -> (smem_partition, gmem_partition)
```

| Parameter | Meaning |
| --- | --- |
| `atom` | TMA copy atom |
| `cta_coord` | current coordinate in the CTA layout |
| `cta_layout` | participating CTA/cluster layout |
| `smem_tensor` | shared-memory tensor compatible with the atom |
| `gmem_tensor` | TMA coordinate tensor or compatible global view |

Partitioning preserves descriptor semantics. An equal-shape manual view is not
necessarily an equivalent replacement.

## Issuing TMA through `cute.copy`

The TMA atom and partitions are passed to `cute.copy`. Instruction-specific
keyword arguments may include:

| Parameter | Role |
| --- | --- |
| `tma_bar_ptr` | transaction-barrier pointer associated with the destination stage |
| `mcast_mask` | cluster recipient mask for multicast loads |

The selected copy operation defines which arguments are legal. TMA issue is
asynchronous; completion is represented by the associated transaction barrier.

## Transaction bytes

A transaction barrier tracks both thread arrivals and asynchronous byte
arrivals. The expected byte count must equal the transfers associated with that
barrier phase.

Derive bytes from:

```text
element type
exact tile layout
number of operand/scale transfers
multicast protocol
predicated or omitted transfers
```

Do not count logical elements when the physical transfer uses a different
internal type or packed representation.

## Barrier API

```text
cute.arch.mbarrier_init(mbar_ptr, arrival_count)
cute.arch.mbarrier_init_fence()
cute.arch.mbarrier_expect_tx(
    mbar_ptr,
    transaction_bytes,
    peer_cta_rank_in_cluster=None,
)
cute.arch.mbarrier_arrive_and_expect_tx(
    mbar_ptr,
    transaction_bytes,
    peer_cta_rank_in_cluster=None,
    relaxed=False,
    scope=CTA,
)
cute.arch.mbarrier_arrive(
    mbar_ptr,
    peer_cta_rank_in_cluster=None,
    arrive_count=1,
)
cute.arch.mbarrier_wait(mbar_ptr, phase)
cute.arch.mbarrier_try_wait(mbar_ptr, phase)
```

| Argument | Meaning |
| --- | --- |
| `mbar_ptr` | pointer to barrier storage in shared memory |
| `arrival_count` / `arrive_count` | initialized phase count / decrement performed by one arrive |
| `transaction_bytes` | asynchronous bytes required for completion |
| `peer_cta_rank_in_cluster` | optional remote CTA whose barrier is addressed |
| `phase` | parity/phase expected by the waiter |

The try/test operations return DSL Booleans suitable for generated control
flow.

## Election rules

Barrier initialization and expected-byte registration are single-thread state
operations. Use `cute.arch.elect_one()` around such operations when the
documented call requires one issuing lane.

The TMA copy operation already performs its own issue election. Wrapping the
TMA `cute.copy` itself in another election can change participant behavior and
cause deadlock.

After barrier initialization:

1. publish initialized barrier state with the required init fence;
2. synchronize the participating CTA or cluster as required;
3. issue transfers against that barrier;
4. perform required explicit arrivals;
5. wait for the matching phase before consuming data.

This ordering is architectural. It does not specify a task's tensors or tile
sizes.

## Multicast masks

```text
cpasync.create_tma_multicast_mask(
    cta_layout_vmnk,
    cta_coord_vmnk,
    mcast_mode,
) -> Int16
```

The mask is derived from the cluster layout, current CTA coordinate, and the
logical mode shared across recipients. Every selected receiver must follow the
same stage and barrier protocol.

## Descriptor utilities

```text
cpasync.prefetch_descriptor(tma_atom)
cpasync.copy_tensormap(tma_atom, tensormap_ptr)
cpasync.update_tma_descriptor(tma_atom, gmem_tensor, descriptor_ptr)
cpasync.fence_tma_desc_acquire(descriptor_ptr)
cpasync.cp_fence_tma_desc_release(global_descriptor_ptr, shared_descriptor_ptr)
cpasync.fence_tma_desc_release()
```

Descriptor update changes the encoded base address, shape, and strides while
preserving other descriptor fields. Acquire/release fences order descriptor
visibility. These utilities are needed only when descriptors are managed
dynamically; ordinary static descriptors do not require manual updates.
