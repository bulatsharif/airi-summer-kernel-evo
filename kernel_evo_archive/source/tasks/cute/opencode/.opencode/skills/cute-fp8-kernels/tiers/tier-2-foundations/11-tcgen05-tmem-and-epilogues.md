# `tcgen05`, tensor memory, and epilogues

Blackwell `tcgen05` is an instruction family for tensor-core operations and
tensor-memory transfers. Its operand types, layouts, tile shapes, and CTA-group
mode are coupled.

## Namespace and operation families

```python
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell_helpers
from cutlass.cute.nvgpu import tcgen05
```

The namespace includes:

- dense narrow-precision MMA operations;
- block-scaled MMA operations;
- TMEM load/store/copy operations;
- commit and wait operations;
- CTA-group and field enums.

Use a construction helper or documented operation class. Do not encode an
instruction descriptor from remembered PTX fields.

## Tiled MMA construction inputs

A Blackwell tiled MMA construction needs:

| Input | Meaning |
| --- | --- |
| A element type | physical scalar type of operand A |
| A major mode | stride/layout interpretation for A |
| B element type or compatible family | physical scalar type of operand B |
| B major mode | stride/layout interpretation for B |
| accumulator type | type accumulated by the instruction |
| CTA group | one-CTA or cooperating-CTA instruction mode |
| MMA tiler M/N | instruction-level output tiler |
| CTA/MMA tiler | operation-level M/N/K tiling |

These facts determine legal operand fragments. They are not independent
keywords to guess.

The recommended Blackwell helper form for a dense trivial tiled MMA has seven
required positional roles:

```text
blackwell_helpers.make_trivial_tiled_mma(
    a_dtype,
    b_dtype,
    a_major,
    b_major,
    accumulator_dtype,
    cta_group,
    mma_tiler_mn,
)
```

An optional eighth positional argument selects the source kind for A and
defaults to shared memory. A deprecated overload accepts one shared A/B dtype;
use separate dtypes for new code. The helper derives remaining instruction
details from types and major modes. No concrete atom shape is prescribed here.

## Operand major modes

`cutlass.utils.LayoutEnum.from_tensor(tensor).mma_major_mode()` derives the
major mode from a tensor's physical layout. A and B may use different modes.

The mode describes how the instruction sees contiguous structure. It is not
the same as a mathematical transpose flag and must remain consistent with TMA
and shared-memory layouts.

## Shared-memory operand layouts

Blackwell helpers provide operation-compatible staged layouts:

```text
blackwell_helpers.make_smem_layout_a(
    tiled_mma,
    mma_tiler_mnk,
    a_dtype,
    num_stages,
    *,
    is_k_major=None,
)

blackwell_helpers.make_smem_layout_b(
    tiled_mma,
    mma_tiler_mnk,
    b_dtype,
    num_stages,
    *,
    is_k_major=None,
)
```

The result includes a stage mode. Allocate from the complete returned layout;
do not append a second stage mode manually.

## TMEM allocation model

Tensor memory is allocated in columns through `cutlass.utils.TmemAllocator`.
The constructor binds shared-memory address storage to a named barrier:

```text
utils.TmemAllocator(
    alloc_result_dst_smem_ptr=None,
    *,
    barrier_for_retrieve,
    allocator_warp_id=0,
    is_two_cta=False,
    num_allocated_columns=0,
    two_cta_tmem_dealloc_mbar_ptr=None,
    arch="sm_100",
    initialize_mbarrier=True,
)
```

`arch` selects the allocator instruction target. `initialize_mbarrier` controls
automatic setup of the optional two-CTA deallocation barrier.

Representative methods:

```text
allocator.reserve(number_of_columns) -> buffer pool
allocator.allocate(number_of_columns)
allocator.wait_for_alloc()
allocator.retrieve_ptr(dtype=...) -> TMEM Pointer
buffer_pool.allocate_tensor(layout, dtype) -> Tensor
allocator.relinquish_alloc_permit()
allocator.free(tmem_ptr, num_columns=0)
```

Exact allocation/free ownership depends on the selected allocator style and
CTA group. Use one coherent style.

TMEM allocation generally has:

1. one designated participant reserves or allocates;
2. required participants wait for allocation visibility;
3. fragments are rebound to the allocated TMEM pointer;
4. MMA writes TMEM;
5. consumers wait before reading;
6. the epilogue finishes;
7. the owner releases TMEM.

Releasing before all consumers finish is a use-after-release even if the
Python object remains in scope.

## Accumulator fragments

The tiled MMA supplies a compatible C/D partition shape and fragment layout:

```text
thread_mma.partition_C(output_tile)
tiled_mma.partition_shape_C(output_shape)
tiled_mma.make_fragment_C(partition_or_shape)
```

An accumulator fragment may initially describe layout without usable backing
storage. Binding that layout to an allocated TMEM pointer creates the actual
TMEM tensor.

Do not initialize a TMEM fragment with a generic register `.fill()` unless its
type and address space explicitly support that operation.

## MMA accumulation semantics

For a reduction over K tiles:

```text
first contribution: initialize or overwrite the accumulator
later contributions: accumulate into the existing value
```

The chosen operation exposes this distinction through an accumulator/source
field, an operation variant, or an instruction parameter. Applying accumulate
mode to uninitialized TMEM reads undefined prior state. Reinitializing every K
iteration drops earlier contributions.

## Commit and completion

`tcgen05` work is asynchronous. Issuing MMA does not make TMEM immediately
readable. The design must connect:

- instruction issue;
- commit of the correct group;
- completion barrier or pipeline token;
- consumer wait;
- TMEM load.

Some commit operations must be issued by one elected lane. All relevant
participants must observe compatible control flow around the wait.

## TMEM-to-register copies

A TMEM load operation is combined with a copy atom/tiled copy whose value
layout matches the accumulator fragment:

```text
tcgen05.make_tmem_copy(load_operation, accumulator_tensor)
tiled_copy.get_slice(participant)
partition source TMEM
partition destination register fragment
cute.copy(...)
```

The load operation width and repetition must cover the fragment owned by each
participant. Changing the accumulator tile changes this relationship.

## Epilogue choices

Simple direct epilogue:

```text
TMEM -> register fragment -> conversion/arithmetic -> predicated GMEM store
```

Staged epilogue:

```text
TMEM -> register fragment -> shared-memory layout -> asynchronous GMEM store
```

The staged path can improve transfer efficiency but introduces:

- extra SMEM;
- copy partitioning;
- CTA synchronization;
- optional store-pipeline state;
- more tail constraints.

Select it based on a measured bottleneck and a compatible output contract.

## Epilogue helper families

The installed Blackwell utilities include helper families for:

```text
make_smem_layout_epi
get_tmem_load_op
get_smem_store_op
epilog_tmem_copy_and_partition
epilog_smem_copy_and_partition
epilog_gmem_copy_and_partition
```

These helpers construct layouts and partitions. Their concrete arguments
depend on the tiled MMA, output type/layout, and chosen subtile structure.
Their existence does not prescribe an epilogue design.

## Completion boundary

Before releasing TMEM or returning from the kernel, prove:

1. all MMA contributions were committed;
2. the accumulator consumer waited for completion;
3. every promised output element was loaded and stored;
4. shared-memory stores, if any, completed;
5. all participants reached required barriers;
6. no later instruction refers to released TMEM.

## Exact member names

These are the names the installed Python DSL exposes. They do not mirror the
CUTLASS C++ spellings, and the difference is usually one of case convention
rather than a missing feature.

```text
tcgen05.CtaGroup.ONE          CTA group selector for a single-CTA MMA
tcgen05.Field.ACCUMULATE      the accumulate field set on a tiled MMA
tcgen05.Repetition.x64        repetition selector
tcgen05.make_tmem_copy(...)   TMEM copy construction
```

`CTA_GROUP`, `CTA_Group`, and `CtaGroup.CTA_SINGLE` do not exist.

Helpers for Blackwell tiled-MMA and shared-memory layout construction live in
`cutlass.utils.blackwell_helpers`:

```text
sm100_utils.make_trivial_tiled_mma(...)
sm100_utils.make_smem_layout_a(...)
sm100_utils.make_smem_layout_b(...)
```

## TMEM load operations

Loading an accumulator out of tensor memory uses a load-op object parameterized
by a repetition count:

```text
tcgen05.Ld32x32bOp(tcgen05.Repetition.x64)
```

The op is passed to `tcgen05.make_tmem_copy(...)` to build the copy that moves
accumulator fragments from TMEM into registers for the epilogue.
