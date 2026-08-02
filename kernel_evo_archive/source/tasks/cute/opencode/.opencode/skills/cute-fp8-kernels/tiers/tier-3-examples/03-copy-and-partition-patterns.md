# Copy and partition patterns

These fragments show the boundaries among copy operations, tiled copies,
participant slices, and tensors. The operation, layouts, widths, and
participants remain unspecified.

## Synchronous tiled copy

```python
copy_atom = cute.make_copy_atom(
    COPY_OPERATION,
    ELEMENT_TYPE,
    num_bits_per_copy=COPY_WIDTH_BITS,
)
tiled_copy = cute.make_tiled_copy_tv(
    copy_atom,
    THREAD_LAYOUT,
    VALUE_LAYOUT,
)

participant_copy = tiled_copy.get_slice(participant_index)
source_partition = participant_copy.partition_S(source_tensor)
destination_partition = participant_copy.partition_D(destination_tensor)

cute.copy(copy_atom, source_partition, destination_partition)
```

The fragment does not define the operation or layouts. The source and
destination partitions must have congruent per-participant values.

## Predicated copy

```python
predicate = BUILD_PREDICATE_FROM_LOGICAL_COORDINATES()
cute.copy(
    copy_atom,
    source_partition,
    destination_partition,
    pred=predicate,
)
```

The predicate layout must correspond to the copy's value layout. A scalar
predicate may guard an entire participant transfer; a tensor predicate can
guard individual values when supported.

## Compiler-selected vector copy

```python
cute.autovec_copy(source_fragment, destination_fragment)
```

Use this only when both fragments expose compatible contiguous structure and
alignment. The compiler chooses a legal vectorization; the call does not add
asynchronous completion.

## Generic TMA atom

```python
tma_atom, tma_coordinate_tensor = cpasync.make_tiled_tma_atom(
    TMA_OPERATION,
    global_tensor,
    SHARED_LAYOUT,
    CTA_TILER,
    num_multicast=MULTICAST_COUNT,
)
```

The operation, shared layout, CTA tiler, and multicast count must describe one
physical transfer. This fragment does not issue it.

## TMA partition

```python
smem_partition, gmem_partition = cpasync.tma_partition(
    tma_atom,
    cta_coordinate,
    CTA_LAYOUT,
    shared_tensor,
    tma_coordinate_tensor,
)
```

The output partitions are associated with the descriptor. Do not replace them
with equal-shape manual tensors.

## Transaction-barrier setup

```python
with cute.arch.elect_one():
    cute.arch.mbarrier_init(barrier_pointer, ARRIVAL_COUNT)
    cute.arch.mbarrier_expect_tx(barrier_pointer, TRANSACTION_BYTES)

cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()
```

`ARRIVAL_COUNT` and `TRANSACTION_BYTES` follow from participants and physical
transfers. No values are supplied.

## TMA issue

```python
cute.copy(
    tma_atom,
    gmem_partition,
    smem_partition,
    tma_bar_ptr=barrier_pointer,
    mcast_mask=MULTICAST_MASK,
)
```

Do not wrap this call in `elect_one`; the TMA operation handles its issue
election. Omit `mcast_mask` for a non-multicast operation.

## Completion wait

```python
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(barrier_pointer)

cute.arch.mbarrier_wait(barrier_pointer, phase)
```

This illustrates separate thread arrival and byte completion. A pipeline may
encapsulate these transitions instead of exposing raw barrier calls.

## Multicast mask

```python
mask = cpasync.create_tma_multicast_mask(
    CTA_LAYOUT_VMNK,
    cta_coordinate_vmnk,
    MULTICAST_MODE,
)
```

The cluster layout and current coordinate define recipients. `MULTICAST_MODE`
identifies the shared logical mode, not a bit number chosen independently.

## Copy audit

For each fragment, fill in:

```text
source address space:
destination address space:
element/internal type:
source layout:
destination layout:
participants:
bytes or values per participant:
boundary predicate:
completion event:
lifetime after completion:
```

The blank audit prevents a syntax pattern from becoming a task recipe.
