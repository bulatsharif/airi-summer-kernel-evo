# Language, control-flow, and launch patterns

Tier III adds code fragments. Every uppercase name is deliberately undefined.
Fragments demonstrate one API boundary at a time and do not form a complete
task kernel.

## Decorated kernel and JIT boundary

```python
@cute.kernel
def device_operation(
    source: cute.Tensor,
    destination: cute.Tensor,
    logical_count: cutlass.Int32,
):
    # Device implementation is intentionally omitted.
    ...


@cute.jit
def entrypoint(
    source: cute.Tensor,
    destination: cute.Tensor,
    logical_count: cutlass.Int32,
):
    device_operation(source, destination, logical_count).launch(
        grid=(GRID_X, GRID_Y, 1),
        block=(BLOCK_X, 1, 1),
    )
```

The kernel arguments are bound before `.launch`. `GRID_X`, `GRID_Y`, and
`BLOCK_X` must be derived from the operation and resource model.

## Coordinate extraction

```python
thread_x, thread_y, thread_z = cute.arch.thread_idx()
block_x, block_y, block_z = cute.arch.block_idx()
block_size_x, block_size_y, block_size_z = cute.arch.block_dim()
grid_size_x, grid_size_y, grid_size_z = cute.arch.grid_dim()
```

The functions return tuples. A one-dimensional linear coordinate can be
expressed as:

```python
linear_thread = block_x * block_size_x + thread_x
```

This expression says nothing about which tensor mode the coordinate owns.

## Runtime predicate

```python
coordinate = COMPUTE_LOGICAL_COORDINATE()
if coordinate < runtime_extent:
    USE_VALID_COORDINATE(coordinate)
```

The branch is generated runtime control flow because the predicate contains
DSL values. Every object used in both regions must keep a compatible type.

## Compile-time specialization

```python
@cute.jit
def specialized_helper(
    runtime_value: cutlass.Int32,
    feature_enabled: cutlass.Constexpr,
):
    if cutlass.const_expr(feature_enabled):
        APPLY_OPTIONAL_STATIC_FEATURE(runtime_value)
    else:
        APPLY_BASE_FEATURE(runtime_value)
```

Each `feature_enabled` value creates a specialization. The uppercase calls
represent compatible generated operations.

## Static and dynamic loops

```python
@cute.jit
def loop_forms(
    runtime_count: cutlass.Int32,
    static_count: cutlass.Constexpr,
):
    for static_index in cutlass.range_constexpr(static_count):
        BUILD_STATIC_OBJECT(static_index)

    for runtime_index in cutlass.range(runtime_count):
        PROCESS_RUNTIME_ITEM(runtime_index)
```

The static loop creates code during tracing. The runtime loop emits an IR loop.
Do not pass `runtime_count` to `range_constexpr`.

## Runtime loop with compiler controls

```python
for runtime_index in cutlass.range(
    runtime_count,
    unroll=UNROLL_POLICY,
):
    PROCESS_RUNTIME_ITEM(runtime_index)
```

`UNROLL_POLICY` is a compile-time choice. It is deliberately not assigned a
value.

## Static and runtime debugging

```python
print("static layout:", layout)
print("static shape:", tensor.shape)

if thread_x == 0 and block_x == 0:
    cute.printf("runtime coordinate={}, value={}", coordinate, value)
```

Python `print` runs while compiling. `cute.printf` runs on the device. Remove
device printing before timing.

## Alignment and divisibility assumptions

```python
runtime_extent = cute.assume(runtime_extent, divby=REQUIRED_DIVISIBILITY)
```

This is a compiler assumption, not a check. Only use divisibility guaranteed
by the external contract or preceding control flow.

## Optional cluster launch

```python
bound = device_operation(arguments...)
bound.launch(
    grid=(GRID_X, GRID_Y, GRID_Z),
    block=(BLOCK_X, BLOCK_Y, BLOCK_Z),
    cluster=(CLUSTER_X, CLUSTER_Y, CLUSTER_Z),
    smem=SHARED_MEMORY_BYTES,
    stream=stream,
)
```

Cluster, dynamic shared memory, and stream are independent launch parameters
but must agree with kernel allocation and synchronization. No values shown here
are defaults.

## Candidate interface preservation

```python
class ModelNew:
    forward = staticmethod(entrypoint)
```

This compatibility alias does not compile or launch the kernel by itself. The
harness calls the named JIT entry point and owns external inputs and
evaluation.
