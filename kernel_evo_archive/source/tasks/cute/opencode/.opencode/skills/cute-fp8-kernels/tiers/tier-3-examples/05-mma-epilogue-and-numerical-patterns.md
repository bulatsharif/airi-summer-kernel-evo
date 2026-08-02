# MMA, epilogue, and low-precision patterns

These fragments illustrate roles and arithmetic boundaries. They do not
construct an instruction, choose a tile, or implement a complete reduction.

## Derive operand major modes

```python
a_major = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
```

Major mode follows the physical tensor layout. It is not selected from the
mathematical notation alone.

## Construct a dense Blackwell tiled MMA

```python
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    A_ELEMENT_TYPE,
    B_ELEMENT_TYPE,
    a_major,
    b_major,
    ACCUMULATOR_TYPE,
    CTA_GROUP,
    MMA_TILER_MN,
)
```

The seven positional arguments show the installed roles. Every uppercase
value must be derived from the operation and architecture contract.

## Participant MMA partitions

```python
participant_mma = tiled_mma.get_slice(participant_index)

a_partition = participant_mma.partition_A(a_tile_mk)
b_partition = participant_mma.partition_B(b_tile_nk)
c_partition = participant_mma.partition_C(c_tile_mn)
```

A, B, and C consume different logical mode pairs. The tile views must already
use the physical convention expected by the tiled MMA.

## Fragment construction

```python
a_fragment = tiled_mma.make_fragment_A(shared_a_partition)
b_fragment = tiled_mma.make_fragment_B(shared_b_partition)
accumulator_fragment = tiled_mma.make_fragment_C(c_partition)
```

The A/B fragment constructors consume compatible shared-memory tensors. The C
fragment may require TMEM binding before it can hold accumulator values.

## Five-role MMA call

```python
cute.gemm(
    tiled_mma,
    destination_accumulator,
    a_fragment,
    b_fragment,
    source_accumulator,
)
```

This fragment omits initialization versus accumulation fields, async commit,
and completion. Those belong to the selected operation.

## First and later reduction contributions

```python
for reduction_index in cutlass.range(runtime_reduction_tiles):
    is_first = reduction_index == 0
    CONFIGURE_ACCUMULATION_MODE(is_first)
    ISSUE_MMA_FOR_REDUCTION_TILE(reduction_index)
```

`CONFIGURE_ACCUMULATION_MODE` is conceptual. It is not a CuTe API name.

## Scalar conversion

```python
compute_value = cutlass.Float32(input_value)
output_value = OUTPUT_TYPE(compute_value)
```

The public scalar type constructor performs conversion. The output type and
conversion point come from the numerical contract.

## Piecewise arithmetic

```python
if runtime_value > cutlass.Float32(0.0):
    transformed = POSITIVE_BRANCH(runtime_value)
else:
    transformed = NEGATIVE_BRANCH(runtime_value)
```

Both branches must produce compatible DSL types. The uppercase functions
represent operation-specific arithmetic.

## Dequantization equation

```python
stored = LOAD_LOW_PRECISION_VALUE(logical_coordinate)
scale_coordinate = MAP_DATA_TO_SCALE_COORDINATE(logical_coordinate)
scale = LOAD_SCALE(scale_coordinate)
real_approximation = cutlass.Float32(stored) * cutlass.Float32(scale)
```

The mapping and multiplication direction are placeholders. A task may specify
inverse scale or instruction-owned scaling instead.

## Packed nibble extraction

```python
packed_byte = LOAD_PACKED_BYTE(byte_coordinate)
low_nibble = packed_byte & cutlass.Uint8(0x0F)
high_nibble = (packed_byte >> cutlass.Uint8(4)) & cutlass.Uint8(0x0F)
```

This demonstrates packed storage only. Decoding sign/exponent/fraction and
mapping nibble order remain part of the format contract.

## Predicated output fragment

```python
logical_coordinate = MAP_FRAGMENT_VALUE_TO_OUTPUT(participant, value_index)
if OUTPUT_COORDINATE_IS_VALID(logical_coordinate):
    output_tensor[logical_coordinate] = OUTPUT_TYPE(register_fragment[value_index])
```

The participant mapping and predicate are intentionally unspecified.

## Adaptation sequence

When adapting any code fragment:

1. identify every uppercase placeholder;
2. derive it from the task or Tier II contracts;
3. preserve object kind and address space;
4. compile after one subsystem becomes concrete;
5. validate before adding another subsystem.

Combining every fragment on this page does not produce a complete kernel.
