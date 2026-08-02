# Packed E2M1 semantics and CuTe API

Use this page when the task exposes raw packed FP4 storage rather than a native
block-scaled MMA tensor. The task contract decides nibble order and scale.

## E2M1 values

An E2M1 nibble contains one sign bit and three magnitude bits:

| Magnitude bits | Value |
|---:|---:|
| `000` | `0.0` |
| `001` | `0.5` |
| `010` | `1.0` |
| `011` | `1.5` |
| `100` | `2.0` |
| `101` | `3.0` |
| `110` | `4.0` |
| `111` | `6.0` |

Bit `0x8` negates the magnitude. E2M1 is not a four-bit integer and is not a
uniform fixed-point format.

For the common low-nibble-first convention:

```text
byte i:
  logical element 2*i     = byte & 0x0f
  logical element 2*i + 1 = byte >> 4
```

Do not assume this order if the task states otherwise.

## Candidate-facing API

The scalar path needs only public CuTe mechanics:

```python
import cutlass
import cutlass.cute as cute
```

Relevant scalar types:

```python
cutlass.Uint8
cutlass.Int32
cutlass.Float16
cutlass.Float32
```

Convert a loaded byte before masking:

```python
packed = input_tensor[byte_index].to(cutlass.Int32)
low = packed & 0xF
high = packed >> 4
```

Use the public coordinate helpers. They take no axis argument and return
three-tuples:

```python
thread_x, _, _ = cute.arch.thread_idx()
block_x, _, _ = cute.arch.block_idx()
block_size_x, _, _ = cute.arch.block_dim()
byte_index = block_x * block_size_x + thread_x
```

Bind kernel arguments before launch:

```python
kernel(input_tensor, output_tensor).launch(
    grid=(grid_x, 1, 1),
    block=(threads, 1, 1),
)
```

If the starter exposes pointers, create views without extracting Python-side
addresses:

```python
packed = cute.make_tensor(
    input_ptr,
    cute.make_layout((packed_bytes,), stride=(1,)),
)
output = cute.make_tensor(
    output_ptr,
    cute.make_layout((logical_elements,), stride=(1,)),
)
```

Preserve the starter's pointer element types and alignment assumptions.

## Scaling

Write the exact equation before implementing it. Common contracts include:

```text
output = decode(fp4) * scale
output = decode(fp4) / inverse_scale
```

Block- or tensor-scaled tasks may add another scale level. Variable names do
not prove direction. Never apply a quantization scale once while packing and a
second time while decoding unless the public equation explicitly requires it.

## Correctness probes

Start with the exhaustive byte sequence:

```text
0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
```

Under low-nibble-first ordering, this visits nibble codes `0..15` in order.
Check:

- both zeros, including the negative-zero code;
- every positive and negative magnitude;
- both nibble positions;
- the first and last byte;
- an odd final logical element when the contract permits one;
- scale values other than one.

Build the oracle from the actual packed bytes, not from the pre-quantized
source values.

## Evidence status

The generic launch, tensor-view, and coordinate forms match the repository's
version-pinned B300 API pack. The bundled packed-E2M1 template is a neutral
draft and is syntax-checked locally; record remote compile, numerical, timing,
and profile evidence before labeling it B300-verified.
