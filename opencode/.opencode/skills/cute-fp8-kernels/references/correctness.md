# Correctness and validation

Correctness is a release gate, not a diagnostic print. Use this reference to
define the operation, construct the oracle, choose tests, classify failures, and
decide whether optimization may begin.

## Contents

- [Write the operation contract](#write-the-operation-contract)
- [Separate three error sources](#separate-three-error-sources)
- [Dense FP8 reference](#dense-fp8-reference)
- [Scaled FP8 reference](#scaled-fp8-reference)
- [Epilogue and output conversion](#epilogue-and-output-conversion)
- [Tolerance policy](#tolerance-policy)
- [Required error metrics](#required-error-metrics)
- [Test matrix](#test-matrix)
- [Layout and stride tests](#layout-and-stride-tests)
- [Boundary and tail behavior](#boundary-and-tail-behavior)
- [Memory-safety checks](#memory-safety-checks)
- [Determinism and reproducibility](#determinism-and-reproducibility)
- [Native-path evidence](#native-path-evidence)
- [Remote validation sequence](#remote-validation-sequence)
- [Failure behavior](#failure-behavior)
- [Acceptance checklist](#acceptance-checklist)

## Write the operation contract

Before implementation, record these facts in the submission as constants,
arguments, comments, or assertions:

```text
mathematical operation
logical A/B/C shapes
physical A/B/C shapes and strides
batch semantics
A and B FP8 formats
scale direction, granularity, dtype, shape, and layout
accumulator dtype
output dtype
alpha/beta and activation order
supported dimensions and alignment
tail/predication guarantee
NaN/Inf/saturation behavior
performance metric
```

Example:

```text
C_fp16 = cast_fp16(A_e4m3 @ B_e4m3)
A logical/physical: MxK, K-contiguous
B logical: KxN; physical: NxK, K-contiguous
accumulator: FP32
no external scales; no alpha/beta
M,N,K are multiples of 128,128,64
all bases 16-byte aligned
```

If scale semantics are ambiguous, do not code until they are resolved. A factor
named `scale` may mean either:

```text
q = quantize(x / scale); x_hat = q * scale
```

or:

```text
q = quantize(x * inv_scale); x_hat = q / inv_scale
```

Those are not interchangeable.

## Separate three error sources

Report these independently:

1. **Quantization error**

   Difference between the original high-precision operation and the operation
   on dequantized quantized operands.

2. **Kernel arithmetic error**

   Difference between kernel output and a high-precision implementation of the
   exact quantized/scaled operation.

3. **Output conversion error**

   Difference caused by the specified C dtype or epilogue conversion.

Only item 2 determines whether the kernel implements the requested quantized
operation correctly. A large item 1 does not justify a large item 2.

## Dense FP8 reference

Generate high-precision inputs, then quantize once:

```python
torch.manual_seed(seed)
a_fp32 = make_a(...)
b_fp32 = make_b(...)

a_q = a_fp32.to(torch.float8_e4m3fn)
b_q = b_fp32.to(torch.float8_e4m3fn)
```

Build the kernel oracle from the stored FP8 values:

```python
a_hat = a_q.float()
b_hat = b_q.float()
acc_ref = a_hat @ b_hat
```

Match physical B convention explicitly:

```python
# If kernel stores B physically as [N,K]:
b_kernel_storage = b_q.transpose(0, 1).contiguous()

# Mathematical reference remains [M,K] @ [K,N].
acc_ref = a_q.float() @ b_q.float()
```

Do not dequantize A/B inside the submitted GPU implementation and call
`torch.matmul`; dequantization and Torch matmul are reference-only.

For batched GEMM, compute every batch independently or use a correctly shaped
batched reference. Verify that a batch stride bug cannot pass because all
batches contain the same data.

## Scaled FP8 reference

For tensorwise inverse scaling:

```python
a_q = (a_fp32 * a_inv_scale).to(fp8_dtype)
b_q = (b_fp32 * b_inv_scale).to(fp8_dtype)

a_hat = a_q.float() / a_inv_scale
b_hat = b_q.float() / b_inv_scale
acc_ref = a_hat @ b_hat
```

For direct scales:

```python
a_q = (a_fp32 / a_scale).to(fp8_dtype)
a_hat = a_q.float() * a_scale
```

For block scaling, expand only for the host reference:

```text
A_hat[m,k,l] = float(A_q[m,k,l]) * SFA[m,floor(k/V),l]
B_hat[k,n,l] = float(B_q[k,n,l]) * SFB[n,floor(k/V),l]
C_ref = A_hat @ B_hat
```

where `V=sf_vec_size`. The kernel must consume the physical scale layout
defined by the contract; the host reference may construct a logical expanded
tensor for clarity.

Verify:

- last partial scale block behavior if K is not divisible by V
- scale padding values
- E8M0/other scale decoding
- whether zero scale is legal
- whether scale tensors are shared across batch

## Epilogue and output conversion

Apply the reference in the specified order. A typical GEMM epilogue is:

```text
acc = A_hat @ B_hat
out_acc = alpha * acc + beta * C_source
activated = activation(out_acc)
C_ref = convert_to_output_dtype(activated)
```

Changing activation/conversion order changes results:

```text
convert(relu(acc)) != relu(convert(acc))
```

For output FP8 with output scale or amax generation, the oracle must implement
the exact saturation, rounding, and scale-update order.

Compare kernel output with the final `C_ref`, and optionally compare an
accumulator debug output with `acc` when localizing an epilogue failure.

## Tolerance policy

Use:

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=rtol,
    atol=atol,
    equal_nan=False,
)
```

Tolerance depends on:

- accumulation precision
- output precision
- K/reduction length
- input magnitude/distribution
- fused operation order
- whether reference matches hardware rounding points

Good starting investigation ranges for a quantized-input reference are:

| Output/accumulator | Initial `rtol` | Initial `atol` |
|---|---:|---:|
| FP32 output, FP32 acc | `1e-3` | scale-dependent, often `1e-3` |
| FP16 output | `1e-2` | scale-dependent, often `1e-2` |
| BF16 output | `2e-2` | scale-dependent, often `2e-2` |
| FP8 output | derive from output quantization step | derive from output quantization step |

These are starting points, not acceptance constants. Tighten after observing a
known-correct implementation. If the task gives tolerances, use them.

Never:

- compare only maximum relative error
- ignore values near zero without an absolute criterion
- set a broad tolerance such as 50% because inputs are FP8
- change tolerance after seeing a candidate fail without explaining the numeric
  model

For magnitude-dependent tests, define `atol` relative to a documented output
scale, not the candidate's own maximum value.

## Required error metrics

Always compute:

```python
diff = (actual.float() - expected.float()).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()
rmse = diff.square().mean().sqrt().item()

denom = expected.float().abs().clamp_min(reference_floor)
max_rel = (diff / denom).max().item()
```

Also report:

- mismatched element count under chosen `rtol/atol`
- first/worst mismatching coordinate
- actual and expected value there
- NaN count
- Inf count

For debugging, compare per-row/per-column maxima. A regular stripe pattern often
indicates layout or tile-coordinate error; errors that grow with K often
indicate accumulate/reset or scale indexing problems.

## Test matrix

Use several test families.

### Smoke cases

- all zeros
- one nonzero element
- identity-like operands
- constant small exactly representable FP8 values
- one output tile
- one K tile

### Random cases

- uniform small range
- normal distribution clipped away from saturation
- independent data per batch
- multiple seeds

### Numerical stress

- alternating signs/cancellation
- values near FP8 quantization transitions
- values near maximum intended magnitude
- very small normal values
- heterogeneous row/column magnitude

### Shape cases

- exact one-tile shape
- multiple M tiles only
- multiple N tiles only
- multiple K tiles
- multiple batch values
- minimum supported dimensions
- maximum required benchmark dimensions
- non-square matrices
- tails only when explicitly supported

Avoid using only powers of two with identical M/N/K. Symmetry hides swapped
dimensions and transposition bugs.

## Layout and stride tests

When supported, test:

- A K-major and MN-major
- B K-major and MN-major
- C row-major and column-major
- nontrivial batch strides
- views with padded leading dimensions
- size-one modes

Verify physical storage independently:

```python
assert tensor.stride() == expected_stride
assert tensor.data_ptr() % required_alignment == 0
```

Use asymmetric coordinate patterns such as:

```text
A[m,k] = f(m,k)
B[k,n] = g(k,n)
```

where `f` and `g` encode coordinates. This catches accidental transpose or
stride aliasing better than IID random values.

## Boundary and tail behavior

State one of:

1. dimensions must be tile multiples and are validated before launch
2. GMEM loads are predicated, invalid values become zero, stores are predicated
3. TMA supports OOB fill/store for the exact configured path

Do not leave the behavior implicit.

For supported tails, test:

```text
tile-1, tile+1
2*tile-1, 2*tile+1
K instruction boundary +/- 1
scale-vector boundary +/- 1
```

For unsupported tails, assert/fail before kernel launch.

## Memory-safety checks

Correct values do not prove absence of out-of-bounds stores.

Useful guards:

- initialize C with a sentinel before launch
- surround logical output with guard regions when practical
- verify input tensors remain unchanged
- verify untouched output padding remains sentinel
- run a memory checker when available

For asynchronous kernels, synchronize before reading guards.

A kernel timeout can be a synchronization bug rather than merely poor
performance; treat it as correctness failure.

## Determinism and reproducibility

Every result should include:

- random seed
- shape/layout/dtype/scales
- tile, cluster, stages, CTA group
- GPU name/capability
- Torch/CUDA/CUTLASS versions

Use fixed seeds for comparison. Keep input generation outside timing.

If the operation is expected to be deterministic, run the same input multiple
times and compare outputs bitwise or with zero tolerance. Nondeterminism often
reveals races or missing waits.

## Native-path evidence

Passing FP8 tensors is insufficient evidence that hardware FP8 MMA ran.

Require at least one:

- selected `tcgen05.MmaF8F6F4Op`/matching helper configuration
- generated PTX/SASS containing the intended tcgen05 MMA kind
- profiler instruction/counter evidence

Reject implementations that:

- convert FP8 input to FP16/FP32 and use a different matmul
- call Torch/CuBLAS as the implementation
- use an unrelated C++/Triton operator
- only benchmark quantization or conversion

## Remote validation sequence

Use progressively more expensive runs:

1. **Environment/import**
   - import Torch, CUTLASS, CuTe, CUDA bindings
   - print target versions/capability
2. **Compile**
   - compile smallest supported configuration
3. **Smoke correctness**
   - one tile, one K tile, simple input
4. **General correctness**
   - random/asymmetric input, multiple K tiles
5. **Required suite**
   - every promised shape/layout
6. **Repeatability**
   - repeated same-input execution
7. **Native-path inspection**
8. **Performance**

Do not submit large benchmark shapes while import or one-tile correctness is
still failing.

## Failure behavior

Raise or exit nonzero on:

- import/compile failure
- unsupported target/configuration
- incorrect output
- NaN/Inf where not expected
- timeout
- required profile/timing failure

Do not:

```python
try:
    run_kernel()
except Exception as exc:
    print(exc)  # then exit 0
```

Instead:

```python
try:
    run_kernel()
except Exception:
    traceback.print_exc()
    raise
```

Print a machine-readable final line when useful:

```text
RESULT success=true max_abs=... max_rel=... median_us=...
```

but let the process exit status remain authoritative.

## Acceptance checklist

- [ ] Operation, layouts, dtypes, scaling, and epilogue are explicit.
- [ ] Reference starts from the actual quantized/scaled operands.
- [ ] Quantization and kernel errors are separate.
- [ ] Tolerance has a numeric justification.
- [ ] Smoke, random, stress, shape, and layout cases pass.
- [ ] Required tails are tested or rejected before launch.
- [ ] NaN/Inf and mismatched element counts are checked.
- [ ] Inputs/padding/guards show no obvious memory corruption.
- [ ] Repeated execution is stable.
- [ ] Native FP8 path has evidence.
- [ ] Remote process exits nonzero on every failed gate.
