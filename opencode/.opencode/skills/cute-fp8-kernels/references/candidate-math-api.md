# Candidate math API: CUTLASS CuTe DSL 4.6.1 on B300

Use this page for scalar elementwise expressions and reductions in
candidate-only tasks. Every direct intrinsic below was compile, launch, and
numerically verified on the shared B300 worker. Do not substitute similarly
named Torch, Triton, CUDA, or guessed CuTe helpers.

## FP32 scalar contract

Load FP32 values explicitly:

```python
x = input_tensor[index].to(cutlass.Float32)
```

An intrinsic can return an MLIR `ArithValue`. Assignment and ordinary
arithmetic accept it, but another CuTe intrinsic may reject it. Materialize the
argument at **every nested intrinsic boundary**:

```python
exp_x = cute.exp(cutlass.Float32(x))
log_exp_x = cute.log(cutlass.Float32(exp_x))
```

Do not create a one-element CuTe tensor or layout to convert a scalar.

## Direct B300-verified intrinsics

| Operation | CuTe DSL spelling | Input domain |
|---|---|---|
| exponential | `cute.exp(cutlass.Float32(x))` | finite FP32 |
| base-2 exponential | `cute.exp2(cutlass.Float32(x))` | finite FP32 |
| natural logarithm | `cute.log(cutlass.Float32(x))` | `x > 0` |
| base-2 logarithm | `cute.log2(cutlass.Float32(x))` | `x > 0` |
| hyperbolic tangent | `cute.tanh(cutlass.Float32(x))` | finite FP32 |
| error function | `cute.erf(cutlass.Float32(x))` | finite FP32 |
| square root | `cute.sqrt(cutlass.Float32(x))` | `x >= 0` |
| reciprocal square root | `cute.rsqrt(cutlass.Float32(x))` | `x > 0` |
| sine | `cute.sin(cutlass.Float32(x))` | finite FP32 |
| cosine | `cute.cos(cutlass.Float32(x))` | finite FP32 |
| floor | `cute.floor(cutlass.Float32(x))` | finite FP32 |

The same probe confirmed that `cute.ceil` is **not exposed**. Use:

```python
ceil_x = -cute.floor(cutlass.Float32(-x))
```

## Scalar selection without dynamic Python control flow

Runtime SSA predicates cannot be converted to a Python `bool`. Use arithmetic
selection, not `if x > 0.0:`:

```python
maximum = a * (a >= b) + b * (b > a)
minimum = a * (a <= b) + b * (b < a)
absolute = x * (x >= 0.0) - x * (x < 0.0)
relu = x * (x > 0.0)
leaky_relu = x * ((x >= 0.0) + slope * (x < 0.0))
clamped = maximum * (maximum <= upper) + upper * (maximum > upper)
```

These recipes assume finite inputs. They do not reproduce special NaN
propagation rules of a framework `maximum`/`minimum` operation.

Do not invent `cute.maximum`, `cute.minimum`, `cute.fmax`, `cute.fmin`,
`cutlass.relu`, or `cute.ceil`; those helpers are not available on this worker.

## Stable activation recipes

### Sigmoid

Use the sign-stable form rather than unconditional `exp(-x)`:

```python
x = cutlass.Float32(x)
abs_x = x * (x >= 0.0) - x * (x < 0.0)
z = cute.exp(cutlass.Float32(-abs_x))
positive = 1.0 / (1.0 + z)
negative = z / (1.0 + z)
sigmoid = positive * (x >= 0.0) + negative * (x < 0.0)
```

### SiLU / Swish

```python
silu = x * sigmoid
```

### Softplus

```python
x = cutlass.Float32(x)
positive = x * (x > 0.0)
abs_x = x * (x >= 0.0) - x * (x < 0.0)
softplus = positive + cute.log(
    cutlass.Float32(1.0 + cute.exp(cutlass.Float32(-abs_x)))
)
```

### Mish

```python
x = cutlass.Float32(x)
positive = x * (x > 0.0)
abs_x = x * (x >= 0.0) - x * (x < 0.0)
softplus = positive + cute.log(
    cutlass.Float32(1.0 + cute.exp(cutlass.Float32(-abs_x)))
)
mish = cutlass.Float32(
    x * cute.tanh(cutlass.Float32(softplus))
)
```

### Exact GELU

```python
x = cutlass.Float32(x)
gelu = 0.5 * x * (
    1.0 + cute.erf(cutlass.Float32(x * 0.7071067811865476))
)
```

## Reduction compositions

Use the task-selected warp/block reduction recipe for `warp_max` and
`warp_sum`. CuTe math intrinsics do not perform cross-lane reductions.

### Stable row softmax

```text
maximum    = warp_max(local maximum)
numerator  = exp(x - maximum)
denominator = warp_sum(local sum of numerator)
output     = numerator / denominator
```

At the intrinsic boundary:

```python
numerator = cute.exp(cutlass.Float32(x - maximum))
```

### LogSumExp

```python
maximum = warp_max(local_maximum)
shifted_sum = warp_sum(
    cute.exp(cutlass.Float32(x - maximum))
)
result = maximum + cute.log(cutlass.Float32(shifted_sum))
```

### LayerNorm / RMSNorm inverse standard deviation

```python
inverse_std = cute.rsqrt(cutlass.Float32(variance + epsilon))
```

## Retry rule

If a nested intrinsic reports `ArithValue`, add one
`cutlass.Float32(...)` conversion at that exact boundary. Do not rewrite a
compiling GEMM, layout, TMA pipeline, or reduction to fix scalar math.
