# Task: LayerNorm, CuTe FP8

Implement candidate code for:

```text
input shape:      [16, 64, 256, 256]
normalized shape: [64, 256, 256]
epsilon:          1e-5
weight:           ones
bias:             zeros
```

The evaluator presents the input and output as two-dimensional tensors with
shape `[16, 4_194_304]`. Each row is normalized independently.

## Precision contract

- Input storage is FP8 E4M3FN.
- Convert every input value to FP32 and multiply by `INPUT_SCALE` before use.
- Mean, centered variance, normalization, and output are FP32.
- Variance is the mean of `(x - mean) ** 2`.
- A correctness-first streaming implementation is acceptable; speed is not
  scored yet.

## Candidate ABI

- Edit only the prepared `submission.py`.
- Keep at least one `@cute.kernel`, one `@cute.jit`, `cute.rsqrt`, and a real
  butterfly warp reduction using `cute.arch.shuffle_sync_bfly`.
- `layer_norm(input_tensor, output_tensor)` is the evaluator entry point.
- Launch a CuTe kernel that writes every output element.
- Do not define/call `main()`, create inputs, compute a PyTorch reference, or
  print a PASS marker. The harness appends those evaluator-owned parts.

## Acceptance

- Full max absolute error versus PyTorch LayerNorm: `<= 0.01`.
- Every output value must be finite.

## Iteration loop

```text
python -m cute_harness check level1_40_layer_norm_fp8 submission.py
python -m cute_harness run level1_40_layer_norm_fp8 submission.py
```
