# Task: packed FP4 vector scaling

Implement a CuTe DSL kernel for:

```text
output[i] = 0.5 * input[i]
```

- `input` contains 67,108,864 logical FP4 E2M1FN values, packed two per byte.
- `output` is FP16 with the same logical length.
- Compute in FP32 and store FP16.
- The harness owns allocation, compilation, validation, and timing.

Preserve `class ModelNew: forward = staticmethod(scale_fp4)`, at least one
`@cute.kernel`, and `@cute.jit def scale_fp4(...)`.
Do not add `main()`, PyTorch calls, inputs, reference code, or PASS output.
After correctness, fitness is speedup relative to the verified parent.
