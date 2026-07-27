# Single-file submission

The submitted `.py` file is the reproducible unit: kernel, JIT launcher, input
creation, oracle, checks, and bounded timing. The remote service cannot depend
on adjacent project files.

## Required structure

```python
import torch
import cutlass
import cutlass.cute as cute


class Operation:
    def can_implement(self, a, b, c):
        return True, ""

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        ...

    @cute.jit
    def __call__(self, a, b, c, stream):
        self.kernel(a, b, c).launch(
            grid=(...), block=(128, 1, 1), stream=stream
        )


def main() -> None:
    torch.manual_seed(0)
    # Allocate tensors and make CuTe views.
    operation = Operation(...)
    ok, reason = operation.can_implement(a, b, c)
    if not ok:
        raise RuntimeError(reason)
    executor = cute.compile(operation, a, b, c, stream)
    executor(a, b, c, stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, reference, atol=..., rtol=...)


if __name__ == "__main__":
    main()
```

This is structural; derive all placeholders from the operation.

## Boundary rules

- `@cute.kernel`: device code and explicit launch resources.
- `@cute.jit`: compiled host setup or device helper.
- `main()`: allocation, deterministic inputs, framework conversion, oracle,
  assertions, and timing.
- `cutlass.Constexpr`: values that specialize code.

Torch storage may cross through supported DLPack conversion or an explicit CuTe
pointer/layout view. Preserve the Torch owner until synchronization. Use the
same underlying stream for Torch events and CuTe launch; installed examples
define the exact stream wrapper.

Expose tile, stages, cluster, instruction mode, types, alignment, and tail
support. Reject unsupported cases before launch.

## Compile, check, measure

Compile once per specialization and reuse the executor. Keep compilation,
descriptors, allocation, input generation, and reference computation outside
kernel-only timing.

Print compact results:

```text
case M=... N=... K=... passed=true max_abs=... max_rel=...
timing warmup=10 iterations=50 median_ms=... tflops=...
```

Assertions must fail the process. Never catch a kernel failure and return zero,
or fall back to `torch.matmul`.

## Submit

Use the URL/header/form command in `AGENTS.md`; read the API key only from
`CUTE_HARNESS_API_KEY`. Inspect process fields before profiler fields. Download
a profile only to answer a concrete question.

Before submission:

- file is self-contained and calls `main()`
- implementation is CuTe DSL
- required cases and exact oracle are present
- unsupported cases exit nonzero
- executor reuse and synchronization are correct
- no secret, local absolute path, or generated artifact is embedded
