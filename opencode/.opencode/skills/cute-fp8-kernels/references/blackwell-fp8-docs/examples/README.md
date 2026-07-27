# Code example layer

This directory contains small task-independent examples without mixing them
into the conceptual pages.

An example may be added only after it has:

1. passed the local compatibility checker;
2. compiled and launched on the shared B300;
3. performed a real numerical assertion where applicable;
4. recorded device time and profile ID;
5. avoided the exact benchmark shape and task epilogue.

Available example:

- `fp8-mma-one-tile.py` — neutral `A[128,64] @ B[128,64].T` FP8 GEMM bridge.
  It completed numerical validation with `max_abs_error=0.0`, device time
  `0.13721599999791942 ms`, and profile ID
  `1bf96cbb-9f27-4888-9539-95ff03883227` on the shared B300.
- `elementwise-scale-bias-relu.py` — neutral two-row scalar epilogue. It passed
  with `max_abs_error=0.0`, device time `0.06582400000351481 ms`, and profile
  ID `8b73adb4-5b02-4906-9715-ba4ee7910382`.

The example includes a standalone `main()` only to record how it was validated.
A benchmark candidate must copy only the reusable structs, kernels, and JIT
entrypoint patterns. It must not define or call `main()` because the harness
owns evaluation.
