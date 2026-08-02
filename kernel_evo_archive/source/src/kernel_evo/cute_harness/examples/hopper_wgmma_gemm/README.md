# Hopper BF16/FP8 TMA-WGMMA GEMM

This is NVIDIA's Python CuTe DSL Hopper dense GEMM from CUTLASS v4.2.1, kept with its BSD-3-Clause notice and adapted in two narrow ways: BF16 input/output is enabled and made the default, and FP8 reference checking uses the values actually stored after quantization.

It is the complete low-level example for this harness: TMA, cluster multicast, swizzled shared memory, multi-stage transaction barriers, WGMMA, Float32 accumulators, register/shared-memory epilogue, and TMA store.

BF16 smoke:

```bash
CUTE_DSL_ARCH=sm_90a python kernel.py \
  --mnkl 128,128,128,1 \
  --tile_shape_mn 64,64 \
  --a_dtype BFloat16 --b_dtype BFloat16 --c_dtype BFloat16 \
  --acc_dtype Float32 --warmup_iterations 1 --iterations 5
```

FP8 E4M3 smoke with BF16 output:

```bash
CUTE_DSL_ARCH=sm_90a python kernel.py \
  --mnkl 128,128,128,1 \
  --tile_shape_mn 64,64 \
  --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN --c_dtype BFloat16 \
  --acc_dtype Float32 --rtol 0.01 --warmup_iterations 1 --iterations 5
```

Add `--artifact_dir .kernelevo/cute-artifacts/fp8-gemm` to extract the 4.2.x executor's embedded CUBIN, disassemble SASS, and record resource usage. Then inspect it with `kernel-evo cute inspect-codegen .../hopper_wgmma_gemm.cubin --expect wgmma --expect tma`.

Use `--compile_only` for a compilation/artifact probe without correctness or timing. Normal execution prints one final JSON metrics object so the harness `check` and `benchmark` commands can consume it directly.

For real tuning, benchmark representative large shapes, compilation excluded. Treat tile shape, cluster shape, stages, warp-group count, and epilogue policy as coupled resource decisions. Do not paste the entire file into an author prompt; retrieve the manifest, relevant semantic cards, and this coherent source bundle only for MMA-like tasks.
