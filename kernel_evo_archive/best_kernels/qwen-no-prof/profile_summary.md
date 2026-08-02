# B300 device profile of your last candidate

- device: NVIDIA B300 SXM6 AC, 148 SMs, 232448 B opt-in shared memory/block
- GPU busy: 34.262 ms total; kernels 34.228 ms over 900 launches of 57 distinct names
- memcpy: 0.032 ms over 17 calls; memset: 0.002 ms over 2 calls

| kernel | ms | calls | % GPU |
| --- | ---: | ---: | ---: |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 6.965 | 55 | 20.3 |
| kernel_cutlass_attention_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmem… | 4.743 | 55 | 13.8 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 3.854 | 55 | 11.2 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 2.851 | 55 | 8.3 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 2.722 | 55 | 7.9 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 2.515 | 55 | 7.3 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 2.500 | 55 | 7.3 |
| kernel_cutlass_fp8_gemm_kernel_TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom… | 2.495 | 55 | 7.3 |
| kernel_cutlass_rms_norm_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf32gmemo25601… | 1.679 | 55 | 4.9 |
| kernel_cutlass_post_norm_kernel_tensorptrf32gmemoi641_tensorptrf32gmemo25601_tensorptrf… | 1.643 | 55 | 4.8 |
