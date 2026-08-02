# B300 device profile of your last candidate

- device: NVIDIA B300 SXM6 AC, 148 SMs, 232448 B opt-in shared memory/block
- GPU busy: 2814.714 ms total; kernels 2814.678 ms over 790 launches of 55 distinct names
- memcpy: 0.032 ms over 17 calls; memset: 0.003 ms over 2 calls

| kernel | ms | calls | % GPU |
| --- | ---: | ---: | ---: |
| kernel_cutlass_mlp_gateup_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgme… | 1160.736 | 55 | 41.2 |
| kernel_cutlass_mlp_down_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmema… | 600.597 | 55 | 21.3 |
| kernel_cutlass_qkv_q_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmemalig… | 529.210 | 55 | 18.8 |
| kernel_cutlass_out_proj_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmema… | 263.931 | 55 | 9.4 |
| kernel_cutlass_attention_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmem… | 98.669 | 55 | 3.5 |
| kernel_cutlass_qkv_kv_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmemali… | 76.246 | 55 | 2.7 |
| kernel_cutlass_qkv_kv_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmemali… | 76.245 | 55 | 2.7 |
| kernel_cutlass_q_rope_norm_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf32gmemo25… | 2.807 | 55 | 0.1 |
| kernel_cutlass_rms_norm_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf32gmemo25601… | 2.547 | 55 | 0.1 |
| kernel_cutlass_mlp_norm_kernel_tensorptrf32gmemoi641_tensorptrf32gmemo25601_tensorptrf8… | 2.038 | 55 | 0.1 |
