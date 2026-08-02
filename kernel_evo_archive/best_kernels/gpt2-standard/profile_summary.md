# B300 device profile of your last candidate

- device: NVIDIA B300 SXM6 AC, 148 SMs, 232448 B opt-in shared memory/block
- GPU busy: 45.098 ms total; kernels 45.065 ms over 503 launches of 40 distinct names
- memcpy: 0.031 ms over 15 calls; memset: 0.002 ms over 1 calls

| kernel | ms | calls | % GPU |
| --- | ---: | ---: | ---: |
| kernel_cutlass_mlp_projection_kernel_tensorptrf32gmemoi641_tensorptrf8E4M3FNgmemalign16… | 16.774 | 55 | 37.2 |
| kernel_cutlass_mlp_fc_gelu_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgm… | 11.335 | 55 | 25.1 |
| kernel_cutlass_qkv_projection_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3F… | 8.511 | 55 | 18.9 |
| kernel_cutlass_attention_projection_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf… | 4.497 | 55 | 10.0 |
| kernel_cutlass_attention_fused_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3… | 3.162 | 55 | 7.0 |
| kernel_cutlass_layer_norm_fp32_kernel_tensorptrf32gmemoi641_tensorptrf32gmemo7681_tenso… | 0.185 | 55 | 0.4 |
| kernel_cutlass_layer_norm_fp8_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf32gmem… | 0.185 | 55 | 0.4 |
| void sgemm_largek_lds64<true, false, 6, 3, 4, 5, 2, 66>(float*, float const*, float con… | 0.059 | 1 | 0.1 |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu | 0.041 | 4 | 0.1 |
| void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 0.033 | 17 | 0.1 |
