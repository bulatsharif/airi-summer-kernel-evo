# B300 device profile of your last candidate

- device: NVIDIA B300 SXM6 AC, 148 SMs, 232448 B opt-in shared memory/block
- GPU busy: 28.097 ms total; kernels 28.066 ms over 503 launches of 40 distinct names
- memcpy: 0.029 ms over 15 calls; memset: 0.002 ms over 1 calls

| kernel | ms | calls | % GPU |
| --- | ---: | ---: | ---: |
| kernel_cutlass_mlp_fc_gelu_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgm… | 8.245 | 55 | 29.4 |
| kernel_cutlass_mlp_projection_kernel_tensorptrf32gmemoi641_tensorptrf8E4M3FNgmemalign16… | 7.595 | 55 | 27.0 |
| kernel_cutlass_qkv_projection_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3F… | 5.975 | 55 | 21.3 |
| kernel_cutlass_attention_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf8E4M3FNgmem… | 2.961 | 55 | 10.5 |
| kernel_cutlass_attention_projection_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf… | 2.153 | 55 | 7.7 |
| kernel_cutlass_layer_norm_fp8_kernel_tensorptrf8E4M3FNgmemalign16oi641_tensorptrf32gmem… | 0.363 | 55 | 1.3 |
| kernel_cutlass_layer_norm_fp32_kernel_tensorptrf32gmemoi641_tensorptrf32gmemo7681_tenso… | 0.354 | 55 | 1.3 |
| void sgemm_largek_lds64<true, false, 6, 3, 4, 5, 2, 66>(float*, float const*, float con… | 0.060 | 1 | 0.2 |
| cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu | 0.041 | 4 | 0.1 |
| void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 0.034 | 17 | 0.1 |
