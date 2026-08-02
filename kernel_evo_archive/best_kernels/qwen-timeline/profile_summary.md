# B300 complete GPU timeline of your last candidate

- complete GPU-side trace: 754 activities; CPU/Python profiler records are intentionally omitted
- whole capture: span 32471.418 ms; active union 404.403 ms; idle holes 32067.015 ms (98.8%); largest hole 5047.701 ms
- that idle figure covers input generation, JIT compilation and the reference as well as your block, so it is not a measure of your kernels; the per-iteration section below is
- summed activity time: 404.403 ms; this may exceed active union when streams overlap
- source labels are hints: `candidate:<symbol>` is matched against this candidate's `@cute.kernel` definitions; other labels describe likely harness/runtime work
- candidate `@cute.kernel` symbols: `attention_kernel`, `down_proj_kernel`, `mlp_proj_kernel`, `norm1_kernel`, `norm2_kernel`, `out_proj_kernel`, `qkv_kernel`, `rotary_k_kernel`, `rotary_q_kernel`, `swiglu_kernel`
- observed candidate symbols: `attention_kernel`, `down_proj_kernel`, `mlp_proj_kernel`, `norm1_kernel`, `norm2_kernel`, `out_proj_kernel`, `qkv_kernel`, `rotary_k_kernel`, `rotary_q_kernel`, `swiglu_kernel`
- candidate symbols not observed in this trace: none

## One iteration of your block, launch by launch

Median of 55 timed iterations, 10 launches each. Microseconds from the start of the iteration.

- wall span 7338.901; kernels busy 7336.597; idle between your kernels 2.304 (0.0% of the span)
- largest single gap 0.257 immediately before `swiglu_kernel`
- 45.984 elapses between one iteration and the next; that is host-side dispatch outside your block, excluded from the idle figure above, and fusing cannot remove it

| # | kernel | id | start | end | dur | idle before |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | norm1_kernel | K39 | 0.000 | 9.376 | 9.376 | — |
| 2 | qkv_kernel | K40 | 9.632 | 1789.221 | 1779.589 | 0.256 |
| 3 | rotary_q_kernel | K41 | 1789.477 | 1809.477 | 20.000 | 0.256 |
| 4 | rotary_k_kernel | K42 | 1809.733 | 1827.845 | 18.112 | 0.256 |
| 5 | attention_kernel | K43 | 1828.101 | 1983.302 | 155.201 | 0.256 |
| 6 | out_proj_kernel | K44 | 1983.558 | 2605.288 | 621.730 | 0.256 |
| 7 | norm2_kernel | K45 | 2605.543 | 2614.408 | 8.865 | 0.255 |
| 8 | mlp_proj_kernel | K46 | 2614.663 | 5867.184 | 3252.521 | 0.255 |
| 9 | swiglu_kernel | K47 | 5867.441 | 5873.904 | 6.463 | 0.257 |
| 10 | down_proj_kernel | K48 | 5874.161 | 7338.901 | 1464.740 | 0.257 |

## Hottest GPU kernels (legacy aggregate)

| kernel | ms | calls | % GPU |
| --- | ---: | ---: | ---: |
| kernel_cutlass_mlp_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_… | 179.014 | 55 | 44.3 |
| kernel_cutlass_qkv_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_gmem_… | 97.930 | 55 | 24.2 |
| kernel_cutlass_down_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN… | 80.399 | 55 | 19.9 |
| kernel_cutlass_out_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_… | 34.217 | 55 | 8.5 |
| kernel_cutlass_attention_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN… | 8.475 | 55 | 2.1 |
| kernel_cutlass_rotary_q_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_… | 1.108 | 55 | 0.3 |
| kernel_cutlass_rotary_k_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_… | 1.011 | 55 | 0.2 |
| kernel_cutlass_norm1_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_o_2… | 0.518 | 55 | 0.1 |
| kernel_cutlass_norm2_kernel_tensorptrf32_gmem_o_i641_tensorptrf32_gmem_o_25601_tensorpt… | 0.493 | 55 | 0.1 |
| kernel_cutlass_swiglu_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_gm… | 0.355 | 55 | 0.1 |

## Kernel legend

| id | source hint | kernel | calls | total us | launch |
| --- | --- | --- | ---: | ---: | --- |
| K1 | framework/validation | void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo… | 8 | 163.330 | grid=1184x1x1, block=256x1x1, smem=0, regs=56 |
| K2 | framework/validation | void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo… | 2 | 4.033 | grid=10x1x1, block=256x1x1, smem=0, regs=56 |
| K3 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 2 | 3.488 | grid=3x1x1, block=128x1x1, smem=0, regs=32 |
| K4 | framework/validation | void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo… | 2 | 3.936 | grid=1x1x1, block=256x1x1, smem=0, regs=56 |
| K5 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 3 | 5.600 | grid=1x1x1, block=128x1x1, smem=0, regs=32 |
| K6 | framework/validation | void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_… | 1 | 1.600 | grid=2x1x1, block=64x1x1, smem=0, regs=16 |
| K7 | framework/validation | void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_… | 1 | 1.248 | grid=1x1x1, block=64x1x1, smem=0, regs=16 |
| K8 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 1 | 1.440 | grid=1x1x1, block=128x1x1, smem=0, regs=32 |
| K9 | framework/validation | void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std:… | 1 | 1.312 | grid=1x1x1, block=128x1x1, smem=0, regs=22 |
| K10 | framework/validation | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::(an… | 1 | 2.528 | grid=1x1x1, block=128x1x1, smem=0, regs=24 |
| K11 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::reciprocal_kernel_cuda(at… | 1 | 1.920 | grid=1x1x1, block=128x1x1, smem=0, regs=32 |
| K12 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 1.855 | grid=16x1x1, block=128x1x1, smem=0, regs=20 |
| K13 | framework/validation | void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(ano… | 1 | 1.824 | grid=8x2x1, block=128x1x1, smem=0, regs=20 |
| K14 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::cos_kernel_cuda(at::Tenso… | 1 | 2.336 | grid=8x1x1, block=128x1x1, smem=0, regs=32 |
| K15 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::sin_kernel_cuda(at::Tenso… | 1 | 2.112 | grid=8x1x1, block=128x1x1, smem=0, regs=32 |
| K16 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 1 | 23.744 | grid=20480x1x1, block=128x1x1, smem=0, regs=32 |
| K17 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 1 | 26.624 | grid=40960x1x1, block=128x1x1, smem=0, regs=20 |
| K18 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 2 | 7.104 | grid=2560x1x1, block=128x1x1, smem=0, regs=32 |
| K19 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 2 | 9.247 | grid=5120x1x1, block=128x1x1, smem=0, regs=20 |
| K20 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 1 | 8.736 | grid=10240x1x1, block=128x1x1, smem=0, regs=32 |
| K21 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 1 | 13.120 | grid=20480x1x1, block=128x1x1, smem=0, regs=20 |
| K22 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 3 | 80.512 | grid=23040x1x1, block=128x1x1, smem=0, regs=32 |
| K23 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 3 | 90.241 | grid=46080x1x1, block=128x1x1, smem=0, regs=20 |
| K24 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 5 | 9.664 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K25 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 3 | 6.015 | grid=640x1x1, block=128x1x1, smem=0, regs=20 |
| K26 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::… | 2 | 3.168 | grid=320x1x1, block=128x1x1, smem=0, regs=16 |
| K27 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::… | 1 | 1.791 | grid=1024x1x1, block=128x1x1, smem=0, regs=16 |
| K28 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 2 | 4.639 | grid=1024x1x1, block=128x1x1, smem=0, regs=32 |
| K29 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 1 | 2.976 | grid=2048x1x1, block=128x1x1, smem=0, regs=20 |
| K30 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::… | 3 | 4.096 | grid=128x1x1, block=128x1x1, smem=0, regs=16 |
| K31 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 6 | 9.280 | grid=128x1x1, block=128x1x1, smem=0, regs=32 |
| K32 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 3 | 5.536 | grid=256x1x1, block=128x1x1, smem=0, regs=20 |
| K33 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::… | 2 | 3.104 | grid=512x1x1, block=128x1x1, smem=0, regs=16 |
| K34 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 4 | 7.713 | grid=512x1x1, block=128x1x1, smem=0, regs=32 |
| K35 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 2 | 4.480 | grid=1024x1x1, block=128x1x1, smem=0, regs=20 |
| K36 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::… | 3 | 5.696 | grid=1152x1x1, block=128x1x1, smem=0, regs=16 |
| K37 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 6 | 13.600 | grid=1152x1x1, block=128x1x1, smem=0, regs=32 |
| K38 | CuTe helper/setup | kernel_cutlass__convert_kernel_tensorptrf32gmemo151201i64512_tensorptrf8E4M3FNgmemalign… | 3 | 9.088 | grid=2304x1x1, block=128x1x1, smem=0, regs=20 |
| K39 | candidate:norm1_kernel | kernel_cutlass_norm1_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_o_2… | 55 | 518.175 | grid=128x1x1, block=128x1x1, smem=16, regs=28 |
| K40 | candidate:qkv_kernel | kernel_cutlass_qkv_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_gmem_… | 55 | 97929.585 | grid=128x80x1, block=128x1x1, smem=27136, regs=80 |
| K41 | candidate:rotary_q_kernel | kernel_cutlass_rotary_q_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_… | 55 | 1107.939 | grid=2048x1x1, block=1x1x1, smem=0, regs=64 |
| K42 | candidate:rotary_k_kernel | kernel_cutlass_rotary_k_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf32_gmem_… | 55 | 1011.363 | grid=512x1x1, block=1x1x1, smem=0, regs=64 |
| K43 | candidate:attention_kernel | kernel_cutlass_attention_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN… | 55 | 8475.000 | grid=2048x1x1, block=128x1x1, smem=2048, regs=32 |
| K44 | candidate:out_proj_kernel | kernel_cutlass_out_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_… | 55 | 34217.056 | grid=128x20x1, block=128x1x1, smem=33280, regs=79 |
| K45 | candidate:norm2_kernel | kernel_cutlass_norm2_kernel_tensorptrf32_gmem_o_i641_tensorptrf32_gmem_o_25601_tensorpt… | 55 | 493.341 | grid=128x1x1, block=128x1x1, smem=16, regs=32 |
| K46 | candidate:mlp_proj_kernel | kernel_cutlass_mlp_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_… | 55 | 179013.938 | grid=128x144x1, block=128x1x1, smem=27136, regs=72 |
| K47 | candidate:swiglu_kernel | kernel_cutlass_swiglu_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN_gm… | 55 | 355.263 | grid=128x72x1, block=128x1x1, smem=0, regs=17 |
| K48 | candidate:down_proj_kernel | kernel_cutlass_down_proj_kernel_tensorptrf8E4M3FN_gmem_align16_o_i641_tensorptrf8E4M3FN… | 55 | 80398.781 | grid=128x20x1, block=128x1x1, smem=53760, regs=128 |
| K49 | framework/validation | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te… | 1 | 4.896 | grid=640x1x1, block=128x1x1, smem=0, regs=30 |
| K50 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 1 | 1.760 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K51 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::po… | 2 | 3.488 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K52 | framework/validation | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<… | 2 | 17.345 | grid=8x1x1, block=32x16x1, smem=16, regs=30 |
| K53 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<flo… | 5 | 8.928 | grid=1x1x1, block=128x1x1, smem=0, regs=32 |
| K54 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::Ten… | 3 | 4.832 | grid=1x1x1, block=128x1x1, smem=0, regs=30 |
| K55 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 4 | 10.048 | grid=1280x1x1, block=128x1x1, smem=0, regs=20 |
| K56 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<flo… | 2 | 4.000 | grid=3x1x1, block=128x1x1, smem=0, regs=32 |
| K57 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::float8_copy_kernel_cuda(a… | 2 | 3.840 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K58 | framework/validation | nvjet_sm103_qqsss_112x64_128x14_1x2_2cta_h_bz_TNN | 1 | 6.975 | grid=148x1x1, block=256x1x1, smem=223720, regs=255 |
| K59 | framework/validation | nvjet_sm103_qqsss_64x16_128x16_2x2_2cta_h_bz_TNT | 2 | 8.256 | grid=128x1x1, block=256x1x1, smem=156168, regs=255 |
| K60 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::float8_copy_kernel_cuda(a… | 1 | 2.496 | grid=1024x1x1, block=128x1x1, smem=0, regs=32 |
| K61 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::float8_copy_kernel_cuda(a… | 3 | 5.312 | grid=128x1x1, block=128x1x1, smem=0, regs=32 |
| K62 | framework/validation | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te… | 1 | 4.415 | grid=2048x1x1, block=128x1x1, smem=0, regs=30 |
| K63 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 1 | 2.272 | grid=1024x1x1, block=128x1x1, smem=0, regs=32 |
| K64 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 2.848 | grid=2048x1x1, block=128x1x1, smem=0, regs=20 |
| K65 | framework/validation | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<… | 1 | 3.136 | grid=128x1x1, block=32x16x1, smem=16, regs=30 |
| K66 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<flo… | 1 | 1.568 | grid=2x1x1, block=128x1x1, smem=0, regs=32 |
| K67 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::Ten… | 1 | 1.408 | grid=2x1x1, block=128x1x1, smem=0, regs=30 |
| K68 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 3 | 9.280 | grid=2048x1x1, block=128x1x1, smem=0, regs=20 |
| K69 | framework/validation | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te… | 3 | 11.072 | grid=256x1x1, block=128x1x1, smem=0, regs=30 |
| K70 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 3 | 5.440 | grid=128x1x1, block=128x1x1, smem=0, regs=32 |
| K71 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::po… | 1 | 1.793 | grid=128x1x1, block=128x1x1, smem=0, regs=32 |
| K72 | framework/validation | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<… | 1 | 2.688 | grid=32x1x1, block=32x16x1, smem=16, regs=30 |
| K73 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 4 | 9.184 | grid=512x1x1, block=128x1x1, smem=0, regs=20 |
| K74 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 2.176 | grid=256x1x1, block=128x1x1, smem=0, regs=20 |
| K75 | framework/validation | void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name… | 4 | 13.858 | grid=296x2x1, block=512x1x1, smem=0, regs=28 |
| K76 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s… | 1 | 1.760 | grid=128x1x1, block=128x1x1, smem=0, regs=32 |
| K77 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 2.143 | grid=64x1x1, block=128x1x1, smem=0, regs=20 |
| K78 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 2 | 4.641 | grid=128x1x1, block=128x1x1, smem=0, regs=20 |
| K79 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s… | 1 | 1.600 | grid=32x1x1, block=128x1x1, smem=0, regs=32 |
| K80 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::float8_copy_kernel_cuda(a… | 2 | 4.032 | grid=512x1x1, block=128x1x1, smem=0, regs=32 |
| K81 | framework/validation | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te… | 1 | 4.000 | grid=1024x1x1, block=128x1x1, smem=0, regs=30 |
| K82 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 1 | 1.984 | grid=512x1x1, block=128x1x1, smem=0, regs=32 |
| K83 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 3 | 8.801 | grid=2048x1x1, block=128x1x1, smem=0, regs=20 |
| K84 | framework/validation | cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu | 1 | 7.903 | grid=4x4x16, block=64x1x1, smem=13056, regs=72 |
| K85 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, floa… | 1 | 1.888 | grid=256x1x1, block=128x1x1, smem=0, regs=32 |
| K86 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<bool>, std::a… | 1 | 1.216 | grid=8x1x1, block=128x1x1, smem=0, regs=16 |
| K87 | framework/validation | void at::native::triu_tril_kernel<bool, int, true, 8, false>(at::cuda::detail::TensorIn… | 1 | 2.177 | grid=16x1x1, block=128x1x1, smem=0, regs=30 |
| K88 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 2.465 | grid=1024x1x1, block=128x1x1, smem=0, regs=20 |
| K89 | framework/validation | void (anonymous namespace)::softmax_warp_forward<float, float, float, 7, false, false>(… | 1 | 3.424 | grid=256x1x1, block=32x4x1, smem=0, regs=32 |
| K90 | framework/validation | cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_nnn_align1_bias_f32_relu | 1 | 6.592 | grid=4x4x16, block=64x1x1, smem=18816, regs=162 |
| K91 | framework/validation | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati… | 1 | 2.912 | grid=2048x1x1, block=128x1x1, smem=0, regs=20 |
| K92 | framework/validation | nvjet_sm103_qqsss_48x64_128x16_4x2_2cta_h_bz_TNN | 2 | 15.392 | grid=112x1x1, block=256x1x1, smem=188944, regs=255 |
| K93 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s… | 3 | 6.944 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K94 | framework/validation | nvjet_sm103_qqsss_128x64_128x9_1x2_h_bz_TNT | 2 | 15.903 | grid=144x1x1, block=256x1x1, smem=229736, regs=255 |
| K95 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::float8_copy_kernel_cuda(a… | 3 | 7.424 | grid=1152x1x1, block=128x1x1, smem=0, regs=32 |
| K96 | framework/validation | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te… | 2 | 10.144 | grid=2304x1x1, block=128x1x1, smem=0, regs=30 |
| K97 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 2 | 4.608 | grid=1152x1x1, block=128x1x1, smem=0, regs=32 |
| K98 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::si… | 1 | 2.880 | grid=1152x1x1, block=128x1x1, smem=0, regs=32 |
| K99 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa… | 1 | 2.465 | grid=1152x1x1, block=128x1x1, smem=0, regs=32 |
| K100 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, std::a… | 2 | 4.705 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K101 | framework/validation | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wra… | 1 | 5.824 | grid=1x40x1, block=512x1x1, smem=2064, regs=30 |
| K102 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, floa… | 1 | 1.600 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K103 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa… | 1 | 1.696 | grid=320x1x1, block=128x1x1, smem=0, regs=32 |
| K104 | framework/validation | void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<bool, bool,… | 1 | 1.792 | grid=320x1x1, block=128x1x1, smem=0, regs=30 |
| K105 | framework/validation | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<bool, at::native::func_wrap… | 1 | 5.728 | grid=1x40x1, block=512x1x1, smem=528, regs=28 |

## Ordered GPU activities

`gap before` is device-wide idle time since all earlier GPU activity ended; zero means overlap or no measurable hole.

| # | start ms | gap before us | duration us | lane | activity |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.000 | 0.000 | 4.320 | d0/s7 | K1 |
| 2 | 50.535 | 50530.316 | 2.144 | d0/s7 | K2 |
| 3 | 669.591 | 619054.424 | 1.856 | d0/s7 | K3 |
| 4 | 669.766 | 173.376 | 1.889 | d0/s7 | K2 |
| 5 | 669.814 | 45.792 | 1.632 | d0/s7 | K3 |
| 6 | 669.842 | 26.400 | 1.824 | d0/s7 | K4 |
| 7 | 669.875 | 30.944 | 1.888 | d0/s7 | K5 |
| 8 | 669.895 | 17.888 | 2.112 | d0/s7 | K4 |
| 9 | 669.910 | 13.344 | 1.856 | d0/s7 | K5 |
| 10 | 760.633 | 90721.308 | 1.600 | d0/s7 | K6 |
| 11 | 760.776 | 141.569 | 1.248 | d0/s7 | K7 |
| 12 | 1602.868 | 842089.795 | 1.440 | d0/s7 | K8 |
| 13 | 1901.414 | 298545.309 | 1.312 | d0/s7 | K9 |
| 14 | 6154.003 | 4252587.608 | 2.528 | d0/s7 | K10 |
| 15 | 6968.928 | 814922.333 | 1.920 | d0/s7 | K11 |
| 16 | 6969.301 | 370.881 | 1.856 | d0/s7 | K5 |
| 17 | 6969.682 | 379.042 | 1.855 | d0/s7 | K12 |
| 18 | 6973.427 | 3742.955 | 1.824 | d0/s7 | K13 |
| 19 | 7151.902 | 178473.809 | 2.336 | d0/s7 | K14 |
| 20 | 7323.720 | 171815.326 | 2.112 | d0/s7 | K15 |
| 21 | 7324.605 | 883.139 | 29.600 | d0/s7 | K1 |
| 22 | 7325.195 | 560.129 | 23.744 | d0/s7 | K16 |
| 23 | 7672.260 | 347041.671 | 26.624 | d0/s7 | K17 |
| 24 | 7672.474 | 187.105 | 7.360 | d0/s7 | K1 |
| 25 | 7672.591 | 110.145 | 3.616 | d0/s7 | K18 |
| 26 | 7697.111 | 24515.587 | 4.640 | d0/s7 | K19 |
| 27 | 7697.290 | 174.912 | 7.137 | d0/s7 | K1 |
| 28 | 7697.370 | 72.959 | 3.488 | d0/s7 | K18 |
| 29 | 7721.317 | 23942.724 | 4.607 | d0/s7 | K19 |
| 30 | 7721.461 | 139.489 | 16.352 | d0/s7 | K1 |
| 31 | 7721.517 | 39.840 | 8.736 | d0/s7 | K20 |
| 32 | 7745.281 | 23755.074 | 13.120 | d0/s7 | K21 |
| 33 | 7745.918 | 624.258 | 33.088 | d0/s7 | K1 |
| 34 | 7746.211 | 260.161 | 26.784 | d0/s7 | K22 |
| 35 | 7770.359 | 24120.611 | 30.272 | d0/s7 | K23 |
| 36 | 7770.500 | 111.232 | 32.737 | d0/s7 | K1 |
| 37 | 7770.546 | 13.440 | 26.944 | d0/s7 | K22 |
| 38 | 7794.251 | 23678.113 | 30.049 | d0/s7 | K23 |
| 39 | 7794.395 | 113.376 | 32.736 | d0/s7 | K1 |
| 40 | 7794.442 | 14.400 | 26.784 | d0/s7 | K22 |
| 41 | 7819.102 | 24633.733 | 29.920 | d0/s7 | K23 |
| 42 | 7819.257 | 124.128 | 2.528 | d0/s7 | K24 |
| 43 | 7843.861 | 24602.277 | 1.984 | d0/s7 | K25 |
| 44 | 7844.242 | 378.721 | 1.568 | d0/s7 | K26 |
| 45 | 7844.304 | 60.417 | 1.600 | d0/s7 | K24 |
| 46 | 7868.375 | 24068.963 | 2.048 | d0/s7 | K25 |
| 47 | 7868.526 | 148.961 | 1.791 | d0/s7 | K27 |
| 48 | 7868.599 | 71.713 | 2.271 | d0/s7 | K28 |
| 49 | 7893.164 | 24562.405 | 2.976 | d0/s7 | K29 |
| 50 | 7893.632 | 464.961 | 1.632 | d0/s7 | K30 |
| 51 | 7893.693 | 59.776 | 1.408 | d0/s7 | K31 |
| 52 | 7918.257 | 24562.053 | 1.824 | d0/s7 | K32 |
| 53 | 7918.388 | 129.600 | 1.216 | d0/s7 | K30 |
| 54 | 7918.465 | 76.032 | 1.408 | d0/s7 | K31 |
| 55 | 7943.150 | 24683.781 | 1.888 | d0/s7 | K32 |
| 56 | 7943.275 | 122.464 | 1.632 | d0/s7 | K33 |
| 57 | 7943.325 | 48.417 | 1.888 | d0/s7 | K34 |
| 58 | 7968.032 | 24705.156 | 2.240 | d0/s7 | K35 |
| 59 | 7968.156 | 122.145 | 1.248 | d0/s7 | K30 |
| 60 | 7968.203 | 45.952 | 1.440 | d0/s7 | K31 |
| 61 | 7992.677 | 24471.652 | 1.824 | d0/s7 | K32 |
| 62 | 7992.803 | 124.128 | 1.472 | d0/s7 | K33 |
| 63 | 7992.855 | 50.976 | 1.953 | d0/s7 | K34 |
| 64 | 8017.978 | 25120.741 | 2.240 | d0/s7 | K35 |
| 65 | 8018.132 | 152.545 | 1.600 | d0/s7 | K26 |
| 66 | 8018.186 | 51.680 | 1.824 | d0/s7 | K24 |
| 67 | 8042.769 | 24581.701 | 1.983 | d0/s7 | K25 |
| 68 | 8042.900 | 129.121 | 1.824 | d0/s7 | K36 |
| 69 | 8042.953 | 50.944 | 2.176 | d0/s7 | K37 |
| 70 | 8067.892 | 24936.646 | 2.976 | d0/s7 | K38 |
| 71 | 8068.023 | 128.129 | 1.920 | d0/s7 | K36 |
| 72 | 8068.077 | 52.160 | 2.208 | d0/s7 | K37 |
| 73 | 8092.861 | 24781.797 | 3.136 | d0/s7 | K38 |
| 74 | 8092.997 | 132.256 | 1.952 | d0/s7 | K36 |
| 75 | 8093.048 | 49.056 | 2.144 | d0/s7 | K37 |
| 76 | 8117.991 | 24940.870 | 2.976 | d0/s7 | K38 |
| 77 | 10816.718 | 2698723.996 | 9.504 | d0/s7 | K39 |
| 78 | 10816.727 | 0.288 | 1780.773 | d0/s7 | K40 |
| 79 | 10818.508 | 0.256 | 20.096 | d0/s7 | K41 |
| 80 | 10818.529 | 0.256 | 18.784 | d0/s7 | K42 |
| 81 | 10818.548 | 0.255 | 154.176 | d0/s7 | K43 |
| 82 | 10818.702 | 0.289 | 625.026 | d0/s7 | K44 |
| 83 | 10819.327 | 0.256 | 8.928 | d0/s7 | K45 |
| 84 | 10819.337 | 0.256 | 3254.185 | d0/s7 | K46 |
| 85 | 10822.591 | 0.256 | 6.496 | d0/s7 | K47 |
| 86 | 10822.598 | 0.256 | 1459.812 | d0/s7 | K48 |
| 87 | 10824.058 | 0.256 | 9.504 | d0/s7 | K39 |
| 88 | 10824.068 | 0.256 | 1783.461 | d0/s7 | K40 |
| 89 | 10825.851 | 0.256 | 20.320 | d0/s7 | K41 |
| 90 | 10825.872 | 0.256 | 18.112 | d0/s7 | K42 |
| 91 | 10825.890 | 0.256 | 153.825 | d0/s7 | K43 |
| 92 | 10826.044 | 0.255 | 622.114 | d0/s7 | K44 |
| 93 | 10826.667 | 0.257 | 8.927 | d0/s7 | K45 |
| 94 | 10826.676 | 0.256 | 3256.073 | d0/s7 | K46 |
| 95 | 10829.932 | 0.289 | 6.368 | d0/s7 | K47 |
| 96 | 10829.939 | 0.255 | 1462.757 | d0/s7 | K48 |
| 97 | 10831.402 | 0.256 | 9.664 | d0/s7 | K39 |
| 98 | 10831.412 | 0.256 | 1779.589 | d0/s7 | K40 |
| 99 | 10833.192 | 0.256 | 20.320 | d0/s7 | K41 |
| 100 | 10833.212 | 0.256 | 18.368 | d0/s7 | K42 |
| 101 | 10833.231 | 0.255 | 154.336 | d0/s7 | K43 |
| 102 | 10833.386 | 0.256 | 621.506 | d0/s7 | K44 |
| 103 | 10834.007 | 0.256 | 9.248 | d0/s7 | K45 |
| 104 | 10834.017 | 0.256 | 3253.705 | d0/s7 | K46 |
| 105 | 10837.271 | 0.256 | 6.560 | d0/s7 | K47 |
| 106 | 10837.278 | 0.288 | 1457.956 | d0/s7 | K48 |
| 107 | 10838.736 | 0.256 | 9.824 | d0/s7 | K39 |
| 108 | 10838.746 | 0.256 | 1781.957 | d0/s7 | K40 |
| 109 | 10840.528 | 0.256 | 20.096 | d0/s7 | K41 |
| 110 | 10840.548 | 0.256 | 18.432 | d0/s7 | K42 |
| 111 | 10840.567 | 0.256 | 154.145 | d0/s7 | K43 |
| 112 | 10840.722 | 0.255 | 620.578 | d0/s7 | K44 |
| 113 | 10841.342 | 0.256 | 8.736 | d0/s7 | K45 |
| 114 | 10841.351 | 0.256 | 3254.697 | d0/s7 | K46 |
| 115 | 10844.606 | 0.256 | 6.496 | d0/s7 | K47 |
| 116 | 10844.613 | 0.256 | 1462.052 | d0/s7 | K48 |
| 117 | 10846.075 | 0.289 | 9.728 | d0/s7 | K39 |
| 118 | 10846.085 | 0.257 | 1779.396 | d0/s7 | K40 |
| 119 | 10847.865 | 0.257 | 20.575 | d0/s7 | K41 |
| 120 | 10847.886 | 0.257 | 18.783 | d0/s7 | K42 |
| 121 | 10847.905 | 0.257 | 154.464 | d0/s7 | K43 |
| 122 | 10848.060 | 0.288 | 621.570 | d0/s7 | K44 |
| 123 | 10848.682 | 0.256 | 8.960 | d0/s7 | K45 |
| 124 | 10848.691 | 0.256 | 3255.689 | d0/s7 | K46 |
| 125 | 10851.947 | 0.256 | 6.496 | d0/s7 | K47 |
| 126 | 10851.953 | 0.256 | 1460.356 | d0/s7 | K48 |
| 127 | 10853.635 | 220.737 | 9.280 | d0/s7 | K39 |
| 128 | 10853.644 | 0.287 | 1783.269 | d0/s7 | K40 |
| 129 | 10855.428 | 0.257 | 20.032 | d0/s7 | K41 |
| 130 | 10855.448 | 0.254 | 18.048 | d0/s7 | K42 |
| 131 | 10855.466 | 0.257 | 154.400 | d0/s7 | K43 |
| 132 | 10855.621 | 0.256 | 621.858 | d0/s7 | K44 |
| 133 | 10856.243 | 0.256 | 9.120 | d0/s7 | K45 |
| 134 | 10856.252 | 0.256 | 3253.641 | d0/s7 | K46 |
| 135 | 10859.506 | 0.256 | 6.400 | d0/s7 | K47 |
| 136 | 10859.513 | 0.256 | 1463.556 | d0/s7 | K48 |
| 137 | 10861.039 | 62.656 | 9.152 | d0/s7 | K39 |
| 138 | 10861.049 | 0.256 | 1777.605 | d0/s7 | K40 |
| 139 | 10862.826 | 0.256 | 19.936 | d0/s7 | K41 |
| 140 | 10862.847 | 0.256 | 18.336 | d0/s7 | K42 |
| 141 | 10862.865 | 0.256 | 153.793 | d0/s7 | K43 |
| 142 | 10863.019 | 0.255 | 620.450 | d0/s7 | K44 |
| 143 | 10863.640 | 0.256 | 8.800 | d0/s7 | K45 |
| 144 | 10863.649 | 0.257 | 3254.953 | d0/s7 | K46 |
| 145 | 10866.904 | 0.255 | 6.400 | d0/s7 | K47 |
| 146 | 10866.911 | 0.257 | 1462.563 | d0/s7 | K48 |
| 147 | 10868.426 | 52.897 | 9.600 | d0/s7 | K39 |
| 148 | 10868.436 | 0.256 | 1781.157 | d0/s7 | K40 |
| 149 | 10870.218 | 0.256 | 19.872 | d0/s7 | K41 |
| 150 | 10870.238 | 0.256 | 18.272 | d0/s7 | K42 |
| 151 | 10870.256 | 0.256 | 154.496 | d0/s7 | K43 |
| 152 | 10870.411 | 0.256 | 622.402 | d0/s7 | K44 |
| 153 | 10871.034 | 0.256 | 8.992 | d0/s7 | K45 |
| 154 | 10871.043 | 0.256 | 3254.089 | d0/s7 | K46 |
| 155 | 10874.297 | 0.256 | 6.496 | d0/s7 | K47 |
| 156 | 10874.304 | 0.256 | 1460.068 | d0/s7 | K48 |
| 157 | 10875.810 | 45.984 | 9.376 | d0/s7 | K39 |
| 158 | 10875.820 | 0.256 | 1779.589 | d0/s7 | K40 |
| 159 | 10877.600 | 0.256 | 20.000 | d0/s7 | K41 |
| 160 | 10877.620 | 0.256 | 18.112 | d0/s7 | K42 |
| 161 | 10877.638 | 0.256 | 155.201 | d0/s7 | K43 |
| 162 | 10877.794 | 0.256 | 621.730 | d0/s7 | K44 |
| 163 | 10878.416 | 0.255 | 8.865 | d0/s7 | K45 |
| 164 | 10878.425 | 0.255 | 3252.521 | d0/s7 | K46 |
| 165 | 10881.677 | 0.257 | 6.463 | d0/s7 | K47 |
| 166 | 10881.684 | 0.257 | 1464.740 | d0/s7 | K48 |
| 167 | 10883.194 | 44.928 | 9.280 | d0/s7 | K39 |
| 168 | 10883.203 | 0.288 | 1780.261 | d0/s7 | K40 |
| 169 | 10884.984 | 0.256 | 20.192 | d0/s7 | K41 |
| 170 | 10885.004 | 0.224 | 17.984 | d0/s7 | K42 |
| 171 | 10885.023 | 0.256 | 153.984 | d0/s7 | K43 |
| 172 | 10885.177 | 0.256 | 622.210 | d0/s7 | K44 |
| 173 | 10885.799 | 0.256 | 9.280 | d0/s7 | K45 |
| 174 | 10885.809 | 0.256 | 3255.433 | d0/s7 | K46 |
| 175 | 10889.065 | 0.288 | 6.368 | d0/s7 | K47 |
| 176 | 10889.071 | 0.256 | 1461.636 | d0/s7 | K48 |
| 177 | 10890.586 | 53.023 | 9.248 | d0/s7 | K39 |
| 178 | 10890.595 | 0.256 | 1778.757 | d0/s7 | K40 |
| 179 | 10892.374 | 0.256 | 19.968 | d0/s7 | K41 |
| 180 | 10892.395 | 0.257 | 18.751 | d0/s7 | K42 |
| 181 | 10892.414 | 0.289 | 154.080 | d0/s7 | K43 |
| 182 | 10892.568 | 0.256 | 621.889 | d0/s7 | K44 |
| 183 | 10893.190 | 0.257 | 9.023 | d0/s7 | K45 |
| 184 | 10893.199 | 0.257 | 3253.929 | d0/s7 | K46 |
| 185 | 10896.454 | 0.256 | 6.400 | d0/s7 | K47 |
| 186 | 10896.460 | 0.256 | 1462.180 | d0/s7 | K48 |
| 187 | 10897.973 | 50.560 | 9.344 | d0/s7 | K39 |
| 188 | 10897.983 | 0.256 | 1781.765 | d0/s7 | K40 |
| 189 | 10899.765 | 0.256 | 19.968 | d0/s7 | K41 |
| 190 | 10899.785 | 0.256 | 18.240 | d0/s7 | K42 |
| 191 | 10899.803 | 0.256 | 154.208 | d0/s7 | K43 |
| 192 | 10899.958 | 0.257 | 623.009 | d0/s7 | K44 |
| 193 | 10900.581 | 0.256 | 9.088 | d0/s7 | K45 |
| 194 | 10900.590 | 0.256 | 3254.665 | d0/s7 | K46 |
| 195 | 10903.845 | 0.256 | 6.496 | d0/s7 | K47 |
| 196 | 10903.852 | 0.256 | 1461.092 | d0/s7 | K48 |
| 197 | 10905.359 | 45.825 | 9.664 | d0/s7 | K39 |
| 198 | 10905.369 | 0.256 | 1781.573 | d0/s7 | K40 |
| 199 | 10907.151 | 0.254 | 20.129 | d0/s7 | K41 |
| 200 | 10907.171 | 0.256 | 17.952 | d0/s7 | K42 |
| 201 | 10907.189 | 0.255 | 154.145 | d0/s7 | K43 |
| 202 | 10907.344 | 0.256 | 623.778 | d0/s7 | K44 |
| 203 | 10907.968 | 0.256 | 8.864 | d0/s7 | K45 |
| 204 | 10907.977 | 0.256 | 3256.809 | d0/s7 | K46 |
| 205 | 10911.234 | 0.256 | 6.496 | d0/s7 | K47 |
| 206 | 10911.241 | 0.256 | 1463.588 | d0/s7 | K48 |
| 207 | 10912.754 | 49.312 | 9.760 | d0/s7 | K39 |
| 208 | 10912.764 | 0.288 | 1779.557 | d0/s7 | K40 |
| 209 | 10914.543 | 0.256 | 20.512 | d0/s7 | K41 |
| 210 | 10914.564 | 0.256 | 18.336 | d0/s7 | K42 |
| 211 | 10914.583 | 0.288 | 154.049 | d0/s7 | K43 |
| 212 | 10914.737 | 0.255 | 622.658 | d0/s7 | K44 |
| 213 | 10915.360 | 0.255 | 8.896 | d0/s7 | K45 |
| 214 | 10915.369 | 0.256 | 3254.378 | d0/s7 | K46 |
| 215 | 10918.624 | 0.287 | 6.368 | d0/s7 | K47 |
| 216 | 10918.631 | 0.257 | 1464.131 | d0/s7 | K48 |
| 217 | 10920.138 | 43.681 | 9.376 | d0/s7 | K39 |
| 218 | 10920.148 | 0.256 | 1779.684 | d0/s7 | K40 |
| 219 | 10921.928 | 0.257 | 19.968 | d0/s7 | K41 |
| 220 | 10921.948 | 0.255 | 18.785 | d0/s7 | K42 |
| 221 | 10921.967 | 0.256 | 154.048 | d0/s7 | K43 |
| 222 | 10922.121 | 0.256 | 621.794 | d0/s7 | K44 |
| 223 | 10922.744 | 0.256 | 9.248 | d0/s7 | K45 |
| 224 | 10922.753 | 0.256 | 3255.529 | d0/s7 | K46 |
| 225 | 10926.009 | 0.256 | 6.400 | d0/s7 | K47 |
| 226 | 10926.015 | 0.256 | 1463.556 | d0/s7 | K48 |
| 227 | 10927.523 | 43.584 | 9.440 | d0/s7 | K39 |
| 228 | 10927.532 | 0.256 | 1780.005 | d0/s7 | K40 |
| 229 | 10929.313 | 0.256 | 19.968 | d0/s7 | K41 |
| 230 | 10929.333 | 0.224 | 18.816 | d0/s7 | K42 |
| 231 | 10929.352 | 0.288 | 153.825 | d0/s7 | K43 |
| 232 | 10929.506 | 0.256 | 623.233 | d0/s7 | K44 |
| 233 | 10930.129 | 0.256 | 8.960 | d0/s7 | K45 |
| 234 | 10930.139 | 0.257 | 3252.393 | d0/s7 | K46 |
| 235 | 10933.391 | 0.255 | 6.497 | d0/s7 | K47 |
| 236 | 10933.398 | 0.255 | 1463.717 | d0/s7 | K48 |
| 237 | 10934.911 | 49.312 | 9.408 | d0/s7 | K39 |
| 238 | 10934.921 | 0.288 | 1781.029 | d0/s7 | K40 |
| 239 | 10936.702 | 0.256 | 20.192 | d0/s7 | K41 |
| 240 | 10936.723 | 0.289 | 18.368 | d0/s7 | K42 |
| 241 | 10936.741 | 0.255 | 154.336 | d0/s7 | K43 |
| 242 | 10936.896 | 0.256 | 621.410 | d0/s7 | K44 |
| 243 | 10937.517 | 0.288 | 9.280 | d0/s7 | K45 |
| 244 | 10937.527 | 0.256 | 3255.625 | d0/s7 | K46 |
| 245 | 10940.783 | 0.256 | 6.496 | d0/s7 | K47 |
| 246 | 10940.790 | 0.256 | 1462.980 | d0/s7 | K48 |
| 247 | 10942.296 | 43.712 | 9.312 | d0/s7 | K39 |
| 248 | 10942.306 | 0.288 | 1781.733 | d0/s7 | K40 |
| 249 | 10944.088 | 0.256 | 20.352 | d0/s7 | K41 |
| 250 | 10944.108 | 0.224 | 18.816 | d0/s7 | K42 |
| 251 | 10944.128 | 0.257 | 153.888 | d0/s7 | K43 |
| 252 | 10944.282 | 0.256 | 622.081 | d0/s7 | K44 |
| 253 | 10944.904 | 0.257 | 9.023 | d0/s7 | K45 |
| 254 | 10944.913 | 0.257 | 3253.833 | d0/s7 | K46 |
| 255 | 10948.167 | 0.256 | 6.400 | d0/s7 | K47 |
| 256 | 10948.174 | 0.256 | 1459.652 | d0/s7 | K48 |
| 257 | 10949.679 | 45.376 | 9.504 | d0/s7 | K39 |
| 258 | 10949.689 | 0.256 | 1779.301 | d0/s7 | K40 |
| 259 | 10951.468 | 0.256 | 20.352 | d0/s7 | K41 |
| 260 | 10951.489 | 0.224 | 18.784 | d0/s7 | K42 |
| 261 | 10951.508 | 0.256 | 153.664 | d0/s7 | K43 |
| 262 | 10951.662 | 0.257 | 622.081 | d0/s7 | K44 |
| 263 | 10952.284 | 0.256 | 8.960 | d0/s7 | K45 |
| 264 | 10952.293 | 0.256 | 3255.369 | d0/s7 | K46 |
| 265 | 10955.549 | 0.256 | 6.400 | d0/s7 | K47 |
| 266 | 10955.556 | 0.256 | 1463.300 | d0/s7 | K48 |
| 267 | 10957.063 | 43.745 | 9.247 | d0/s7 | K39 |
| 268 | 10957.072 | 0.256 | 1779.333 | d0/s7 | K40 |
| 269 | 10958.852 | 0.256 | 20.193 | d0/s7 | K41 |
| 270 | 10958.872 | 0.255 | 18.337 | d0/s7 | K42 |
| 271 | 10958.891 | 0.255 | 154.337 | d0/s7 | K43 |
| 272 | 10959.046 | 0.256 | 622.882 | d0/s7 | K44 |
| 273 | 10959.669 | 0.288 | 8.672 | d0/s7 | K45 |
| 274 | 10959.678 | 0.256 | 3255.657 | d0/s7 | K46 |
| 275 | 10962.934 | 0.255 | 6.752 | d0/s7 | K47 |
| 276 | 10962.941 | 0.256 | 1457.540 | d0/s7 | K48 |
| 277 | 10964.442 | 43.808 | 9.344 | d0/s7 | K39 |
| 278 | 10964.452 | 0.256 | 1778.469 | d0/s7 | K40 |
| 279 | 10966.230 | 0.256 | 20.224 | d0/s7 | K41 |
| 280 | 10966.251 | 0.256 | 18.080 | d0/s7 | K42 |
| 281 | 10966.269 | 0.288 | 154.049 | d0/s7 | K43 |
| 282 | 10966.423 | 0.255 | 621.858 | d0/s7 | K44 |
| 283 | 10967.045 | 0.256 | 8.960 | d0/s7 | K45 |
| 284 | 10967.055 | 0.256 | 3257.033 | d0/s7 | K46 |
| 285 | 10970.312 | 0.256 | 6.752 | d0/s7 | K47 |
| 286 | 10970.319 | 0.256 | 1460.260 | d0/s7 | K48 |
| 287 | 10971.824 | 45.122 | 9.440 | d0/s7 | K39 |
| 288 | 10971.834 | 0.256 | 1782.500 | d0/s7 | K40 |
| 289 | 10973.617 | 0.257 | 20.383 | d0/s7 | K41 |
| 290 | 10973.638 | 0.289 | 17.952 | d0/s7 | K42 |
| 291 | 10973.656 | 0.257 | 154.336 | d0/s7 | K43 |
| 292 | 10973.810 | 0.256 | 620.162 | d0/s7 | K44 |
| 293 | 10974.431 | 0.288 | 8.864 | d0/s7 | K45 |
| 294 | 10974.440 | 0.224 | 3253.161 | d0/s7 | K46 |
| 295 | 10977.693 | 0.256 | 6.432 | d0/s7 | K47 |
| 296 | 10977.700 | 0.224 | 1460.772 | d0/s7 | K48 |
| 297 | 10979.205 | 44.224 | 9.376 | d0/s7 | K39 |
| 298 | 10979.215 | 0.256 | 1781.157 | d0/s7 | K40 |
| 299 | 10980.996 | 0.255 | 20.000 | d0/s7 | K41 |
| 300 | 10981.016 | 0.256 | 18.208 | d0/s7 | K42 |
| 301 | 10981.035 | 0.256 | 153.633 | d0/s7 | K43 |
| 302 | 10981.189 | 0.255 | 622.018 | d0/s7 | K44 |
| 303 | 10981.811 | 0.256 | 9.248 | d0/s7 | K45 |
| 304 | 10981.820 | 0.224 | 3252.969 | d0/s7 | K46 |
| 305 | 10985.074 | 0.256 | 6.400 | d0/s7 | K47 |
| 306 | 10985.080 | 0.256 | 1462.245 | d0/s7 | K48 |
| 307 | 10986.592 | 49.984 | 9.536 | d0/s7 | K39 |
| 308 | 10986.602 | 0.256 | 1780.581 | d0/s7 | K40 |
| 309 | 10988.383 | 0.288 | 19.936 | d0/s7 | K41 |
| 310 | 10988.403 | 0.256 | 18.272 | d0/s7 | K42 |
| 311 | 10988.422 | 0.256 | 154.272 | d0/s7 | K43 |
| 312 | 10988.576 | 0.256 | 621.250 | d0/s7 | K44 |
| 313 | 10989.198 | 0.256 | 9.024 | d0/s7 | K45 |
| 314 | 10989.207 | 0.256 | 3256.201 | d0/s7 | K46 |
| 315 | 10992.464 | 0.256 | 6.528 | d0/s7 | K47 |
| 316 | 10992.470 | 0.224 | 1460.068 | d0/s7 | K48 |
| 317 | 10993.977 | 46.528 | 9.696 | d0/s7 | K39 |
| 318 | 10993.987 | 0.288 | 1781.061 | d0/s7 | K40 |
| 319 | 10995.768 | 0.256 | 20.160 | d0/s7 | K41 |
| 320 | 10995.789 | 0.256 | 18.784 | d0/s7 | K42 |
| 321 | 10995.808 | 0.256 | 154.273 | d0/s7 | K43 |
| 322 | 10995.962 | 0.256 | 621.185 | d0/s7 | K44 |
| 323 | 10996.584 | 0.257 | 8.895 | d0/s7 | K45 |
| 324 | 10996.593 | 0.257 | 3252.936 | d0/s7 | K46 |
| 325 | 10999.846 | 0.288 | 6.497 | d0/s7 | K47 |
| 326 | 10999.853 | 0.224 | 1459.556 | d0/s7 | K48 |
| 327 | 11001.355 | 42.464 | 9.344 | d0/s7 | K39 |
| 328 | 11001.364 | 0.256 | 1779.845 | d0/s7 | K40 |
| 329 | 11003.144 | 0.256 | 20.128 | d0/s7 | K41 |
| 330 | 11003.165 | 0.256 | 18.816 | d0/s7 | K42 |
| 331 | 11003.184 | 0.256 | 154.080 | d0/s7 | K43 |
| 332 | 11003.338 | 0.288 | 620.834 | d0/s7 | K44 |
| 333 | 11003.959 | 0.288 | 8.864 | d0/s7 | K45 |
| 334 | 11003.969 | 0.256 | 3255.369 | d0/s7 | K46 |
| 335 | 11007.224 | 0.256 | 6.400 | d0/s7 | K47 |
| 336 | 11007.231 | 0.224 | 1462.596 | d0/s7 | K48 |
| 337 | 11008.737 | 43.392 | 9.088 | d0/s7 | K39 |
| 338 | 11008.746 | 0.417 | 1781.477 | d0/s7 | K40 |
| 339 | 11010.528 | 0.256 | 20.032 | d0/s7 | K41 |
| 340 | 11010.548 | 0.224 | 18.432 | d0/s7 | K42 |
| 341 | 11010.567 | 0.257 | 154.208 | d0/s7 | K43 |
| 342 | 11010.721 | 0.256 | 620.226 | d0/s7 | K44 |
| 343 | 11011.342 | 0.255 | 8.897 | d0/s7 | K45 |
| 344 | 11011.351 | 0.255 | 3254.858 | d0/s7 | K46 |
| 345 | 11014.606 | 0.256 | 6.432 | d0/s7 | K47 |
| 346 | 11014.613 | 0.255 | 1461.733 | d0/s7 | K48 |
| 347 | 11016.119 | 44.160 | 9.536 | d0/s7 | K39 |
| 348 | 11016.129 | 0.256 | 1781.477 | d0/s7 | K40 |
| 349 | 11017.910 | 0.256 | 20.128 | d0/s7 | K41 |
| 350 | 11017.931 | 0.256 | 18.272 | d0/s7 | K42 |
| 351 | 11017.949 | 0.256 | 153.760 | d0/s7 | K43 |
| 352 | 11018.103 | 0.256 | 622.018 | d0/s7 | K44 |
| 353 | 11018.725 | 0.256 | 9.152 | d0/s7 | K45 |
| 354 | 11018.735 | 0.256 | 3254.921 | d0/s7 | K46 |
| 355 | 11021.990 | 0.256 | 6.624 | d0/s7 | K47 |
| 356 | 11021.997 | 0.256 | 1464.996 | d0/s7 | K48 |
| 357 | 11023.506 | 43.712 | 9.344 | d0/s7 | K39 |
| 358 | 11023.515 | 0.256 | 1780.037 | d0/s7 | K40 |
| 359 | 11025.296 | 0.256 | 20.097 | d0/s7 | K41 |
| 360 | 11025.316 | 0.255 | 18.112 | d0/s7 | K42 |
| 361 | 11025.334 | 0.257 | 153.984 | d0/s7 | K43 |
| 362 | 11025.489 | 0.288 | 622.018 | d0/s7 | K44 |
| 363 | 11026.111 | 0.288 | 8.896 | d0/s7 | K45 |
| 364 | 11026.120 | 0.256 | 3255.721 | d0/s7 | K46 |
| 365 | 11029.376 | 0.256 | 6.496 | d0/s7 | K47 |
| 366 | 11029.383 | 0.256 | 1458.308 | d0/s7 | K48 |
| 367 | 11030.882 | 41.280 | 9.184 | d0/s7 | K39 |
| 368 | 11030.892 | 0.256 | 1779.941 | d0/s7 | K40 |
| 369 | 11032.672 | 0.288 | 20.096 | d0/s7 | K41 |
| 370 | 11032.692 | 0.256 | 18.176 | d0/s7 | K42 |
| 371 | 11032.711 | 0.256 | 154.145 | d0/s7 | K43 |
| 372 | 11032.865 | 0.255 | 620.642 | d0/s7 | K44 |
| 373 | 11033.486 | 0.256 | 8.800 | d0/s7 | K45 |
| 374 | 11033.495 | 0.256 | 3256.169 | d0/s7 | K46 |
| 375 | 11036.752 | 0.256 | 6.400 | d0/s7 | K47 |
| 376 | 11036.758 | 0.256 | 1460.324 | d0/s7 | K48 |
| 377 | 11038.266 | 47.104 | 9.248 | d0/s7 | K39 |
| 378 | 11038.275 | 0.288 | 1777.957 | d0/s7 | K40 |
| 379 | 11040.053 | 0.256 | 19.937 | d0/s7 | K41 |
| 380 | 11040.074 | 0.255 | 18.785 | d0/s7 | K42 |
| 381 | 11040.093 | 0.255 | 153.633 | d0/s7 | K43 |
| 382 | 11040.247 | 0.289 | 620.706 | d0/s7 | K44 |
| 383 | 11040.868 | 0.288 | 8.928 | d0/s7 | K45 |
| 384 | 11040.877 | 0.256 | 3256.169 | d0/s7 | K46 |
| 385 | 11044.133 | 0.256 | 6.368 | d0/s7 | K47 |
| 386 | 11044.140 | 0.256 | 1463.396 | d0/s7 | K48 |
| 387 | 11045.648 | 44.736 | 9.568 | d0/s7 | K39 |
| 388 | 11045.658 | 0.288 | 1781.029 | d0/s7 | K40 |
| 389 | 11047.439 | 0.256 | 20.352 | d0/s7 | K41 |
| 390 | 11047.460 | 0.256 | 18.304 | d0/s7 | K42 |
| 391 | 11047.478 | 0.256 | 153.920 | d0/s7 | K43 |
| 392 | 11047.632 | 0.257 | 622.913 | d0/s7 | K44 |
| 393 | 11048.256 | 0.256 | 9.024 | d0/s7 | K45 |
| 394 | 11048.265 | 0.256 | 3258.441 | d0/s7 | K46 |
| 395 | 11051.524 | 0.256 | 6.496 | d0/s7 | K47 |
| 396 | 11051.530 | 0.224 | 1463.013 | d0/s7 | K48 |
| 397 | 11053.040 | 46.912 | 9.568 | d0/s7 | K39 |
| 398 | 11053.050 | 0.256 | 1779.141 | d0/s7 | K40 |
| 399 | 11054.829 | 0.256 | 20.416 | d0/s7 | K41 |
| 400 | 11054.850 | 0.256 | 18.112 | d0/s7 | K42 |
| 401 | 11054.868 | 0.256 | 153.888 | d0/s7 | K43 |
| 402 | 11055.023 | 0.255 | 624.002 | d0/s7 | K44 |
| 403 | 11055.647 | 0.256 | 9.152 | d0/s7 | K45 |
| 404 | 11055.656 | 0.256 | 3254.473 | d0/s7 | K46 |
| 405 | 11058.911 | 0.256 | 6.496 | d0/s7 | K47 |
| 406 | 11058.918 | 0.256 | 1459.780 | d0/s7 | K48 |
| 407 | 11060.420 | 42.912 | 9.408 | d0/s7 | K39 |
| 408 | 11060.430 | 0.256 | 1782.597 | d0/s7 | K40 |
| 409 | 11062.213 | 0.256 | 20.096 | d0/s7 | K41 |
| 410 | 11062.233 | 0.224 | 18.112 | d0/s7 | K42 |
| 411 | 11062.252 | 0.256 | 154.081 | d0/s7 | K43 |
| 412 | 11062.406 | 0.256 | 624.802 | d0/s7 | K44 |
| 413 | 11063.031 | 0.255 | 8.928 | d0/s7 | K45 |
| 414 | 11063.040 | 0.256 | 3252.520 | d0/s7 | K46 |
| 415 | 11066.293 | 0.256 | 6.400 | d0/s7 | K47 |
| 416 | 11066.300 | 0.257 | 1463.555 | d0/s7 | K48 |
| 417 | 11067.808 | 44.417 | 9.216 | d0/s7 | K39 |
| 418 | 11067.817 | 0.256 | 1778.885 | d0/s7 | K40 |
| 419 | 11069.596 | 0.256 | 20.064 | d0/s7 | K41 |
| 420 | 11069.617 | 0.256 | 18.560 | d0/s7 | K42 |
| 421 | 11069.635 | 0.256 | 154.336 | d0/s7 | K43 |
| 422 | 11069.790 | 0.256 | 622.338 | d0/s7 | K44 |
| 423 | 11070.413 | 0.256 | 9.024 | d0/s7 | K45 |
| 424 | 11070.422 | 0.256 | 3253.353 | d0/s7 | K46 |
| 425 | 11073.675 | 0.256 | 6.400 | d0/s7 | K47 |
| 426 | 11073.682 | 0.256 | 1458.724 | d0/s7 | K48 |
| 427 | 11075.187 | 46.048 | 9.376 | d0/s7 | K39 |
| 428 | 11075.196 | 0.256 | 1780.965 | d0/s7 | K40 |
| 429 | 11076.978 | 0.256 | 19.904 | d0/s7 | K41 |
| 430 | 11076.998 | 0.256 | 18.273 | d0/s7 | K42 |
| 431 | 11077.016 | 0.255 | 154.209 | d0/s7 | K43 |
| 432 | 11077.171 | 0.256 | 622.242 | d0/s7 | K44 |
| 433 | 11077.793 | 0.288 | 8.960 | d0/s7 | K45 |
| 434 | 11077.803 | 0.255 | 3256.490 | d0/s7 | K46 |
| 435 | 11081.059 | 0.256 | 6.496 | d0/s7 | K47 |
| 436 | 11081.066 | 0.256 | 1461.828 | d0/s7 | K48 |
| 437 | 11082.579 | 51.200 | 9.696 | d0/s7 | K39 |
| 438 | 11082.589 | 0.256 | 1782.245 | d0/s7 | K40 |
| 439 | 11084.372 | 0.256 | 20.352 | d0/s7 | K41 |
| 440 | 11084.392 | 0.256 | 18.816 | d0/s7 | K42 |
| 441 | 11084.411 | 0.256 | 154.528 | d0/s7 | K43 |
| 442 | 11084.566 | 0.257 | 621.409 | d0/s7 | K44 |
| 443 | 11085.188 | 0.256 | 8.736 | d0/s7 | K45 |
| 444 | 11085.197 | 0.256 | 3252.329 | d0/s7 | K46 |
| 445 | 11088.449 | 0.256 | 6.464 | d0/s7 | K47 |
| 446 | 11088.456 | 0.256 | 1460.612 | d0/s7 | K48 |
| 447 | 11089.959 | 42.625 | 9.343 | d0/s7 | K39 |
| 448 | 11089.969 | 0.257 | 1779.845 | d0/s7 | K40 |
| 449 | 11091.749 | 0.254 | 20.096 | d0/s7 | K41 |
| 450 | 11091.769 | 0.224 | 17.985 | d0/s7 | K42 |
| 451 | 11091.788 | 0.255 | 153.857 | d0/s7 | K43 |
| 452 | 11091.942 | 0.288 | 621.985 | d0/s7 | K44 |
| 453 | 11092.564 | 0.257 | 8.800 | d0/s7 | K45 |
| 454 | 11092.573 | 0.256 | 3251.849 | d0/s7 | K46 |
| 455 | 11095.825 | 0.256 | 6.400 | d0/s7 | K47 |
| 456 | 11095.832 | 0.224 | 1463.012 | d0/s7 | K48 |
| 457 | 11097.345 | 50.208 | 9.152 | d0/s7 | K39 |
| 458 | 11097.354 | 0.256 | 1779.717 | d0/s7 | K40 |
| 459 | 11099.134 | 0.256 | 19.872 | d0/s7 | K41 |
| 460 | 11099.154 | 0.256 | 18.240 | d0/s7 | K42 |
| 461 | 11099.173 | 0.255 | 153.825 | d0/s7 | K43 |
| 462 | 11099.327 | 0.255 | 620.834 | d0/s7 | K44 |
| 463 | 11099.948 | 0.256 | 8.960 | d0/s7 | K45 |
| 464 | 11099.957 | 0.256 | 3254.889 | d0/s7 | K46 |
| 465 | 11103.212 | 0.288 | 6.528 | d0/s7 | K47 |
| 466 | 11103.219 | 0.226 | 1459.715 | d0/s7 | K48 |
| 467 | 11104.724 | 45.312 | 9.473 | d0/s7 | K39 |
| 468 | 11104.734 | 0.287 | 1781.221 | d0/s7 | K40 |
| 469 | 11106.515 | 0.257 | 19.968 | d0/s7 | K41 |
| 470 | 11106.536 | 0.255 | 18.817 | d0/s7 | K42 |
| 471 | 11106.555 | 0.255 | 154.625 | d0/s7 | K43 |
| 472 | 11106.710 | 0.256 | 622.690 | d0/s7 | K44 |
| 473 | 11107.333 | 0.256 | 8.928 | d0/s7 | K45 |
| 474 | 11107.342 | 0.256 | 3254.793 | d0/s7 | K46 |
| 475 | 11110.597 | 0.256 | 6.496 | d0/s7 | K47 |
| 476 | 11110.604 | 0.256 | 1462.852 | d0/s7 | K48 |
| 477 | 11112.111 | 44.256 | 9.728 | d0/s7 | K39 |
| 478 | 11112.121 | 0.256 | 1781.445 | d0/s7 | K40 |
| 479 | 11113.902 | 0.256 | 20.704 | d0/s7 | K41 |
| 480 | 11113.923 | 0.256 | 18.336 | d0/s7 | K42 |
| 481 | 11113.942 | 0.288 | 154.337 | d0/s7 | K43 |
| 482 | 11114.097 | 0.256 | 624.865 | d0/s7 | K44 |
| 483 | 11114.722 | 0.256 | 8.960 | d0/s7 | K45 |
| 484 | 11114.731 | 0.257 | 3255.305 | d0/s7 | K46 |
| 485 | 11117.986 | 0.255 | 6.496 | d0/s7 | K47 |
| 486 | 11117.993 | 0.257 | 1461.380 | d0/s7 | K48 |
| 487 | 11119.498 | 43.744 | 9.344 | d0/s7 | K39 |
| 488 | 11119.508 | 0.256 | 1779.589 | d0/s7 | K40 |
| 489 | 11121.288 | 0.255 | 20.097 | d0/s7 | K41 |
| 490 | 11121.308 | 0.256 | 18.240 | d0/s7 | K42 |
| 491 | 11121.327 | 0.256 | 154.176 | d0/s7 | K43 |
| 492 | 11121.481 | 0.289 | 621.410 | d0/s7 | K44 |
| 493 | 11122.103 | 0.255 | 8.992 | d0/s7 | K45 |
| 494 | 11122.112 | 0.256 | 3255.113 | d0/s7 | K46 |
| 495 | 11125.367 | 0.256 | 6.368 | d0/s7 | K47 |
| 496 | 11125.374 | 0.256 | 1466.692 | d0/s7 | K48 |
| 497 | 11126.884 | 43.104 | 9.440 | d0/s7 | K39 |
| 498 | 11126.894 | 0.288 | 1781.957 | d0/s7 | K40 |
| 499 | 11128.676 | 0.288 | 20.128 | d0/s7 | K41 |
| 500 | 11128.696 | 0.256 | 18.752 | d0/s7 | K42 |
| 501 | 11128.715 | 0.288 | 153.985 | d0/s7 | K43 |
| 502 | 11128.869 | 0.256 | 621.313 | d0/s7 | K44 |
| 503 | 11129.491 | 0.257 | 9.023 | d0/s7 | K45 |
| 504 | 11129.500 | 0.257 | 3254.185 | d0/s7 | K46 |
| 505 | 11132.755 | 0.256 | 6.400 | d0/s7 | K47 |
| 506 | 11132.761 | 0.255 | 1462.820 | d0/s7 | K48 |
| 507 | 11134.271 | 47.009 | 9.600 | d0/s7 | K39 |
| 508 | 11134.281 | 0.256 | 1780.549 | d0/s7 | K40 |
| 509 | 11136.062 | 0.288 | 19.968 | d0/s7 | K41 |
| 510 | 11136.082 | 0.225 | 18.176 | d0/s7 | K42 |
| 511 | 11136.101 | 0.256 | 153.920 | d0/s7 | K43 |
| 512 | 11136.255 | 0.256 | 621.506 | d0/s7 | K44 |
| 513 | 11136.876 | 0.256 | 8.928 | d0/s7 | K45 |
| 514 | 11136.886 | 0.256 | 3255.273 | d0/s7 | K46 |
| 515 | 11140.141 | 0.256 | 6.496 | d0/s7 | K47 |
| 516 | 11140.148 | 0.256 | 1461.700 | d0/s7 | K48 |
| 517 | 11141.654 | 44.479 | 9.280 | d0/s7 | K39 |
| 518 | 11141.664 | 0.256 | 1780.869 | d0/s7 | K40 |
| 519 | 11143.445 | 0.288 | 20.480 | d0/s7 | K41 |
| 520 | 11143.466 | 0.256 | 18.752 | d0/s7 | K42 |
| 521 | 11143.485 | 0.256 | 154.433 | d0/s7 | K43 |
| 522 | 11143.639 | 0.256 | 625.762 | d0/s7 | K44 |
| 523 | 11144.265 | 0.256 | 9.024 | d0/s7 | K45 |
| 524 | 11144.275 | 0.256 | 3254.121 | d0/s7 | K46 |
| 525 | 11147.529 | 0.256 | 6.528 | d0/s7 | K47 |
| 526 | 11147.536 | 0.224 | 1458.116 | d0/s7 | K48 |
| 527 | 11149.037 | 43.232 | 9.249 | d0/s7 | K39 |
| 528 | 11149.047 | 0.287 | 1781.605 | d0/s7 | K40 |
| 529 | 11150.828 | 0.256 | 20.224 | d0/s7 | K41 |
| 530 | 11150.849 | 0.256 | 18.048 | d0/s7 | K42 |
| 531 | 11150.867 | 0.257 | 154.048 | d0/s7 | K43 |
| 532 | 11151.021 | 0.256 | 623.586 | d0/s7 | K44 |
| 533 | 11151.645 | 0.256 | 9.088 | d0/s7 | K45 |
| 534 | 11151.655 | 0.256 | 3255.369 | d0/s7 | K46 |
| 535 | 11154.910 | 0.256 | 6.432 | d0/s7 | K47 |
| 536 | 11154.917 | 0.224 | 1465.188 | d0/s7 | K48 |
| 537 | 11156.426 | 43.712 | 9.312 | d0/s7 | K39 |
| 538 | 11156.435 | 0.256 | 1781.541 | d0/s7 | K40 |
| 539 | 11158.217 | 0.288 | 20.288 | d0/s7 | K41 |
| 540 | 11158.238 | 0.256 | 18.432 | d0/s7 | K42 |
| 541 | 11158.256 | 0.256 | 153.889 | d0/s7 | K43 |
| 542 | 11158.411 | 0.256 | 620.737 | d0/s7 | K44 |
| 543 | 11159.032 | 0.257 | 8.831 | d0/s7 | K45 |
| 544 | 11159.041 | 0.256 | 3254.986 | d0/s7 | K46 |
| 545 | 11162.296 | 0.288 | 6.367 | d0/s7 | K47 |
| 546 | 11162.303 | 0.257 | 1462.563 | d0/s7 | K48 |
| 547 | 11163.809 | 44.001 | 9.632 | d0/s7 | K39 |
| 548 | 11163.819 | 0.256 | 1777.957 | d0/s7 | K40 |
| 549 | 11165.597 | 0.256 | 19.936 | d0/s7 | K41 |
| 550 | 11165.617 | 0.256 | 18.784 | d0/s7 | K42 |
| 551 | 11165.637 | 0.256 | 154.112 | d0/s7 | K43 |
| 552 | 11165.791 | 0.288 | 623.394 | d0/s7 | K44 |
| 553 | 11166.415 | 0.256 | 8.896 | d0/s7 | K45 |
| 554 | 11166.424 | 0.256 | 3255.849 | d0/s7 | K46 |
| 555 | 11169.680 | 0.256 | 6.496 | d0/s7 | K47 |
| 556 | 11169.687 | 0.224 | 1461.444 | d0/s7 | K48 |
| 557 | 11171.195 | 46.849 | 9.535 | d0/s7 | K39 |
| 558 | 11171.205 | 0.256 | 1781.477 | d0/s7 | K40 |
| 559 | 11172.986 | 0.256 | 20.544 | d0/s7 | K41 |
| 560 | 11173.007 | 0.257 | 18.143 | d0/s7 | K42 |
| 561 | 11173.026 | 0.257 | 154.336 | d0/s7 | K43 |
| 562 | 11173.180 | 0.256 | 621.826 | d0/s7 | K44 |
| 563 | 11173.802 | 0.256 | 9.344 | d0/s7 | K45 |
| 564 | 11173.812 | 0.256 | 3256.073 | d0/s7 | K46 |
| 565 | 11177.068 | 0.288 | 6.464 | d0/s7 | K47 |
| 566 | 11177.075 | 0.256 | 1462.340 | d0/s7 | K48 |
| 567 | 11178.587 | 49.312 | 9.216 | d0/s7 | K39 |
| 568 | 11178.596 | 0.256 | 1779.653 | d0/s7 | K40 |
| 569 | 11180.376 | 0.256 | 20.096 | d0/s7 | K41 |
| 570 | 11180.396 | 0.256 | 18.496 | d0/s7 | K42 |
| 571 | 11180.415 | 0.256 | 153.505 | d0/s7 | K43 |
| 572 | 11180.569 | 0.256 | 620.642 | d0/s7 | K44 |
| 573 | 11181.190 | 0.288 | 8.832 | d0/s7 | K45 |
| 574 | 11181.199 | 0.288 | 3254.953 | d0/s7 | K46 |
| 575 | 11184.454 | 0.256 | 6.336 | d0/s7 | K47 |
| 576 | 11184.461 | 0.256 | 1462.852 | d0/s7 | K48 |
| 577 | 11185.967 | 43.904 | 9.216 | d0/s7 | K39 |
| 578 | 11185.977 | 0.256 | 1780.517 | d0/s7 | K40 |
| 579 | 11187.758 | 0.256 | 19.968 | d0/s7 | K41 |
| 580 | 11187.778 | 0.256 | 18.752 | d0/s7 | K42 |
| 581 | 11187.797 | 0.256 | 153.408 | d0/s7 | K43 |
| 582 | 11187.951 | 0.257 | 623.617 | d0/s7 | K44 |
| 583 | 11188.574 | 0.256 | 8.960 | d0/s7 | K45 |
| 584 | 11188.584 | 0.256 | 3256.361 | d0/s7 | K46 |
| 585 | 11191.840 | 0.256 | 6.368 | d0/s7 | K47 |
| 586 | 11191.847 | 0.256 | 1460.773 | d0/s7 | K48 |
| 587 | 11193.351 | 43.264 | 9.471 | d0/s7 | K39 |
| 588 | 11193.361 | 0.256 | 1780.453 | d0/s7 | K40 |
| 589 | 11195.141 | 0.256 | 19.904 | d0/s7 | K41 |
| 590 | 11195.162 | 0.256 | 18.304 | d0/s7 | K42 |
| 591 | 11195.180 | 0.256 | 153.888 | d0/s7 | K43 |
| 592 | 11195.334 | 0.288 | 621.058 | d0/s7 | K44 |
| 593 | 11195.956 | 0.256 | 9.056 | d0/s7 | K45 |
| 594 | 11195.965 | 0.256 | 3253.257 | d0/s7 | K46 |
| 595 | 11199.218 | 0.288 | 6.528 | d0/s7 | K47 |
| 596 | 11199.225 | 0.256 | 1463.268 | d0/s7 | K48 |
| 597 | 11200.737 | 48.864 | 9.473 | d0/s7 | K39 |
| 598 | 11200.747 | 0.255 | 1780.198 | d0/s7 | K40 |
| 599 | 11202.528 | 0.254 | 20.256 | d0/s7 | K41 |
| 600 | 11202.548 | 0.256 | 18.080 | d0/s7 | K42 |
| 601 | 11202.566 | 0.256 | 153.921 | d0/s7 | K43 |
| 602 | 11202.721 | 0.256 | 621.090 | d0/s7 | K44 |
| 603 | 11203.342 | 0.256 | 8.800 | d0/s7 | K45 |
| 604 | 11203.351 | 0.224 | 3255.273 | d0/s7 | K46 |
| 605 | 11206.606 | 0.256 | 6.496 | d0/s7 | K47 |
| 606 | 11206.613 | 0.256 | 1463.812 | d0/s7 | K48 |
| 607 | 11208.120 | 42.688 | 9.344 | d0/s7 | K39 |
| 608 | 11208.129 | 0.256 | 1780.805 | d0/s7 | K40 |
| 609 | 11209.910 | 0.256 | 20.064 | d0/s7 | K41 |
| 610 | 11209.931 | 0.256 | 18.304 | d0/s7 | K42 |
| 611 | 11209.949 | 0.256 | 153.984 | d0/s7 | K43 |
| 612 | 11210.103 | 0.257 | 622.881 | d0/s7 | K44 |
| 613 | 11210.727 | 0.256 | 8.737 | d0/s7 | K45 |
| 614 | 11210.736 | 0.255 | 3254.762 | d0/s7 | K46 |
| 615 | 11213.991 | 0.255 | 6.367 | d0/s7 | K47 |
| 616 | 11213.997 | 0.257 | 1457.636 | d0/s7 | K48 |
| 617 | 11215.498 | 43.296 | 9.184 | d0/s7 | K39 |
| 618 | 11215.508 | 0.288 | 1781.029 | d0/s7 | K40 |
| 619 | 11217.289 | 0.256 | 20.000 | d0/s7 | K41 |
| 620 | 11217.309 | 0.256 | 18.240 | d0/s7 | K42 |
| 621 | 11217.328 | 0.256 | 154.016 | d0/s7 | K43 |
| 622 | 11217.482 | 0.256 | 622.978 | d0/s7 | K44 |
| 623 | 11218.105 | 0.256 | 8.960 | d0/s7 | K45 |
| 624 | 11218.114 | 0.224 | 3255.241 | d0/s7 | K46 |
| 625 | 11221.370 | 0.288 | 6.368 | d0/s7 | K47 |
| 626 | 11221.377 | 0.256 | 1463.620 | d0/s7 | K48 |
| 627 | 11996.585 | 773744.411 | 4.896 | d0/s7 | K49 |
| 628 | 11996.752 | 162.496 | 1.760 | d0/s7 | K50 |
| 629 | 11997.105 | 351.073 | 1.600 | d0/s7 | K51 |
| 630 | 12997.659 | 1000552.439 | 8.448 | d0/s7 | K52 |
| 631 | 13916.666 | 918998.706 | 1.664 | d0/s7 | K53 |
| 632 | 15433.575 | 1516907.678 | 1.696 | d0/s7 | K54 |
| 633 | 15433.799 | 221.824 | 2.656 | d0/s7 | K55 |
| 634 | 15433.899 | 97.024 | 2.112 | d0/s7 | K56 |
| 635 | 15433.918 | 17.345 | 2.400 | d0/s7 | K55 |
| 636 | 15434.045 | 124.256 | 1.792 | d0/s7 | K24 |
| 637 | 15434.294 | 247.329 | 1.792 | d0/s7 | K57 |
| 638 | 15434.540 | 244.064 | 1.920 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 639 | 15434.682 | 140.449 | 1.728 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 640 | 15446.727 | 12043.266 | 6.975 | d0/s7 | K58 |
| 641 | 15446.902 | 168.129 | 1.024 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 642 | 15446.959 | 55.520 | 1.024 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 643 | 15448.437 | 1477.508 | 4.320 | d0/s7 | K59 |
| 644 | 15448.530 | 88.609 | 1.056 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 645 | 15448.584 | 52.416 | 1.920 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 646 | 15448.708 | 122.912 | 3.936 | d0/s7 | K59 |
| 647 | 15448.754 | 41.664 | 2.368 | d0/s7 | K28 |
| 648 | 15449.089 | 332.769 | 2.496 | d0/s7 | K60 |
| 649 | 15449.120 | 28.448 | 1.632 | d0/s7 | K31 |
| 650 | 15449.140 | 18.560 | 1.632 | d0/s7 | K61 |
| 651 | 15449.152 | 10.560 | 1.600 | d0/s7 | K31 |
| 652 | 15449.166 | 11.712 | 1.664 | d0/s7 | K61 |
| 653 | 15449.208 | 40.929 | 4.415 | d0/s7 | K62 |
| 654 | 15449.274 | 61.409 | 2.272 | d0/s7 | K63 |
| 655 | 15449.725 | 448.065 | 2.848 | d0/s7 | K64 |
| 656 | 15449.797 | 69.312 | 3.136 | d0/s7 | K65 |
| 657 | 15449.831 | 30.848 | 1.568 | d0/s7 | K66 |
| 658 | 15449.858 | 25.376 | 1.408 | d0/s7 | K67 |
| 659 | 15449.879 | 19.711 | 2.976 | d0/s7 | K68 |
| 660 | 15449.898 | 16.352 | 1.664 | d0/s7 | K53 |
| 661 | 15449.908 | 8.704 | 3.040 | d0/s7 | K68 |
| 662 | 15449.943 | 31.105 | 3.904 | d0/s7 | K69 |
| 663 | 15450.204 | 257.888 | 1.984 | d0/s7 | K70 |
| 664 | 15450.260 | 53.504 | 1.793 | d0/s7 | K71 |
| 665 | 15450.279 | 17.247 | 2.688 | d0/s7 | K72 |
| 666 | 15450.292 | 10.720 | 1.760 | d0/s7 | K53 |
| 667 | 15450.304 | 9.952 | 1.600 | d0/s7 | K54 |
| 668 | 15450.315 | 9.537 | 2.208 | d0/s7 | K73 |
| 669 | 15450.326 | 8.192 | 2.048 | d0/s7 | K53 |
| 670 | 15450.335 | 7.488 | 2.080 | d0/s7 | K73 |
| 671 | 16282.249 | 831911.835 | 2.176 | d0/s7 | K74 |
| 672 | 16282.499 | 247.649 | 3.232 | d0/s7 | K75 |
| 673 | 16282.586 | 83.904 | 2.496 | d0/s7 | K73 |
| 674 | 16282.605 | 16.896 | 2.400 | d0/s7 | K73 |
| 675 | 16282.684 | 76.352 | 1.760 | d0/s7 | K76 |
| 676 | 16282.718 | 32.000 | 4.769 | d0/s7 | K75 |
| 677 | 16282.757 | 33.952 | 2.143 | d0/s7 | K77 |
| 678 | 16282.773 | 14.720 | 2.785 | d0/s7 | K75 |
| 679 | 16282.790 | 13.855 | 2.337 | d0/s7 | K78 |
| 680 | 16282.804 | 11.456 | 2.304 | d0/s7 | K78 |
| 681 | 16282.819 | 12.640 | 1.600 | d0/s7 | K79 |
| 682 | 16282.836 | 15.648 | 3.072 | d0/s7 | K75 |
| 683 | 16282.870 | 30.816 | 1.920 | d0/s7 | K34 |
| 684 | 16282.918 | 45.664 | 2.016 | d0/s7 | K80 |
| 685 | 16282.931 | 11.200 | 1.792 | d0/s7 | K31 |
| 686 | 16282.944 | 11.936 | 2.016 | d0/s7 | K61 |
| 687 | 16282.966 | 19.328 | 4.000 | d0/s7 | K81 |
| 688 | 16282.988 | 18.080 | 1.984 | d0/s7 | K82 |
| 689 | 16283.001 | 10.688 | 3.392 | d0/s7 | K69 |
| 690 | 16283.015 | 10.688 | 1.632 | d0/s7 | K70 |
| 691 | 16283.281 | 264.865 | 2.976 | d0/s7 | K83 |
| 692 | 16283.335 | 51.040 | 3.776 | d0/s7 | K69 |
| 693 | 16283.347 | 8.032 | 1.824 | d0/s7 | K70 |
| 694 | 16283.377 | 27.872 | 2.816 | d0/s7 | K83 |
| 695 | 21331.081 | 5047701.040 | 7.903 | d0/s7 | K84 |
| 696 | 21331.635 | 546.530 | 1.888 | d0/s7 | K85 |
| 697 | 21332.086 | 449.282 | 1.216 | d0/s7 | K86 |
| 698 | 21333.944 | 1856.452 | 2.177 | d0/s7 | K87 |
| 699 | 21334.241 | 294.784 | 1.697 | d0/s7 | Memcpy DtoD (Device -> Device) (1048576 B) |
| 700 | 26298.443 | 4964200.728 | 2.465 | d0/s7 | K88 |
| 701 | 27276.845 | 978399.603 | 3.424 | d0/s7 | K89 |
| 702 | 27278.643 | 1794.405 | 6.592 | d0/s7 | K90 |
| 703 | 27817.770 | 539120.264 | 2.912 | d0/s7 | K91 |
| 704 | 27817.929 | 155.745 | 3.264 | d0/s7 | K68 |
| 705 | 27818.049 | 117.600 | 3.009 | d0/s7 | K83 |
| 706 | 27818.202 | 149.984 | 1.952 | d0/s7 | K34 |
| 707 | 27818.292 | 87.264 | 2.016 | d0/s7 | K80 |
| 708 | 27818.401 | 107.392 | 1.888 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 709 | 27818.466 | 63.296 | 1.888 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 710 | 27820.252 | 1783.365 | 5.728 | d0/s7 | K92 |
| 711 | 27820.377 | 119.713 | 2.272 | d0/s7 | K93 |
| 712 | 27820.480 | 101.216 | 1.888 | d0/s7 | K51 |
| 713 | 27820.571 | 88.576 | 8.897 | d0/s7 | K52 |
| 714 | 27820.609 | 29.440 | 1.792 | d0/s7 | K53 |
| 715 | 27820.654 | 42.976 | 1.536 | d0/s7 | K54 |
| 716 | 27820.678 | 22.368 | 2.592 | d0/s7 | K55 |
| 717 | 27820.693 | 12.864 | 1.888 | d0/s7 | K56 |
| 718 | 27820.705 | 9.248 | 2.400 | d0/s7 | K55 |
| 719 | 27820.730 | 23.360 | 1.920 | d0/s7 | K24 |
| 720 | 27820.787 | 55.136 | 2.048 | d0/s7 | K57 |
| 721 | 27820.843 | 53.504 | 2.112 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 722 | 27820.878 | 32.672 | 1.728 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 723 | 27822.503 | 1623.429 | 7.680 | d0/s7 | K94 |
| 724 | 27822.621 | 110.944 | 1.952 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 725 | 27822.659 | 35.104 | 1.952 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 726 | 27822.779 | 118.625 | 8.223 | d0/s7 | K94 |
| 727 | 27822.819 | 31.937 | 2.560 | d0/s7 | K37 |
| 728 | 27822.849 | 27.552 | 2.496 | d0/s7 | K95 |
| 729 | 27822.863 | 11.456 | 2.240 | d0/s7 | K37 |
| 730 | 27822.879 | 13.184 | 2.432 | d0/s7 | K95 |
| 731 | 27822.928 | 47.072 | 5.056 | d0/s7 | K96 |
| 732 | 27822.968 | 34.240 | 2.336 | d0/s7 | K97 |
| 733 | 28547.393 | 724422.705 | 2.880 | d0/s7 | K98 |
| 734 | 28547.521 | 125.280 | 5.088 | d0/s7 | K96 |
| 735 | 28547.583 | 57.185 | 2.272 | d0/s7 | K97 |
| 736 | 28547.748 | 163.104 | 2.465 | d0/s7 | K99 |
| 737 | 28547.777 | 26.208 | 2.272 | d0/s7 | K37 |
| 738 | 28547.813 | 33.920 | 2.496 | d0/s7 | K95 |
| 739 | 28547.865 | 49.344 | 1.056 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 740 | 28547.905 | 38.464 | 1.888 | d0/s7 | Memcpy HtoD (Pageable -> Device) (4 B) |
| 741 | 28548.155 | 248.065 | 9.664 | d0/s7 | K92 |
| 742 | 28548.199 | 35.168 | 2.272 | d0/s7 | K93 |
| 743 | 28548.299 | 97.024 | 2.400 | d0/s7 | K93 |
| 744 | 28750.287 | 201986.071 | 1.888 | d0/s7 | K100 |
| 745 | 28750.477 | 188.224 | 0.736 | d0/s7 | Memset (Device) (4 B) |
| 746 | 29829.783 | 1079304.756 | 5.824 | d0/s7 | K101 |
| 747 | 29831.373 | 1584.516 | 5.440 | d0/s7 | Memcpy DtoH (Device -> Pinned) (4 B) |
| 748 | 29831.745 | 366.785 | 2.817 | d0/s7 | K100 |
| 749 | 31313.740 | 1481992.158 | 1.600 | d0/s7 | K102 |
| 750 | 31313.904 | 161.984 | 1.696 | d0/s7 | K103 |
| 751 | 31314.072 | 166.145 | 1.792 | d0/s7 | K104 |
| 752 | 31314.212 | 138.368 | 1.696 | d0/s7 | Memset (Device) (4 B) |
| 753 | 32471.250 | 1157036.051 | 5.728 | d0/s7 | K105 |
| 754 | 32471.415 | 159.744 | 2.720 | d0/s7 | Memcpy DtoH (Device -> Pinned) (1 B) |
