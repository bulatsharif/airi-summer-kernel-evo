# Level 1 / 01 — Square matrix multiplication на CuTe FP8

Контракт задачи:

```text
A: [4096, 4096], FP8 E4M3FN
B: [4096, 4096], FP8 E4M3FN
C = A @ B: [4096, 4096], FP32
```

В памяти правый операнд хранится как `B_nk = B.T`, то есть в форме `[N, K]`.
Так обе FP8-матрицы имеют K-major layout, подходящий Blackwell MMA.

## Что делает ядро

1. `@cute.jit` выбирает FP8 `tcgen05` MMA atom `128 x 256 x 32`.
2. Один CTA считает output tile `128 x 256`, проходя K блоками по 128.
3. TMA асинхронно копирует FP8 A/B из GMEM в четырёхстадийный SMEM pipeline.
4. `cute.gemm` выполняет MMA, а FP32 accumulator живёт в TMEM.
5. Epilogue читает TMEM в регистры, умножает на `scale_a * scale_b` и пишет FP32.

Сама математика GEMM находится в `cute.gemm`; всё вокруг неё — организация
layout, копирований, pipeline и epilogue.

## Проверка на B300

```text
shape=(4096, 4096)
full_max_abs_vs_torch_fp8=0.000122
sample_max_rel_vs_fp32=0.001740
device_time_ms=1.291013
profile_id=537f77d8-b040-4420-87c3-38d639573efd
PASS
```

`device_time_ms` относится ко всему профилируемому `main()`, а не является
изолированным benchmark ядра.

Trace: [537f77d8-b040-4420-87c3-38d639573efd.json](../../profiles/537f77d8-b040-4420-87c3-38d639573efd.json).

Код основан на NVIDIA CuTe DSL Blackwell tutorial GEMM; условия лицензии:
[NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
