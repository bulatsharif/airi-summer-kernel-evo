# Level 2 / 76 — GEMM + BiasAdd + ReLU на CuTe FP8

Контракт:

```text
X: [1024, 8192], FP8 E4M3FN
W: [8192, 8192], FP8 E4M3FN
bias: [8192], FP32
Y = relu(X @ W.T + bias): [1024, 8192], FP32
```

Correctness-first реализация состоит из двух CuTe launch:

1. Blackwell `tcgen05` FP8 GEMM с TMA/SMEM pipeline и FP32 TMEM accumulator.
2. Простое FP32 ядро, которое добавляет `bias[column]` и применяет ReLU.

Такой вариант буквально повторяет три операции исходной KernelBench-задачи и
даёт удобную контрольную точку. Следующая оптимизация — перенести bias и ReLU
в TMEM epilogue GEMM, убрав второй launch и промежуточную запись/чтение.

## Проверка на B300

```text
shape=(1024, 8192)
full_max_abs_vs_torch_fp8=0.000000
full_mean_abs_vs_torch_fp8=0.000000000
sample_max_abs_vs_fp32=0.029657
device_time_ms=1.799175
profile_id=e1ba95a0-c7d2-4ea4-91fe-b5d2d216fa23
PASS
```

Trace: [e1ba95a0-c7d2-4ea4-91fe-b5d2d216fa23.json](../../profiles/e1ba95a0-c7d2-4ea4-91fe-b5d2d216fa23.json).

GEMM pipeline основан на NVIDIA CuTe DSL Blackwell tutorial GEMM; условия
лицензии: [NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
