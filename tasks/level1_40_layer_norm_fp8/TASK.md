# Task: LayerNorm, CuTe FP8

Реализуй candidate-код для:

```text
input shape:      [16, 64, 256, 256]
normalized shape: [64, 256, 256]
epsilon:          1e-5
weight:           ones
bias:             zeros
```

Каждая batch row содержит `64 * 256 * 256 = 4,194,304` элементов.

## Precision contract

- Input storage — FP8 E4M3FN.
- Dequantization, mean, variance и normalization — FP32.
- Output — FP32.
- Variance считается как mean от `(x - mean)^2`.
- Используй streaming passes и warp/CTA reductions.

## Candidate ABI

- Редактируй только выданный `submission.py`.
- Сохрани как минимум один `@cute.kernel`, один `@cute.jit`, `cute.rsqrt` и
  butterfly warp shuffle.
- Не определяй и не вызывай `main()`.
- Не создавай inputs, PyTorch reference или PASS-строку.
- Harness владеет `main()`, компиляцией, входами и numerical validation.

## Acceptance

- Full max absolute error относительно PyTorch LayerNorm: `<= 0.01`.
- Все output values должны быть finite.
- Скорость пока не оценивается.

## Iteration loop

```powershell
python -m cute_harness check level1_40_layer_norm_fp8 submission.py
python -m cute_harness run level1_40_layer_norm_fp8 submission.py
```
