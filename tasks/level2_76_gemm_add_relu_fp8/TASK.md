# Task: GEMM + BiasAdd + ReLU, CuTe FP8

Реализуй candidate-код для:

```text
X:    [1024, 8192]
W:    [8192, 8192]
bias: [8192]
Y = relu(X @ W.T + bias)
```

## Precision contract

- X и W хранятся в FP8 E4M3FN.
- GEMM accumulation — FP32.
- Bias и output — FP32.
- FP8 scale возвращается до BiasAdd/ReLU.

Корректный первый вариант может использовать два CuTe launch: FP8 GEMM, затем
BiasAdd + ReLU.

## Candidate ABI

- Редактируй только выданный `submission.py`.
- Сохрани минимум два `@cute.kernel`, один `@cute.jit` и `cute.gemm`.
- Не определяй и не вызывай `main()`.
- Не создавай inputs, PyTorch reference или PASS-строку.
- Harness владеет `main()`, компиляцией, входами и numerical validation.
- Candidate output должен полностью вычисляться CuTe-ядрами.

## Acceptance

- Full max absolute error относительно PyTorch FP8 pipeline: `<= 0.01`.
- Sample max absolute error относительно FP32 pipeline: `<= 0.1`.
- Скорость пока не оценивается.

## Iteration loop

```powershell
python -m cute_harness check level2_76_gemm_add_relu_fp8 submission.py
python -m cute_harness run level2_76_gemm_add_relu_fp8 submission.py
```
