# Task: square matrix multiplication, CuTe FP8

Реализуй candidate-код для:

```text
C = A @ B
A, B: [4096, 4096]
C:    [4096, 4096]
```

## Precision contract

- A и B физически хранятся в FP8 E4M3FN.
- MMA выполняется через CuTe DSL.
- Accumulation и output — FP32.
- Правый операнд хранится как `B_nk = B.T` формы `[N, K]`.

## Candidate ABI

- Редактируй только выданный `submission.py`.
- Сохрани как минимум один `@cute.kernel`, один `@cute.jit` и используй
  `cute.gemm`.
- Не определяй и не вызывай `main()`.
- Не создавай inputs, PyTorch reference или PASS-строку.
- Harness владеет `main()`, компиляцией, входами и numerical validation. Перед
  upload он автоматически добавит evaluator к candidate-коду.
- Candidate output должен быть записан CuTe-ядром.

## Acceptance

- Full max absolute error относительно PyTorch FP8 GEMM: `<= 0.125`.
- Sample max relative error относительно исходного FP32 matmul: `<= 0.01`.
- Скорость пока не оценивается.

## Iteration loop

```powershell
python3 -m cute_harness check level1_01_square_matrix_multiplication_fp8 submission.py
python3 -m cute_harness run level1_01_square_matrix_multiplication_fp8 submission.py
```
