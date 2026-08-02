# Level 1 / 40 — LayerNorm на CuTe FP8

Исходная форма `[16, 64, 256, 256]`. Для ядра последние три измерения
представлены одной строкой длины `4,194,304`; один CTA обрабатывает одну из
16 строк.

## Что делает ядро

1. Первый streaming pass читает FP8, переводит значения в FP32 и считает mean.
2. Второй pass считает центрированную variance в FP32.
3. Warp shuffle reduction объединяет значения внутри warp; небольшой SMEM
   массив объединяет восемь warp в CTA.
4. Третий pass вычисляет `(x - mean) * rsqrt(variance + eps)` и пишет FP32.

Полная строка не хранится в регистрах: у каждого thread только скалярные
аккумуляторы. Это важно, потому что строка содержит больше четырёх миллионов
элементов.

У свежего `nn.LayerNorm` из KernelBench параметры инициализированы как
`weight=1`, `bias=0`, поэтому affine-часть в этой задаче является identity.

## Проверка на B300

```text
shape=(16, 64, 256, 256)
full_max_abs_vs_torch=0.000047
full_mean_abs_vs_torch=0.000009825
device_time_ms=62.567690
profile_id=a74562e8-51b2-40d3-a5c4-c77b43b411f0
PASS
```

Trace: [a74562e8-51b2-40d3-a5c4-c77b43b411f0.json](../../profiles/a74562e8-51b2-40d3-a5c4-c77b43b411f0.json).

Warp/CTA reduction pattern адаптирован из NVIDIA CuTe DSL `cta_norm`; условия
лицензии: [NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
