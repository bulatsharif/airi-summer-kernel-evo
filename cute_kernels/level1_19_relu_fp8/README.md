# CuTe FP8 ReLU — учебный пример

Исходная задача:
[KernelBench Level 1 / 19_ReLU.py](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/19_ReLU.py).

Форма сохранена без изменений: `[4096, 393216]`, или 1 610 612 736
элементов. В оригинале вход создаётся через `torch.rand` и всегда
неотрицателен, поэтому для содержательной проверки здесь используется
`uniform(-1, 1)`.

## Контракт

```text
input:  FP8 E4M3FN [4096, 393216]
output: FP8 E4M3FN [4096, 393216]

output[i] = max(float(input[i]), 0), затем округление обратно в FP8
```

ReLU коммутирует с положительным FP8 scale:

```text
ReLU(q * scale) = ReLU(q) * scale, scale > 0
```

Поэтому при одинаковом input/output scale самому ядру scale не нужен.
Для GEMM это уже не так: произведение accumulator нужно умножить на
`scale_a * scale_b` в epilogue.

## Структура файла

### 1. Host-side FP8 storage

PyTorch FP8 пока нельзя напрямую передать через DLPack в установленную CuTe
DSL. Вход и выход поэтому физически создаются как `torch.uint8`. Helper
`create_cute_tensor_for_fp8` конвертирует FP32 source в E4M3FN и назначает
CuTe tensor правильный element type.

Для output создаётся такой же byte storage, после чего его tensor view
помечается как `Float8E4M3FN`.

### 2. `@cute.jit` — специализация и launch

JIT-функция не является GPU kernel. Она во время компиляции:

1. Вычисляет `vector_size = 128 bits / 8 bits = 16` FP8-элементов.
2. Создаёт layout для 128 threads: `(4, 32)`.
3. Создаёт value layout одного thread: `(4, 16)`.
4. Композицией получает CTA tile `(16, 512)`.
5. Делит глобальные tensors на CTA tiles через `zipped_divide`.
6. Запускает `@cute.kernel` с 128 threads.

Один CTA обрабатывает:

```text
16 * 512 = 8192 FP8 elements
```

Grid содержит:

```text
(4096 / 16) * (393216 / 512)
= 256 * 768
= 196608 CTAs
```

Обе размерности делятся на tile полностью, поэтому в первом примере нет
predicate mask для хвостов.

### 3. `@cute.kernel` — runtime на GPU

Kernel выполняет следующий путь:

```text
GMEM FP8
  -> TiledCopy
  -> register fragment FP8
  -> TensorSSA FP32
  -> cute.where(x > 0, x, 0)
  -> register fragment FP8
  -> TiledCopy
  -> GMEM FP8
```

`TiledCopy.get_slice(thread_idx)` получает часть copy layout конкретного
thread. `partition_S` применяет её к CTA tile и создаёт thread-local view.
В коде больше нет ручного вычисления глобального индекса каждого элемента.

Арифметика выполняется в FP32:

```python
input_fp32 = input_fragment.load().to(cutlass.Float32)
zero = cute.full_like(input_fp32, 0.0)
output_fp32 = cute.where(input_fp32 > zero, input_fp32, zero)
```

После этого результат округляется в E4M3FN и сохраняется.

### 4. Проверка

Для конечных E4M3FN-значений ReLU можно проверить побитово:

- положительный FP8 byte остаётся без изменений;
- byte с установленным sign bit заменяется на `0`.

Проверяются все 1.61 млрд элементов, а не случайная выборка.

## Результат на B300

```text
shape=(4096, 393216)
mismatches=0
device_time_ms=108.899986
CuTe kernel time=5.924039 ms
profile_id=8a2462f3-603a-4dd0-a64b-be37eb48941b
```

Profiler trace:
[`profiles/8a2462f3-603a-4dd0-a64b-be37eb48941b.json`](../../profiles/8a2462f3-603a-4dd0-a64b-be37eb48941b.json).

`device_time_ms` относится ко всему `main()`: генерации FP32-входа,
конвертации в FP8, CuTe kernel и полной проверки. Само ядро пока также не
является окончательной bandwidth-оптимизацией: 196608 коротких CTA создают
существенный scheduling overhead. Это сознательно простая версия для
изучения CuTe layout и copy partitioning.

## Связь с тремя целевыми задачами

- Square GEMM заменит register `cute.where` на путь
  `GMEM -> TMA -> SMEM -> tcgen05 MMA -> TMEM -> epilogue`.
- GEMM + Add + ReLU повторно использует тот же ReLU-код уже внутри epilogue,
  не создавая отдельного kernel launch.
- LayerNorm повторно использует FP8 load/FP32 conversion, но добавляет
  warp/CTA reductions для mean и variance.
