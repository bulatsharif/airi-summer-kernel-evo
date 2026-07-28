# Qwen 35B: повышение pass-rate на CuTe FP8

## Итог

На первом полном прогоне Qwen прошла 3 из 9 задач. После добавления
compile-verified templates, task-specific retrieval и карты ошибок полный
следующий прогон прошёл 6 из 9. Точечные повторные прогоны оставшихся трёх задач
на обновлённом контексте также завершились `PASS`, поэтому последовательный
best-of результат стал 9 из 9. После этого новый полный прогон всех девяти задач
на зафиксированном финальном контексте также завершился 9 из 9 `PASS`.

Это подтверждает 100% на одном frozen-config прогоне, но ещё не оценивает
устойчивый pass-rate модели: для него нужны несколько seeds или независимых
agent sessions на каждой задаче.

## Почему не полный исходный репозиторий CUTLASS 4.5.1

Evaluator и подготовленный handbook используют CUTLASS CuTe DSL 4.6.1.
Репозиторий 4.5.1 содержит другую версию API и способен закрепить именно те
ошибки, которые уже наблюдались: смешение старых и новых TMA, pipeline и TMEM
интерфейсов. Полный source dump также увеличивает input/cache tokens и затрудняет
retrieval.

Для eval действует изоляция от сети и внешней документации, поэтому внешний
репозиторий не клонируется во время agent run. Вместо него агент получает
маленький version-pinned пакет:

- compile-verified dense GEMM scaffold для CuTe DSL 4.6.1, выделенный из
  ранее прошедшего agent candidate, а не из скрытого baseline;
- compile-verified двумерный elementwise/epilogue template;
- точный task reference с shapes, scales, indexing и выбранным pattern;
- error atlas, который читается только после конкретной ошибки;
- общий API handbook как диагностический fallback, а не стартовый контекст.

Таким образом, полезен не максимальный объём исходников, а короткий набор
совместимых примеров с известным результатом компиляции.

Это меняет предмет измерения: результат 9/9 относится к агенту вместе с
retrieval/template harness, а не к чистой способности базовой модели написать
Blackwell TMA/TMEM pipeline с нуля.

## Основные типичные ошибки

| Семейство | Наблюдавшаяся ошибка | Изменение контекста |
| --- | --- | --- |
| Dense GEMM | `cute.constexpr`, выдуманный `SharedStorage`, старый `TmaOperandMajorMode` | Готовый 4.6.1 template вместо восстановления API по памяти |
| TMA | Неверный namespace helper-а, распаковка TMA object как tuple, лишняя координата | Зафиксированы `.atom`, `.tma_tensor` и точное TMA indexing |
| Pipeline/TMEM | Выдуманный `producer_get_barrier`, неверный владелец `partition_D` | Сохранение целого compile-verified pipeline/TMEM core |
| Launch | Вызов `@cute.kernel` как обычной функции; host construction внутри kernel | Явная структура `@cute.jit` entrypoint → `.launch()` |
| Epilogue | `cutlass.relu`, `cute.fmax`, `cute.maximum`, `break`/`while` | Отдельный elementwise template с поддерживаемым predication |
| LayerNorm | Каждый lane суммирует одинаковые элементы; повторное суммирование partials | Точное lane ownership: `column = iteration * THREADS + lane` |
| ConvTranspose3d | 3D indexing у 2D tensor, только один input channel, неверный stride веса | Явные flattened row/column formulas и все четыре канала группы |
| Retry loop | Повтор идентичного candidate после timeout/launch failure | Запрет идентичного retry; восстановление template при повторе ошибки |
| Retrieval | Task reference искался внутри skill directory | Раздельное точное разрешение workspace references и skill links |

Полная диагностическая таблица находится в
`opencode/.opencode/skills/cute-fp8-kernels/references/candidate-error-atlas.md`.

## Результаты

Условия: Qwen3.6-35B-A3B, один agent attempt, seed 0, agent timeout 600 s,
GPU timeout 600 s, два warmup и пять измерений. `Agent ms` — медиана пяти
CUDA-event измерений candidate kernel, а не длительность работы агента.

### Контрольный frozen-config прогон

Артефакт: `runs/experiments/qwen35b-curated-v5-frozen-all9/`.

Все девять agent sessions стартовали с одной и той же версией task references,
templates, error atlas и system prompt. Между задачами контекст не менялся.

| Task | Status | Baseline ms | Agent ms | Speedup | Input | Cache input | Output | Agent s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square GEMM | PASS | 0.143872 | 0.141888 | 1.014x | 23,572 | 274,374 | 5,178 | 162.47 |
| LayerNorm | PASS | 8.343104 | 36.610176 | 0.228x | 18,162 | 128,542 | 2,260 | 90.65 |
| ConvTranspose3d | PASS | — | — | — | 25,982 | 313,984 | 5,468 | 163.65 |
| L2-09 | PASS | 0.384864 | 0.152128 | 2.530x | 22,832 | 190,529 | 4,879 | 147.51 |
| L2-12 | PASS | 0.387008 | 0.153056 | 2.529x | 23,884 | 112,455 | 4,571 | 130.40 |
| L2-14 | PASS | 0.363232 | 0.132128 | 2.749x | 25,058 | 391,956 | 7,958 | 272.05 |
| L2-40 | PASS | 0.385888 | 0.153792 | 2.509x | 23,401 | 121,455 | 4,536 | 134.45 |
| L2-63 | PASS | 0.384864 | 0.150816 | 2.552x | 22,517 | 142,166 | 4,512 | 132.30 |
| L2-76 | PASS | 0.124160 | 0.146816 | 0.846x | 28,591 | 161,684 | 4,728 | 150.45 |

Итого: 9/9 `PASS`; 6/8 задач с доступным сопоставимым `kernel_time_ms` быстрее
baseline. Суммарно: 213,999 uncached input, 1,837,145 cached input, 44,090
output tokens и 1,383.92 s agent wall time. Если считать cache-read как часть
логического контекста, это 2,095,234 токена.

ConvTranspose3d прошла correctness, но её evaluator не возвращает отдельный
`kernel_time_ms`, поэтому speedup отсутствует. Candidate profiler
`device_time_ms` равен 16.479015 ms.

Точные authoritative validation/profile данные:

| Task | Dev evals | Validation | Device ms | Profile ID |
| --- | ---: | --- | ---: | --- |
| Square GEMM | 1 | full abs 0.000122; sample rel 0.001324 | 1.973638 | `3cb56d30-153d-4f57-9930-2e60231b1d4e` |
| LayerNorm | 1 | full abs 0.000178; mean abs 0.000044916 | 280.544030 | `22398b35-02c6-4df0-b6cc-840f83b108a3` |
| ConvTranspose3d | 3 | full abs 0.000000; mean abs 0.000000026 | 16.479015 | `9934d314-b885-427d-80a4-20fa55f5efbe` |
| L2-09 | 1 | full abs 0.000000; sample abs 0.000000 | 2.495617 | `800e1c3d-2334-4024-9c86-49e0c7ea3cba` |
| L2-12 | 1 | full abs 0.000000; sample abs 0.072909 | 2.479531 | `8d6ffe5d-f203-4a48-80f6-c0ee6de418c7` |
| L2-14 | 4 | full abs 0.000022; sample abs 1.045462 | 2.390248 | `6bf913dc-07f5-4f4e-a316-86e38068f9aa` |
| L2-40 | 1 | full abs 0.000000; sample abs 0.054682 | 2.462725 | `672f54a4-d634-4e93-979d-efd1d4229542` |
| L2-63 | 1 | full abs 0.000000; sample abs 0.018227 | 2.466669 | `417f9100-8d3f-44ab-b660-c67c348294a9` |
| L2-76 | 1 | full abs 0.000000; sample abs 0.036455 | 2.592938 | `f9c4795c-d8fd-43c6-b9e3-4fbb8aac1722` |

На каждую задачу запускалась одна OpenCode session. `Dev evals` — число
вызовов remote evaluator внутри этой session до первого `PASS`; после agent
session orchestrator выполнил ещё один отдельный authoritative eval. Финальные
submissions находятся по шаблону
`runs/experiments/qwen35b-curated-v5-frozen-all9/<task>/attempt-001/candidate.py`.

### Полный прогон после первой итерации retrieval

Артефакт: `runs/experiments/qwen35b-retrieval-v3-all9/`.

| Task | Status | Baseline ms | Agent ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Square GEMM | PASS | 0.147392 | 0.144736 | 1.018x |
| LayerNorm | PASS | 18.311487 | 90.899261 | 0.201x |
| ConvTranspose3d | TIMEOUT | — | — | — |
| L2-09 | PASS | 0.389536 | 0.148768 | 2.618x |
| L2-12 | FAIL | 0.383680 | — | — |
| L2-14 | PASS | 0.365696 | 0.129792 | 2.818x |
| L2-40 | PASS | 0.385856 | 0.148256 | 2.603x |
| L2-63 | PASS | 0.384160 | 0.149184 | 2.575x |
| L2-76 | TIMEOUT | 0.124160 | — | — |

Итого: 6/9 `PASS`.

### Последовательный best-of после точечных исправлений

Для уже прошедших задач взят результат полного прогона выше. L2-12, L2-76 и
ConvTranspose3d были перезапущены после добавления их точных patterns.

| Task | Status | Baseline ms | Agent ms | Speedup | Input | Cache input | Output | Agent s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square GEMM | PASS | 0.147392 | 0.144736 | 1.018x | 27,450 | 227,935 | 7,493 | 189.19 |
| LayerNorm | PASS | 18.311487 | 90.899261 | 0.201x | 25,837 | 286,988 | 2,987 | 132.21 |
| ConvTranspose3d | PASS | — | — | — | 22,204 | 270,160 | 7,937 | 203.73 |
| L2-09 | PASS | 0.389536 | 0.148768 | 2.618x | 24,893 | 138,416 | 4,532 | 143.88 |
| L2-12 | PASS | 0.386112 | 0.148256 | 2.604x | 22,743 | 88,976 | 4,473 | 140.56 |
| L2-14 | PASS | 0.365696 | 0.129792 | 2.818x | 22,330 | 116,027 | 4,742 | 142.64 |
| L2-40 | PASS | 0.385856 | 0.148256 | 2.603x | 30,869 | 243,491 | 8,967 | 228.35 |
| L2-63 | PASS | 0.384160 | 0.149184 | 2.575x | 29,112 | 242,350 | 8,518 | 227.19 |
| L2-76 | PASS | 0.125600 | 0.152384 | 0.824x | 23,169 | 105,531 | 4,288 | 144.36 |

ConvTranspose3d прошла correctness и profiler; evaluator этой задачи не
возвращает отдельный `kernel_time_ms`, поэтому в общей таблице нет сопоставимого
speedup. Её profiler `device_time_ms` равен 0.464992 ms.

Сумма выбранных успешных sessions: 228,607 uncached input, 1,719,874 cached
input, 53,937 output tokens и 1,552.11 s agent wall time. Это сумма best-of
sessions, а не стоимость одного зафиксированного полного прогона.

## Как измерять эффект дальше

Следующая оценка устойчивости должна сравнить три фиксированные конфигурации с
одинаковыми task order, seed, timeout и числом attempts:

1. task contract без compile-verified templates;
2. полный version-matched source/context pack;
3. текущий curated retrieval: task reference + template + diagnostic fallback.

Для итогового вывода нужны как минимум несколько независимых sessions на
задачу. Сравнивать следует:

- pass-rate и долю задач, прошедших с первого remote attempt;
- число remote iterations до первого `PASS`;
- uncached/cache/output tokens;
- agent wall time и timeout-rate;
- correctness отдельно от `speedup > 1`;
- медиану и разброс kernel time на одинаковых seeds.

Главная текущая гипотеза: pass-rate растёт прежде всего от точного
version-matched executable pattern и диагностической маршрутизации, а не от
добавления максимально большого объёма документации.
