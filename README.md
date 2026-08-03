# AIRI Summer Kernel Evo

Исследовательский проект по автоматической генерации и оптимизации GPU-ядер
на CuTe DSL для NVIDIA Blackwell с помощью coding agents.

```text
задача → изолированная среда агента → CuTe-кандидат
        → проверка корректности на B300 → профилирование → следующий кандидат
```

> **Важно:** это публичная часть проекта. Часть актуального кода KernelEvo,
> внутренняя инфраструктура, серверные конфигурации и некоторые эксперименты
> находятся в закрытом репозитории. Для полного воспроизведения production
> pipeline нужен доступ к этим компонентам и B300-серверам.

## Что находится в репозитории

| Путь | Содержимое |
| --- | --- |
| `cute_harness/` | Сборка submission, проверка политики и клиент B300 evaluator |
| `experiment/` | Запуск агентов, authoritative evaluation и сводные отчёты |
| `tasks/`, `probes/` | FP8-задачи, Torch references и известные baseline-решения |
| `opencode/` | Конфигурация и инструкции для coding agent |
| `kernel_evo_archive/` | Снимок KernelEvo, патчи, полные запуски и лучшие ядра |
| `docs/` | Описание pipeline, формата задач и анализ экспериментов |
| `report.pdf`, `presentation.pdf` | Итоговый отчёт и презентация проекта |

Архив экспериментов подробно описан в
[kernel_evo_archive/README.md](kernel_evo_archive/README.md). В нём сохранены
шесть запусков GPT-2/Qwen, лучшие кандидаты и SHA-256 checksums.

## Ключевые результаты

При повторном парном измерении под эксклюзивной блокировкой GPU:

| Задача | Лучший режим | Ускорение относительно baseline |
| --- | --- | ---: |
| GPT-2 transformer block | timeline profiler | **11.45×** |
| Qwen attention block | без profiler | **79.59×** |

### Откуда взялось ускорение

- **Qwen: 49.88 → 0.627 мс.** Агент заменил скалярные GEMM, где каждый поток
  последовательно считал один dot product, на FP8 Tensor Core `tcgen05`. Данные
  подаются через TMA и трёхступенчатый pipeline, а residual и requantization
  выполняются прямо в epilogue. Score, causal softmax и gated context также
  объединены в одно attention-ядро.
- **GPT-2: 3.59 → 0.314 мс.** GEMM разбиты на tiles `16×32`: общие фрагменты
  загружаются в shared memory, а каждый поток накапливает четыре результата в
  регистрах. Bias, GELU, residual и FP8 conversion встроены в epilogue;
  score + softmax + context слиты из трёх запусков в один. Timeline показал,
  что время уходит в вычисления GEMM/attention, а не в паузы между ядрами.

Это ускорение относительно простого корректного **скалярного CuTe baseline**,
а не относительно оптимизированных cuBLAS, FlashAttention или production
inference engine. У Qwen скачок `3.21 → 0.63` мс произошёл одним удачным ходом
без профилировщика; у GPT-2 timeline направлял последовательные улучшения.

Методика, траектории по ходам и ограничения приведены в
[отчёте](report.pdf) и [архиве результатов](kernel_evo_archive/README.md).

## Быстрый старт

Локальные тесты не требуют GPU и API-ключей:

```bash
python3 -m unittest -v
```

Для полного эксперимента нужны Python 3.10+, `opencode`, OpenAI-compatible
model endpoint и доступ к B300 harness. Ключи передаются только через
переменные окружения:

```bash
export QWEN_BASE_URL=http://127.0.0.1:18001/v1
read -r -s QWEN_API_KEY && export QWEN_API_KEY
read -r -s CUTE_HARNESS_API_KEY && export CUTE_HARNESS_API_KEY

python3 -m experiment doctor --model qwen-server/qwen3.6-35b-a3b
```

Запуск одной задачи:

```bash
python3 -m experiment run \
  --model qwen-server/qwen3.6-35b-a3b \
  --task model_gpt2_small_transformer_block_fp8 \
  --attempts 1 \
  --agent-timeout 600 \
  --gpu-timeout 600 \
  --seed 0
```

Результаты сохраняются в `runs/experiments/<run-id>/`. API-ключи в artifacts
не записываются.

## Документация

- [Архитектура проекта и pipeline](docs/PROJECT_AND_PIPELINE.md)
- [Анализ Qwen 35B](docs/QWEN35B_PASS_RATE.md)
- [Формат новой задачи](docs/TASK_FORMAT.md)
- [API-only эксперимент](docs/API_ONLY_EXPERIMENT.md)
- [Полный архив KernelEvo](kernel_evo_archive/README.md)
