# CuTe Agent Harness

Цель проекта — научиться системно обучать и оценивать coding agents, которые
пишут корректные и затем быстрые CuTe DSL kernels для FP8 и FP4 на NVIDIA
Blackwell.

Это не коллекция из трёх вручную написанных файлов. Проект задаёт повторяемый
цикл:

```text
task -> agent candidate -> local policy check -> owned evaluator assembly
     -> B300 run -> numerical validation -> result.json + profiler trace
```

Сейчас реализован correctness-first v0. Скорость собирается профилировщиком,
но пока не участвует в acceptance.

## Готовые задачи

| Task id | Операция | Precision | Baseline |
| --- | --- | --- | --- |
| `level1_01_square_matrix_multiplication_fp8` | `4096x4096 @ 4096x4096` | FP8 inputs, FP32 accumulate/output | PASS |
| `level1_40_layer_norm_fp8` | LayerNorm `[16,64,256,256]` | FP8 input, FP32 reduction/output | PASS |
| `level2_76_gemm_add_relu_fp8` | GEMM + BiasAdd + ReLU | FP8 GEMM, FP32 epilogue | PASS |

Task-пакеты находятся в [`tasks`](tasks/README.md). Проверенные реализации —
в [`cute_kernels`](cute_kernels).

## Быстрый старт

Требуется только Python 3.10+; клиент harness использует стандартную библиотеку.

```powershell
python -m cute_harness list
python -m cute_harness doctor
```

API key хранится только в окружении:

```powershell
$env:CUTE_HARNESS_API_KEY = '<key>'
```

Подготовить изолированную директорию для coding agent:

```powershell
python -m cute_harness prepare `
  level1_01_square_matrix_multiplication_fp8 `
  --output work/square
```

Агент получает `work/square/TASK.md`, `task.json` и candidate-файл
`submission.py`. `main()`, inputs, reference и PASS принадлежат harness и в
agent workspace не копируются. После изменений:

```powershell
python -m cute_harness check `
  level1_01_square_matrix_multiplication_fp8 `
  work/square/submission.py

python -m cute_harness run `
  level1_01_square_matrix_multiplication_fp8 `
  work/square/submission.py `
  --label qwen-moe-attempt-001
```

Проверить инфраструктуру известным baseline:

```powershell
python -m cute_harness run `
  level1_01_square_matrix_multiplication_fp8 `
  --baseline

python -m cute_harness run-all
```

URL можно переопределить через `CUTE_HARNESS_URL` или `--server`.

## Что сохраняется после run

Каждый запуск создаёт новую директорию, не перезаписывая старые артефакты:

```text
result.json       response сервера, acceptance, hashes, experiment label
candidate.py      точный код агента
submission.py     собранный harness файл, отправленный на B300
stdout.txt
stderr.txt
profile.json      скачанный PyTorch trace
```

API key в артефакты не записывается.

## Ограничение harness-only v1

Evaluator отделён от candidate workspace и автоматически добавляется локальным
harness перед upload. Policy запрещает candidate-коду `main()`, reference
compute и печать PASS. Это существенно честнее открытого v0, но ещё не
server-side security boundary: evaluator находится на машине оркестратора, а
remote endpoint по-прежнему исполняет один собранный Python-файл.

Подробно: [цель и pipeline](docs/PROJECT_AND_PIPELINE.md) и
[формат задач](docs/TASK_FORMAT.md).

## Проверка локальной части

```powershell
python -m unittest -v
```
