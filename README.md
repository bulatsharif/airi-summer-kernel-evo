# CuTe Agent Eval

End-to-end оценка coding agents, которые пишут CuTe DSL kernels для NVIDIA
Blackwell. Один запуск:

```text
task → isolated workspace → OpenCode agent → final candidate
     → B300 correctness/profile → baseline comparison → results table
```

Агент редактирует только `submission.py`. Inputs, Torch reference, validation,
timing и `PASS` принадлежат evaluator и добавляются перед отправкой на B300.

## Reference: B300 HTTP verifier

Серверная часть evaluator опубликована отдельно в
[`http_verifier/`](http_verifier/). Это самостоятельный FastAPI-проект с
аутентификацией, очередью на один GPU, запуском submission в дочернем процессе,
PyTorch/Nsight profiling, тестами и Docker deployment. Секреты и generated
profiles в репозиторий не включены.

## Что нужно

- Python 3.10+;
- установленный и настроенный `opencode`;
- Bash, `jq`, `tee`, `nohup`, `tail`, `pgrep`;
- GNU `timeout` (`brew install coreutils` на macOS);
- SSH-доступ к Qwen server;
- API keys для Qwen и B300 harness.

Все команды ниже выполняются из корня репозитория.

## 1. Поднять model endpoint

В отдельном терминале:

```bash
ssh -N -L 18001:127.0.0.1:8001 User17@176.109.107.137
```

Туннель должен оставаться запущенным во время эксперимента.

## 2. Настроить окружение

Ключи не нужно записывать в репозиторий:

```bash
export QWEN_BASE_URL=http://127.0.0.1:18001/v1

read -r -s QWEN_API_KEY
echo
export QWEN_API_KEY

read -r -s CUTE_HARNESS_API_KEY
echo
export CUTE_HARNESS_API_KEY
```

Проверить OpenCode, задачи, переменные и model endpoint:

```bash
python3 -m experiment doctor \
  --model qwen-server/qwen3.6-35b-a3b
```

Ожидаемый результат: три ключа имеют статус `set`, `opencode` найден,
`model_endpoint=reachable`, `tasks=9`, а requested model не помечена как
`not advertised`.

## 3. Запустить эксперимент

Одна задача, один независимый agent run, timeout 10 минут:

```bash
python3 -m experiment run \
  --model qwen-server/qwen3.6-35b-a3b \
  --task level1_01_square_matrix_multiplication_fp8 \
  --attempts 1 \
  --agent-timeout 600 \
  --gpu-timeout 600 \
  --seed 0
```

Все задачи:

```bash
python3 -m experiment run \
  --model qwen-server/qwen3.6-35b-a3b \
  --all \
  --attempts 1 \
  --agent-timeout 600 \
  --gpu-timeout 600 \
  --seed 0
```

Несколько `--task` можно передать повторно. `--attempts N` создаёт `N`
независимых OpenCode sessions на каждый task. Внутри session агент может делать
несколько development evals; после его завершения orchestrator всегда запускает
отдельный authoritative eval финального файла.

Основные параметры:

- `--attempts` — число независимых agent sessions, а не число замеров kernel;
- `--agent-timeout` — лимит всей OpenCode session;
- `--gpu-timeout` — лимит одного обращения к B300 harness;
- `--seed` — фиксирует входные данные evaluator; baseline и candidate получают
  один seed;
- `--warmup` — число запусков kernel перед измерением, по умолчанию `2`;
- `--repeats` — число измеряемых запусков, по умолчанию `5`; в таблицу попадает
  median.

Для быстрого smoke test инфраструктуры можно добавить `--warmup 1 --repeats 1`.
Такой единичный замер пригоден для проверки pipeline, но слишком шумный для
сравнения производительности.

Во время запуска терминал сразу показывает текущий этап, текст агента, tool
calls и ответы evaluator. Тот же вывод сохраняется в `baseline-eval.log`,
`attempt-*/agent.log` и `attempt-*/candidate-eval.log`. Если дочерний процесс
жив, но 30 секунд ничего не печатает, orchestrator выводит heartbeat.
Если baseline не проходит, agent attempts не запускаются: без валидного
baseline невозможно посчитать корректный speedup.

Каждый workspace получает существующий OpenCode skill `cute-fp8-kernels` с
подробным CuTe handbook, compile-verified templates и короткие task-specific
references. Сначала агент читает только выбранный task reference и один
релевантный template; более общий API handbook и
`candidate-error-atlas.md` используются после конкретной диагностики. Поэтому
простая задача не платит input tokens за весь CUTLASS-контекст сразу и не
смешивает несовместимые API-рецепты.
Subagents разрешены и наследуют тот же skill, references и файловую изоляцию;
предыдущие `runs/**` и `work/**` им недоступны. Один model response ограничен
8192 токенами, а provider request — 180 секундами.

Разбор типичных ошибок Qwen, устройство retrieval и результаты эксперимента:
[docs/QWEN35B_PASS_RATE.md](docs/QWEN35B_PASS_RATE.md).

## Результат

В конце печатается таблица:

```text
Model | Task | Status | Baseline ms | Agent ms | Speedup
      | Input | Cache input | Output | Agent s
```

`Speedup = baseline kernel time / agent kernel time` и выводится только для
корректного candidate. Baseline и candidate используют одинаковые seed,
evaluator, warmup и repeats. Время kernel — median CUDA-event time; общий
`device_time_ms` профилировщика сохраняется отдельно.

Artifacts находятся в `runs/experiments/<run-id>/`:

```text
manifest.json                    параметры эксперимента и Git commit
results.json / .csv / .txt       итоговая таблица
<task>/baseline/                 baseline source, result и profile
<task>/attempt-001/
  agent-events.jsonl             OpenCode event stream
  agent-metrics.json             model, tokens, sessions, wall time
  candidate.py                   финальное решение агента
  candidate-eval/                assembled submission, result и profile
```

Agent workspace дополнительно содержит `references/` и
`.opencode/skills/cute-fp8-kernels/`; точные локальные пути записаны в
публичных `task.json.references` и `task.json.agent_skills`.

API keys в artifacts не сохраняются.

## Готовые задачи

| Task ID | Операция | Precision |
| --- | --- | --- |
| `level1_01_square_matrix_multiplication_fp8` | `4096² × 4096²` GEMM | FP8 → FP32 |
| `level1_40_layer_norm_fp8` | LayerNorm `[16,64,256,256]` | FP8 → FP32 |
| `level1_72_conv_transpose3d_fp8` | Grouped asymmetric ConvTranspose3d | FP8 → FP32 |
| `level2_09_matmul_subtract_multiply_relu_fp8` | GEMM + Subtract + Multiply + ReLU | FP8/FP32 epilogue |
| `level2_12_gemm_multiply_leaky_relu_fp8` | GEMM + Multiply + LeakyReLU | FP8/FP32 epilogue |
| `level2_14_gemm_divide_sum_scaling_fp8` | GEMM + Divide + row Sum + Scaling | FP8/FP32 reduction |
| `level2_40_matmul_scaling_residual_add_fp8` | GEMM + Scaling + ResidualAdd | FP8/FP32 epilogue |
| `level2_63_gemm_relu_divide_fp8` | GEMM + ReLU + Divide | FP8/FP32 epilogue |
| `level2_76_gemm_add_relu_fp8` | GEMM + Bias + ReLU | FP8/FP32 epilogue |

Q8_0 относится к квантизации Qwen в `llama.cpp`; FP8 task’и отдельно
исполняются на B300.

## Проверить только evaluator

Список задач и local checks:

```bash
python3 -m cute_harness list
python3 -m cute_harness doctor --require-key
```

Запустить известный baseline без агента:

```bash
python3 -m cute_harness run \
  level1_01_square_matrix_multiplication_fp8 \
  --baseline \
  --timeout 600 \
  --seed 0
```

Локальные тесты не требуют GPU:

```bash
python3 -m unittest -v
```

## Сравнить несколько submission с baseline

Команда `compare` принимает одну или несколько пар `TASK_ID=PATH`. Для каждой
уникальной задачи baseline измеряется отдельным запуском, после чего каждый
candidate проверяется тем же evaluator с теми же `seed`, `warmup` и `repeats`.
Ускорение считается только для корректных результатов:

```text
speedup = baseline kernel_time_ms / candidate kernel_time_ms
```

Пример для нескольких задач:

```bash
python3 -m cute_harness compare \
  level1_01_square_matrix_multiplication_fp8=work/gemm/submission.py \
  level1_40_layer_norm_fp8=work/layernorm/submission.py \
  --seed 0 \
  --warmup 2 \
  --repeats 5 \
  --timeout 600 \
  --output runs/comparisons/my-run
```

Можно передать несколько реализаций одной задачи. В этом случае baseline
запускается один раз и переиспользуется для всех указанных файлов:

```bash
python3 -m cute_harness compare \
  level2_14_gemm_divide_sum_scaling_fp8=variants/v1/submission.py \
  level2_14_gemm_divide_sum_scaling_fp8=variants/v2/submission.py \
  level2_14_gemm_divide_sum_scaling_fp8=variants/v3/submission.py \
  --seed 0 \
  --warmup 2 \
  --repeats 5
```

По умолчанию артефакты создаются в `runs/<timestamp>_comparison/`:

```text
baselines/<task-id>/                 отдельный baseline run
candidates/<number>_<task-id>/      candidate run
comparison.json                      параметры и полные строки результата
comparison.csv                       машиночитаемая таблица
comparison.txt                       терминальная таблица
```

В таблице `Candidate ms` и `Speedup` выводятся только для прошедшего
correctness candidate. Если baseline задачи не прошёл, связанные candidates
получают статус `SKIPPED`. Команда возвращает код `0`, только когда все
baselines и candidates прошли; validation и measurement contract совпадают с
обычным `cute_harness run`.

## Структура

```text
experiment/      запускает OpenCode, собирает метрики и строит отчёт
cute_harness/    проверяет/собирает candidate и вызывает B300 API
tasks/           task contracts, prompts, starters и evaluators
cute_kernels/    проверенные baseline implementations
opencode/        headless OpenCode runner и подробный CuTe skill/handbook
```

Evaluator отделён от agent workspace, но исполняется с candidate в одном
remote Python process. Это integrity guard для экспериментов, а не полноценная
security sandbox.
