# CuTe Agent Eval

End-to-end оценка coding agents, которые пишут CuTe DSL kernels для NVIDIA
Blackwell. Один запуск:

```text
task → isolated workspace → OpenCode agent → final candidate
     → B300 correctness/profile → baseline comparison → results table
```

Агент редактирует только `submission.py`. Inputs, Torch reference, validation,
timing и `PASS` принадлежат evaluator и добавляются перед отправкой на B300.

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
python -m experiment doctor \
  --model qwen-server/qwen3.6-35b-a3b
```

Все строки должны показывать `set`/`reachable`; сейчас доступно три task’а.

## 3. Запустить эксперимент

Одна задача, один независимый agent run, timeout 10 минут:

```bash
python -m experiment run \
  --model qwen-server/qwen3.6-35b-a3b \
  --task level1_01_square_matrix_multiplication_fp8 \
  --attempts 1 \
  --agent-timeout 600 \
  --gpu-timeout 600 \
  --seed 0
```

Все задачи:

```bash
python -m experiment run \
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

Во время запуска терминал сразу показывает текущий этап, текст агента, tool
calls и ответы evaluator. Тот же вывод сохраняется в `baseline-eval.log`,
`attempt-*/agent.log` и `attempt-*/candidate-eval.log`.
Если baseline не проходит, agent attempts не запускаются: без валидного
baseline невозможно посчитать корректный speedup.

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

API keys в artifacts не сохраняются.

## Готовые задачи

| Task ID | Операция | Precision |
| --- | --- | --- |
| `level1_01_square_matrix_multiplication_fp8` | `4096² × 4096²` GEMM | FP8 → FP32 |
| `level1_40_layer_norm_fp8` | LayerNorm `[16,64,256,256]` | FP8 → FP32 |
| `level2_76_gemm_add_relu_fp8` | GEMM + Bias + ReLU | FP8/FP32 epilogue |

Q8_0 относится к квантизации Qwen в `llama.cpp`; FP8 task’и отдельно
исполняются на B300.

## Проверить только evaluator

Список задач и local checks:

```bash
python -m cute_harness list
python -m cute_harness doctor --require-key
```

Запустить известный baseline без агента:

```bash
python -m cute_harness run \
  level1_01_square_matrix_multiplication_fp8 \
  --baseline \
  --timeout 600 \
  --seed 0
```

Локальные тесты не требуют GPU:

```bash
python -m unittest -v
```

## Структура

```text
experiment/      запускает OpenCode, собирает метрики и строит отчёт
cute_harness/    проверяет/собирает candidate и вызывает B300 API
tasks/           task contracts, prompts, starters и evaluators
cute_kernels/    проверенные baseline implementations
opencode/        headless OpenCode process runner
```

Evaluator отделён от agent workspace, но исполняется с candidate в одном
remote Python process. Это integrity guard для экспериментов, а не полноценная
security sandbox.
