# Формат CuTe task

## Директория

```text
tasks/<task_id>/
  task.json
  TASK.md
  TASK_REFERENCE.md
  starter.py
```

Known-good baseline хранится отдельно и не копируется в agent workspace.
Общий подробный CuTe handbook переиспользуется из OpenCode skill
`opencode/.opencode/skills/cute-fp8-kernels`.

## `task.json`

Manifest фиксирует:

- стабильный task id и источник;
- precision/storage/accumulator/output contract;
- shapes, seed и параметры операции;
- numerical tolerances и success pattern;
- минимальные candidate CuTe primitives;
- список OpenCode skills;
- список публичных reference-файлов;
- private baseline path.

Команда `prepare` удаляет baseline path из публичной копии и устанавливает
skills в `.opencode/skills/`, а reference-файлы — в `references/`, переписывая
manifest на локальные пути.

## `TASK.md`

Prompt описывает математическую операцию, shapes/layout, precision contract,
candidate ABI, acceptance thresholds и команды `check`/`run`.

Candidate ABI v1:

- агент сначала загружает skills из `task.json.agent_skills`, затем читает
  каждый путь из `task.json.references`;
- агент редактирует только `submission.py`;
- candidate содержит constants, kernels и JIT entrypoint;
- candidate не определяет и не вызывает `main()`;
- inputs, compile call, reference, assertions и PASS принадлежат harness.

## `starter.py`

Файл состоит из двух частей:

```python
# imports, constants, @cute.kernel, @cute.jit candidate starter

# === CUTE_HARNESS_EVALUATOR_V1 ===
# private main(), inputs, compile, reference, assertions, PASS
```

`prepare` выдаёт только часть до marker. `run` добавляет часть после marker к
прошедшему policy candidate и отправляет assembled standalone-файл на B300.

## Result contract

Серверный response принимается только при:

```text
success=true
exit_code=0
timed_out=false
stdout matches validation.success_pattern
```

Artifacts включают отдельно `candidate.py` и отправленный `submission.py`.

## Добавление задачи

1. Создать `task.json`, `TASK.md`, `starter.py`, указать agent skills и нужные
   public references.
2. Добавить evaluator marker ровно один раз.
3. Реализовать private evaluator suffix и known-good baseline.
4. Настроить candidate policy.
5. Выполнить:

```powershell
python -m pytest -q
python3 -m cute_harness doctor
python3 -m cute_harness run <task_id> --baseline
```
