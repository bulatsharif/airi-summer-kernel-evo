# Task format

Каждая директория задачи содержит:

- `task.json` — machine-readable контракт;
- `TASK.md` — prompt, который получает coding agent;
- `TASK_REFERENCE.md` — публичные task-specific API/design подсказки;
- `starter.py` — candidate-заготовку и локально принадлежащий harness evaluator,
  разделённые marker-строкой.

Проверенная реализация указывается в поле `baseline` манифеста. Она нужна для
smoke-тестов инфраструктуры и не должна попадать в контекст оцениваемого агента.

Минимальная единица агентского запуска:

```text
TASK.md + public references + candidate prefix -> agent submission.py
```

Агент не получает evaluator suffix. Перед remote run harness объединяет
прошедший policy candidate с evaluator в один standalone-файл, совместимый с
текущим API.

References объявляются в `task.json.references` и копируются в изолированный
workspace. Baseline, предыдущие runs и workspaces в этот список не входят.
