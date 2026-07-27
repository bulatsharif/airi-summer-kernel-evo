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
TASK.md + OpenCode skill + public references + candidate prefix
  -> agent submission.py
```

Агент не получает evaluator suffix. Перед remote run harness объединяет
прошедший policy candidate с evaluator в один standalone-файл, совместимый с
текущим API.

Skills и references объявляются в `task.json.agent_skills` и
`task.json.references`, затем копируются в изолированный workspace. Подробный
общий handbook загружается через skill прогрессивно, а `TASK_REFERENCE.md`
всегда задаёт специфику задачи. Baseline, предыдущие runs и workspaces туда не
входят.
