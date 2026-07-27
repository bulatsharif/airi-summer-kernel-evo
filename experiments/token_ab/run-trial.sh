#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-trial.sh <web|local> <task-id> <trial-id> [timeout]

Example:
  ./experiments/token_ab/run-trial.sh \
    web level1_01_square_matrix_multiplication_fp8 001 30m
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

ARM=$1
TASK_ID=$2
TRIAL_ID=$3
TASK_TIMEOUT=${4:-30m}

[[ "$ARM" == "web" || "$ARM" == "local" ]] || {
  printf 'arm must be web or local\n' >&2
  exit 2
}
[[ "$TRIAL_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'trial-id must contain only letters, digits, dot, underscore, or dash\n' >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
CASES_FILE="$SCRIPT_DIR/cases.json"
PYTHON_BIN=${PYTHON_BIN:-python}

command -v jq >/dev/null 2>&1 || {
  printf 'jq is required\n' >&2
  exit 2
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  printf '%s is not available on PATH\n' "$PYTHON_BIN" >&2
  exit 2
}

jq -e --arg task "$TASK_ID" \
  '.tasks | any(.id == $task)' "$CASES_FILE" >/dev/null || {
  printf 'task is not in experiments/token_ab/cases.json: %s\n' "$TASK_ID" >&2
  exit 2
}

WORK_REL="work/token-ab-${TASK_ID}-${ARM}-${TRIAL_ID}"
WORK_DIR="$REPO_ROOT/$WORK_REL"
RUN_DIR="$REPO_ROOT/ab_runs/$TASK_ID/$ARM/$TRIAL_ID"
CANDIDATE_REL="$WORK_REL/submission.py"
CONFIG_FILE="$SCRIPT_DIR/configs/$ARM.json"
COMMON_INSTRUCTIONS="$SCRIPT_DIR/instructions/common.md"
ARM_INSTRUCTIONS="$SCRIPT_DIR/instructions/$ARM.md"
SKILL_DIR="$REPO_ROOT/opencode/.opencode/skills"
RUNNER="$REPO_ROOT/opencode/opencode-headless.sh"

[[ ! -e "$WORK_DIR" ]] || {
  printf 'refusing to reuse workspace: %s\n' "$WORK_DIR" >&2
  exit 2
}
[[ ! -e "$RUN_DIR" ]] || {
  printf 'refusing to reuse run directory: %s\n' "$RUN_DIR" >&2
  exit 2
}
[[ -x "$RUNNER" ]] || {
  printf 'headless runner is missing or not executable: %s\n' "$RUNNER" >&2
  exit 2
}

CONFIG_CONTENT=$(
  jq -cs \
    --arg arm "$ARM" \
    --arg common "$COMMON_INSTRUCTIONS" \
    --arg specific "$ARM_INSTRUCTIONS" \
    --arg skill_dir "$SKILL_DIR" \
    --arg workspace_pattern "$WORK_REL/**" \
    --arg candidate "$CANDIDATE_REL" \
    -f "$SCRIPT_DIR/config-overlay.jq" \
    "$REPO_ROOT/opencode.json" "$CONFIG_FILE"
)

(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" -m cute_harness prepare "$TASK_ID" --output "$WORK_DIR"
)

PROMPT=$(cat <<EOF
Solve the prepared task in $WORK_REL.

Read $WORK_REL/TASK.md, $WORK_REL/task.json, and
$WORK_REL/submission.py. Edit only $WORK_REL/submission.py.

After editing, run exactly this local compatibility check:

python -m cute_harness check $TASK_ID $CANDIDATE_REL

Do not call the remote harness. Finish after the local check.
EOF
)

mkdir -p "$(dirname "$RUN_DIR")"

env -u CUTE_HARNESS_API_KEY \
  OPENCODE_CONFIG_CONTENT="$CONFIG_CONTENT" \
  OPENCODE_DISABLE_PROJECT_CONFIG=1 \
  OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
  "$RUNNER" \
    --dir "$REPO_ROOT" \
    --timeout "$TASK_TIMEOUT" \
    --run-dir "$RUN_DIR" \
    -- "$PROMPT"

jq -n \
  --arg task_id "$TASK_ID" \
  --arg arm "$ARM" \
  --arg trial_id "$TRIAL_ID" \
  --arg candidate_path "$CANDIDATE_REL" \
  --arg workspace_path "$WORK_REL" \
  --arg timeout "$TASK_TIMEOUT" \
  --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: 1,
    task_id: $task_id,
    arm: $arm,
    trial_id: $trial_id,
    candidate_path: $candidate_path,
    workspace_path: $workspace_path,
    timeout: $timeout,
    started_at: $started_at,
    remote_attempt_budget: 1
  }' >"$RUN_DIR/metadata.json"

printf 'Trial metadata: %s\n' "$RUN_DIR/metadata.json"
printf 'Evaluate once after completion:\n  %q %q\n' \
  "$SCRIPT_DIR/evaluate-trial.sh" "$RUN_DIR"
