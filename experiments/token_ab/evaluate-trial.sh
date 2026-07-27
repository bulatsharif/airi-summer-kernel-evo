#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: evaluate-trial.sh RUN_DIR\n' >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR=$1

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO_ROOT/$RUN_DIR"
fi
RUN_DIR=$(cd "$RUN_DIR" 2>/dev/null && pwd -P) || {
  printf 'run directory does not exist: %s\n' "$1" >&2
  exit 2
}
case "$RUN_DIR/" in
  "$REPO_ROOT/ab_runs/"*) ;;
  *)
    printf 'run directory must stay under %s/ab_runs\n' "$REPO_ROOT" >&2
    exit 2
    ;;
esac

METADATA="$RUN_DIR/metadata.json"
STATUS_FILE="$RUN_DIR/status"
EVALUATION_DIR="$RUN_DIR/evaluation"
ATTEMPT_MARKER="$RUN_DIR/evaluation-attempt"

[[ -f "$METADATA" ]] || {
  printf 'metadata.json is missing: %s\n' "$RUN_DIR" >&2
  exit 2
}
[[ -f "$STATUS_FILE" ]] || {
  printf 'status file is missing: %s\n' "$RUN_DIR" >&2
  exit 2
}

STATUS=$(tr -d '\r\n' <"$STATUS_FILE")
case "$STATUS" in
  completed|failed:*|timed_out) ;;
  *)
    printf 'agent run is not finished: status=%s\n' "$STATUS" >&2
    exit 2
    ;;
esac

[[ ! -e "$EVALUATION_DIR" && ! -e "$ATTEMPT_MARKER" ]] || {
  printf 'refusing a second remote attempt for run: %s\n' "$RUN_DIR" >&2
  exit 2
}
[[ -n "${CUTE_HARNESS_API_KEY-}" ]] || {
  printf 'CUTE_HARNESS_API_KEY is required only in the evaluator shell\n' >&2
  exit 2
}

TASK_ID=$(jq -er '.task_id' "$METADATA")
ARM=$(jq -er '.arm' "$METADATA")
TRIAL_ID=$(jq -er '.trial_id' "$METADATA")
CANDIDATE_REL=$(jq -er '.candidate_path' "$METADATA")
CANDIDATE="$REPO_ROOT/$CANDIDATE_REL"

[[ -f "$CANDIDATE" ]] || {
  printf 'candidate is missing: %s\n' "$CANDIDATE" >&2
  exit 2
}
CANDIDATE=$(cd "$(dirname "$CANDIDATE")" && pwd -P)/$(basename "$CANDIDATE")
case "$CANDIDATE" in
  "$REPO_ROOT/work/"*) ;;
  *)
    printf 'candidate must stay under %s/work\n' "$REPO_ROOT" >&2
    exit 2
    ;;
esac

LABEL="token-ab:${TASK_ID}:${ARM}:${TRIAL_ID}"
mkdir "$ATTEMPT_MARKER" || {
  printf 'another evaluator already claimed this one-shot trial\n' >&2
  exit 2
}
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$ATTEMPT_MARKER/started-at"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m cute_harness run \
  "$TASK_ID" \
  "$CANDIDATE" \
  --label "$LABEL" \
  --output "$EVALUATION_DIR"
