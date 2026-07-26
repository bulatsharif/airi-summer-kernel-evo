#!/usr/bin/env bash

# Run one fresh OpenCode task non-interactively, optionally enforce a wall-clock
# timeout, preserve the JSONL event stream, and report tokens for the main
# session plus any child/subagent sessions.

set -u

PROGRAM_NAME=${0##*/}
DEFAULT_KILL_AFTER="10s"

usage() {
  cat <<EOF
Usage:
  $PROGRAM_NAME [options] "prompt"
  $PROGRAM_NAME [options] -- "prompt"

Options:
  -d, --dir PATH          Working directory (default: current directory)
  -t, --timeout DURATION  Whole-task timeout, for example 30m or 2h
  -o, --progress PATH     Write the raw OpenCode JSONL stream to PATH
  -a, --agents PATH       Explicit instruction/AGENTS.md file to add
      --kill-after TIME   Force-kill grace period after timeout (default: 10s)
  -p, --prompt TEXT       Prompt as an option instead of a positional argument
  -h, --help              Show this help

Without --timeout, the task has no wall-clock limit.
Without --progress, a temporary JSONL file is used and removed afterward.
Without --agents, OpenCode discovers AGENTS.md from the working directory.
EOF
}

fail() {
  printf '%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  exit 2
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    fail "$1 requires a value"
  fi
}

valid_duration() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?([smhd])?$ ]]
}

absolute_existing_file() {
  local input_path=$1
  local parent
  local name

  if [[ "$input_path" != /* ]]; then
    input_path="$INVOCATION_DIR/$input_path"
  fi
  [[ -f "$input_path" ]] || return 1

  parent=$(cd "$(dirname "$input_path")" 2>/dev/null && pwd -P) || return 1
  name=$(basename "$input_path")
  printf '%s/%s\n' "$parent" "$name"
}

INVOCATION_DIR=$(pwd -P)
WORKING_DIR=$INVOCATION_DIR
TIMEOUT_DURATION=""
KILL_AFTER=$DEFAULT_KILL_AFTER
PROGRESS_ARGUMENT=""
AGENTS_ARGUMENT=""
PROMPT_OPTION=""
declare -a PROMPT_PARTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)
      require_value "$1" "${2-}"
      WORKING_DIR=$2
      shift 2
      ;;
    --dir=*)
      WORKING_DIR=${1#*=}
      shift
      ;;
    -t|--timeout)
      require_value "$1" "${2-}"
      TIMEOUT_DURATION=$2
      shift 2
      ;;
    --timeout=*)
      TIMEOUT_DURATION=${1#*=}
      shift
      ;;
    -o|--progress)
      require_value "$1" "${2-}"
      PROGRESS_ARGUMENT=$2
      shift 2
      ;;
    --progress=*)
      PROGRESS_ARGUMENT=${1#*=}
      shift
      ;;
    -a|--agents)
      require_value "$1" "${2-}"
      AGENTS_ARGUMENT=$2
      shift 2
      ;;
    --agents=*)
      AGENTS_ARGUMENT=${1#*=}
      shift
      ;;
    --kill-after)
      require_value "$1" "${2-}"
      KILL_AFTER=$2
      shift 2
      ;;
    --kill-after=*)
      KILL_AFTER=${1#*=}
      shift
      ;;
    -p|--prompt)
      require_value "$1" "${2-}"
      PROMPT_OPTION=$2
      shift 2
      ;;
    --prompt=*)
      PROMPT_OPTION=${1#*=}
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PROMPT_PARTS+=("$1")
        shift
      done
      ;;
    -*)
      fail "unknown option: $1 (use -- before a prompt that starts with '-')"
      ;;
    *)
      PROMPT_PARTS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROMPT_OPTION" && ${#PROMPT_PARTS[@]} -gt 0 ]]; then
  fail "provide the prompt either positionally or with --prompt, not both"
fi

PROMPT=$PROMPT_OPTION
if [[ ${#PROMPT_PARTS[@]} -gt 0 ]]; then
  PROMPT="${PROMPT_PARTS[*]}"
fi
[[ -n "$PROMPT" ]] || fail "a non-empty prompt is required"

command -v opencode >/dev/null 2>&1 || fail "opencode is not available on PATH"
command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v tee >/dev/null 2>&1 || fail "tee is required"

[[ -d "$WORKING_DIR" ]] || fail "working directory does not exist: $WORKING_DIR"
WORKING_DIR=$(cd "$WORKING_DIR" 2>/dev/null && pwd -P) ||
  fail "cannot access working directory: $WORKING_DIR"

TIMEOUT_COMMAND=""
if [[ -n "$TIMEOUT_DURATION" ]]; then
  valid_duration "$TIMEOUT_DURATION" ||
    fail "invalid timeout '$TIMEOUT_DURATION' (examples: 90s, 30m, 2h)"
  valid_duration "$KILL_AFTER" ||
    fail "invalid --kill-after '$KILL_AFTER' (examples: 10s, 1m)"

  if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_COMMAND=$(command -v gtimeout)
  elif command -v timeout >/dev/null 2>&1 &&
       timeout --version 2>/dev/null | head -n 1 | grep -qi 'coreutils'; then
    TIMEOUT_COMMAND=$(command -v timeout)
  else
    fail "GNU timeout is required for --timeout (macOS: brew install coreutils)"
  fi
fi

AGENTS_FILE=""
INLINE_CONFIG=""
if [[ -n "$AGENTS_ARGUMENT" ]]; then
  AGENTS_FILE=$(absolute_existing_file "$AGENTS_ARGUMENT") ||
    fail "instruction file does not exist: $AGENTS_ARGUMENT"

  BASE_INLINE_CONFIG=${OPENCODE_CONFIG_CONTENT-}
  if [[ -z "$BASE_INLINE_CONFIG" ]]; then
    BASE_INLINE_CONFIG='{}'
  fi

  INLINE_CONFIG=$(
    jq -cn \
      --argjson base "$BASE_INLINE_CONFIG" \
      --arg path "$AGENTS_FILE" \
      '$base + {instructions: (($base.instructions // []) + [$path])}'
  ) || fail "OPENCODE_CONFIG_CONTENT is not valid JSON"
fi

TEMPORARY_PROGRESS=0
if [[ -n "$PROGRESS_ARGUMENT" ]]; then
  if [[ "$PROGRESS_ARGUMENT" = /* ]]; then
    PROGRESS_FILE=$PROGRESS_ARGUMENT
  else
    PROGRESS_FILE="$INVOCATION_DIR/$PROGRESS_ARGUMENT"
  fi

  PROGRESS_PARENT=$(dirname "$PROGRESS_FILE")
  [[ -d "$PROGRESS_PARENT" ]] ||
    fail "progress-file directory does not exist: $PROGRESS_PARENT"
  [[ ! -d "$PROGRESS_FILE" ]] ||
    fail "progress path is a directory: $PROGRESS_FILE"
  : >"$PROGRESS_FILE" || fail "cannot write progress file: $PROGRESS_FILE"
else
  PROGRESS_FILE=$(mktemp "${TMPDIR:-/tmp}/opencode-headless.XXXXXX") ||
    fail "could not create a temporary progress file"
  TEMPORARY_PROGRESS=1
fi

cleanup() {
  if [[ "$TEMPORARY_PROGRESS" -eq 1 && -n "${PROGRESS_FILE-}" ]]; then
    rm -f -- "$PROGRESS_FILE"
  fi
}
trap cleanup EXIT

declare -a OPENCODE_COMMAND=(
  opencode run
  --format json
  --dir "$WORKING_DIR"
  "$PROMPT"
)

declare -a TASK_COMMAND=()
if [[ -n "$INLINE_CONFIG" ]]; then
  TASK_COMMAND=(
    env
    "OPENCODE_CONFIG_CONTENT=$INLINE_CONFIG"
    "${OPENCODE_COMMAND[@]}"
  )
else
  TASK_COMMAND=("${OPENCODE_COMMAND[@]}")
fi

if [[ -n "$TIMEOUT_COMMAND" ]]; then
  TASK_COMMAND=(
    "$TIMEOUT_COMMAND"
    --signal=TERM
    --kill-after="$KILL_AFTER"
    "$TIMEOUT_DURATION"
    "${TASK_COMMAND[@]}"
  )
fi

printf 'Working directory: %s\n' "$WORKING_DIR"
if [[ -n "$AGENTS_FILE" ]]; then
  printf 'Instructions:      %s\n' "$AGENTS_FILE"
else
  printf 'Instructions:      OpenCode AGENTS.md auto-discovery\n'
fi
if [[ "$TEMPORARY_PROGRESS" -eq 0 ]]; then
  printf 'Progress JSONL:     %s\n' "$PROGRESS_FILE"
else
  printf 'Progress JSONL:     temporary\n'
fi
if [[ -n "$TIMEOUT_DURATION" ]]; then
  printf 'Task timeout:       %s (force-kill after %s)\n' "$TIMEOUT_DURATION" "$KILL_AFTER"
else
  printf 'Task timeout:       none\n'
fi
printf '\n'

# Keep the complete JSONL stream in PROGRESS_FILE while showing concise progress
# and assistant text in the terminal.
"${TASK_COMMAND[@]}" |
  tee "$PROGRESS_FILE" |
  jq --unbuffered -r '
    if .type == "text" then
      .part.text
    elif .type == "tool_use" then
      "[tool] \(.part.tool // "unknown"): \(.part.state.title // .part.state.status // "completed")"
    elif .type == "error" then
      "[error] \((.error.data.message // .error.message // .error // "unknown error") | tostring)"
    else
      empty
    end
  '

PIPELINE_STATUS=("${PIPESTATUS[@]}")
RUN_STATUS=${PIPELINE_STATUS[0]}
TEE_STATUS=${PIPELINE_STATUS[1]}
RENDER_STATUS=${PIPELINE_STATUS[2]}

printf '\n'
if [[ -n "$TIMEOUT_DURATION" && "$RUN_STATUS" -eq 124 ]]; then
  printf 'Result: timed out after %s; TERM was sent and KILL followed after %s if needed.\n' \
    "$TIMEOUT_DURATION" "$KILL_AFTER"
elif [[ "$RUN_STATUS" -eq 0 ]]; then
  printf 'Result: completed successfully.\n'
else
  printf 'Result: OpenCode exited with status %s.\n' "$RUN_STATUS"
fi

if [[ "$TEE_STATUS" -ne 0 ]]; then
  printf 'Warning: writing the progress stream failed with status %s.\n' "$TEE_STATUS" >&2
fi
if [[ "$RENDER_STATUS" -ne 0 ]]; then
  printf 'Warning: rendering the progress stream failed with status %s.\n' "$RENDER_STATUS" >&2
fi

SESSION_ID=$(
  jq -r 'select(.sessionID != null) | .sessionID' "$PROGRESS_FILE" 2>/dev/null |
    head -n 1
)

if [[ -z "$SESSION_ID" ]]; then
  printf 'Token usage: unavailable because no OpenCode session ID was emitted.\n'
elif [[ ! "$SESSION_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Token usage: unavailable because the emitted session ID was invalid.\n'
else
  USAGE_JSON=$(
    opencode db --format json "
      WITH RECURSIVE run_sessions(id) AS (
        SELECT '$SESSION_ID'

        UNION ALL

        SELECT s.id
        FROM session AS s
        JOIN run_sessions AS r ON s.parent_id = r.id
      )
      SELECT
        COUNT(*) AS sessions,
        printf('%,d', COALESCE(SUM(tokens_input), 0)) AS input_uncached,
        printf('%,d', COALESCE(SUM(tokens_cache_read), 0)) AS input_cached,
        printf('%,d', COALESCE(SUM(tokens_output), 0)) AS outputs,
        printf('%,d', COALESCE(SUM(tokens_reasoning), 0)) AS reasoning
      FROM session
      WHERE id IN (SELECT id FROM run_sessions)
    " 2>/dev/null
  )
  USAGE_STATUS=$?

  if [[ "$USAGE_STATUS" -ne 0 || -z "$USAGE_JSON" ]]; then
    printf 'Token usage: unavailable because the OpenCode database query failed.\n'
  else
    printf '\nToken usage (main session and subagents):\n'
    printf '%s\n' "$USAGE_JSON" |
      jq -r '
        .[0] |
        "Token input uncached: \(.input_uncached)
Token input cached:   \(.input_cached)
Token outputs:        \(.outputs)
Token reasoning:      \(.reasoning)"
      '
  fi
fi

if [[ "$TEMPORARY_PROGRESS" -eq 0 ]]; then
  printf '\nProgress retained at: %s\n' "$PROGRESS_FILE"
fi

if [[ "$TEE_STATUS" -ne 0 ]]; then
  exit "$TEE_STATUS"
fi
if [[ "$RENDER_STATUS" -ne 0 ]]; then
  exit "$RENDER_STATUS"
fi
exit "$RUN_STATUS"
