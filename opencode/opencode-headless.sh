#!/usr/bin/env bash

# Run one fresh OpenCode task non-interactively and detached by default,
# optionally enforce a wall-clock timeout, preserve the JSONL event stream, and
# report tokens for the main session plus any child/subagent sessions.

set -u

PROGRAM_NAME=${0##*/}
DEFAULT_KILL_AFTER="10s"
DEFAULT_ATTACH_LINES=100
ORIGINAL_ARGS=("$@")
INVOCATION_DIR=$(pwd -P)

usage() {
  cat <<EOF
Usage:
  $PROGRAM_NAME [options] "prompt"
  $PROGRAM_NAME [options] -- "prompt"
  $PROGRAM_NAME --attach RUN_DIR

Options:
  -d, --dir PATH          Working directory (default: current directory)
  -t, --timeout DURATION  Whole-task timeout, for example 30m or 2h
  -o, --progress PATH     Write the raw OpenCode JSONL stream to PATH
  -a, --agents PATH       Explicit instruction/AGENTS.md file to add
      --kill-after TIME   Force-kill grace period after timeout (default: 10s)
      --run-dir PATH      Detached run state directory (default: temporary)
      --foreground        Stay attached to the current terminal
      --attach RUN_DIR    Follow a detached run's output
  -p, --prompt TEXT       Prompt as an option instead of a positional argument
  -h, --help              Show this help

Runs are detached by default and survive terminal closure.
Without --timeout, the task has no wall-clock limit.
Without --progress, detached JSONL is kept in the run directory; foreground
JSONL uses a temporary file that is removed afterward.
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

status_exit_code() {
  case "$1" in
    completed)
      return 0
      ;;
    timed_out)
      return 124
      ;;
    failed:*)
      local code=${1#*:}
      if [[ "$code" =~ ^[1-9][0-9]*$ && "$code" -le 255 ]]; then
        return "$code"
      fi
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

attach_run() {
  local requested_dir=$1
  local run_dir
  local output_file
  local pid_file
  local status_file
  local run_pid=""
  local run_status="unknown"
  local tail_pid=""

  if [[ "$requested_dir" != /* ]]; then
    requested_dir="$INVOCATION_DIR/$requested_dir"
  fi
  [[ -d "$requested_dir" ]] || fail "run directory does not exist: $1"
  run_dir=$(cd "$requested_dir" 2>/dev/null && pwd -P) ||
    fail "cannot access run directory: $1"

  output_file="$run_dir/output.log"
  pid_file="$run_dir/pid"
  status_file="$run_dir/status"
  [[ -f "$output_file" ]] || fail "run output does not exist: $output_file"

  if [[ -f "$pid_file" ]]; then
    run_pid=$(head -n 1 "$pid_file")
  fi
  if [[ -f "$status_file" ]]; then
    run_status=$(head -n 1 "$status_file")
  fi

  printf 'Run directory: %s\n' "$run_dir"
  printf 'Status:        %s\n\n' "$run_status"

  if [[ ! "$run_status" =~ ^(starting|running)$ ]] ||
     [[ ! "$run_pid" =~ ^[1-9][0-9]*$ ]] ||
     ! kill -0 "$run_pid" 2>/dev/null; then
    cat "$output_file"
    printf '\nRun status: %s\n' "$run_status"
    status_exit_code "$run_status"
    return $?
  fi

  tail -n "$DEFAULT_ATTACH_LINES" -f "$output_file" &
  tail_pid=$!

  stop_tail() {
    if [[ -n "$tail_pid" ]]; then
      kill "$tail_pid" 2>/dev/null || true
      wait "$tail_pid" 2>/dev/null || true
      tail_pid=""
    fi
  }
  trap stop_tail EXIT
  trap 'stop_tail; exit 130' INT TERM

  while kill -0 "$run_pid" 2>/dev/null; do
    if [[ -f "$status_file" ]]; then
      run_status=$(head -n 1 "$status_file")
      [[ "$run_status" =~ ^(starting|running)$ ]] || break
    fi
    sleep 1
  done

  # BSD and GNU tail normally poll once per second. Give it one last poll so
  # the final result and token report are displayed before stopping it.
  sleep 2
  stop_tail
  trap - EXIT INT TERM

  if [[ -f "$status_file" ]]; then
    run_status=$(head -n 1 "$status_file")
  fi
  printf '\nRun status: %s\n' "$run_status"
  status_exit_code "$run_status"
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

if [[ "${1-}" == "--attach" ]]; then
  require_value "$1" "${2-}"
  [[ $# -eq 2 ]] || fail "--attach accepts exactly one run directory"
  attach_run "$2"
  exit $?
fi

WORKING_DIR=$INVOCATION_DIR
TIMEOUT_DURATION=""
KILL_AFTER=$DEFAULT_KILL_AFTER
PROGRESS_ARGUMENT=""
AGENTS_ARGUMENT=""
PROMPT_OPTION=""
RUN_DIR_ARGUMENT=""
DETACH=1
DETACHED_CHILD=0
DETACHED_RUN_DIR=""
DETACHED_FINALIZED=0
TEMPORARY_PROGRESS=0
PROGRESS_FILE=""
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
    --run-dir)
      require_value "$1" "${2-}"
      RUN_DIR_ARGUMENT=$2
      shift 2
      ;;
    --run-dir=*)
      RUN_DIR_ARGUMENT=${1#*=}
      shift
      ;;
    --foreground)
      DETACH=0
      shift
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --_detached-child)
      require_value "$1" "${2-}"
      DETACHED_CHILD=1
      DETACHED_RUN_DIR=$2
      DETACH=0
      shift 2
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

if [[ "$DETACH" -eq 0 && "$DETACHED_CHILD" -eq 0 &&
      -n "$RUN_DIR_ARGUMENT" ]]; then
  fail "--run-dir is only valid for detached runs"
fi

cleanup() {
  local exit_status=$?

  if [[ "$TEMPORARY_PROGRESS" -eq 1 && -n "$PROGRESS_FILE" ]]; then
    rm -f -- "$PROGRESS_FILE"
  fi

  if [[ "$DETACHED_CHILD" -eq 1 && "$DETACHED_FINALIZED" -eq 0 &&
        -n "$DETACHED_RUN_DIR" && -d "$DETACHED_RUN_DIR" ]]; then
    printf 'failed:%s\n' "$exit_status" >"$DETACHED_RUN_DIR/status"
  fi
}
trap cleanup EXIT

if [[ "$DETACHED_CHILD" -eq 1 ]]; then
  [[ -d "$DETACHED_RUN_DIR" ]] ||
    fail "detached run directory does not exist: $DETACHED_RUN_DIR"
  printf '%s\n' "$$" >"$DETACHED_RUN_DIR/pid"
  printf 'running\n' >"$DETACHED_RUN_DIR/status"
fi

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

if [[ "$DETACH" -eq 1 && "$DETACHED_CHILD" -eq 0 ]]; then
  command -v nohup >/dev/null 2>&1 || fail "nohup is required for detached runs"

  if [[ -n "$RUN_DIR_ARGUMENT" ]]; then
    if [[ "$RUN_DIR_ARGUMENT" = /* ]]; then
      DETACHED_RUN_DIR=$RUN_DIR_ARGUMENT
    else
      DETACHED_RUN_DIR="$INVOCATION_DIR/$RUN_DIR_ARGUMENT"
    fi

    if [[ -e "$DETACHED_RUN_DIR" ]]; then
      [[ -d "$DETACHED_RUN_DIR" ]] ||
        fail "run state path is not a directory: $DETACHED_RUN_DIR"
      [[ -z "$(find "$DETACHED_RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] ||
        fail "run state directory is not empty: $DETACHED_RUN_DIR"
    else
      mkdir -p "$DETACHED_RUN_DIR" ||
        fail "cannot create run state directory: $DETACHED_RUN_DIR"
    fi
    DETACHED_RUN_DIR=$(cd "$DETACHED_RUN_DIR" 2>/dev/null && pwd -P) ||
      fail "cannot access run state directory: $DETACHED_RUN_DIR"
  else
    RUNS_PARENT="${TMPDIR:-/tmp}/opencode-headless-runs"
    mkdir -p "$RUNS_PARENT" ||
      fail "cannot create detached-runs directory: $RUNS_PARENT"
    DETACHED_RUN_DIR=$(mktemp -d "$RUNS_PARENT/run.XXXXXX") ||
      fail "cannot create detached run directory"
  fi

  if [[ -n "$PROGRESS_ARGUMENT" ]]; then
    if [[ "$PROGRESS_ARGUMENT" = /* ]]; then
      DETACHED_PROGRESS_FILE=$PROGRESS_ARGUMENT
    else
      DETACHED_PROGRESS_FILE="$INVOCATION_DIR/$PROGRESS_ARGUMENT"
    fi
    DETACHED_PROGRESS_PARENT=$(dirname "$DETACHED_PROGRESS_FILE")
    [[ -d "$DETACHED_PROGRESS_PARENT" ]] ||
      fail "progress-file directory does not exist: $DETACHED_PROGRESS_PARENT"
    [[ ! -d "$DETACHED_PROGRESS_FILE" ]] ||
      fail "progress path is a directory: $DETACHED_PROGRESS_FILE"
  else
    DETACHED_PROGRESS_FILE="$DETACHED_RUN_DIR/progress.jsonl"
  fi

  SCRIPT_PATH=${BASH_SOURCE[0]}
  if [[ "$SCRIPT_PATH" != */* ]]; then
    SCRIPT_PATH=$(command -v "$SCRIPT_PATH") ||
      fail "cannot resolve runner script path"
  fi
  if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PARENT=$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd -P) ||
      fail "cannot resolve runner script directory"
    SCRIPT_PATH="$SCRIPT_PARENT/$(basename "$SCRIPT_PATH")"
  fi

  printf 'starting\n' >"$DETACHED_RUN_DIR/status"
  printf '%s\n' "$DETACHED_PROGRESS_FILE" >"$DETACHED_RUN_DIR/progress-path"
  : >"$DETACHED_RUN_DIR/output.log" ||
    fail "cannot create detached output log"

  nohup "$SCRIPT_PATH" \
    --_detached-child "$DETACHED_RUN_DIR" \
    --progress "$DETACHED_PROGRESS_FILE" \
    "${ORIGINAL_ARGS[@]}" \
    >"$DETACHED_RUN_DIR/output.log" 2>&1 </dev/null &
  DETACHED_PID=$!
  printf '%s\n' "$DETACHED_PID" >"$DETACHED_RUN_DIR/pid"

  printf 'Detached run started.\n'
  printf 'PID:           %s\n' "$DETACHED_PID"
  printf 'Run directory: %s\n' "$DETACHED_RUN_DIR"
  printf 'Output log:    %s\n' "$DETACHED_RUN_DIR/output.log"
  printf 'Progress JSONL:%s\n' " $DETACHED_PROGRESS_FILE"
  printf 'Attach with:\n  '
  printf '%q --attach %q\n' "$SCRIPT_PATH" "$DETACHED_RUN_DIR"
  exit 0
fi

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

FINAL_STATUS=$RUN_STATUS
if [[ "$TEE_STATUS" -ne 0 ]]; then
  FINAL_STATUS=$TEE_STATUS
elif [[ "$RENDER_STATUS" -ne 0 ]]; then
  FINAL_STATUS=$RENDER_STATUS
fi

if [[ "$DETACHED_CHILD" -eq 1 ]]; then
  if [[ -n "$TIMEOUT_DURATION" && "$RUN_STATUS" -eq 124 ]]; then
    printf 'timed_out\n' >"$DETACHED_RUN_DIR/status"
  elif [[ "$FINAL_STATUS" -eq 0 ]]; then
    printf 'completed\n' >"$DETACHED_RUN_DIR/status"
  else
    printf 'failed:%s\n' "$FINAL_STATUS" >"$DETACHED_RUN_DIR/status"
  fi
  DETACHED_FINALIZED=1
fi

exit "$FINAL_STATUS"
