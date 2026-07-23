#!/bin/bash
# Launch Flash-WAM OPSD stage-2 training.
#   STUDENT_PATH=/path/to/stage1_flashwam DATASET_PATH=/path/to/dataset \
#   OPSD_TASK_LIST=/path/to/selected_12_tasks.json NGPU=4 bash opsd/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${STUDENT_PATH:-${TEACHER_PATH:-}}}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
OPSD_TASK_LIST="${OPSD_TASK_LIST:-}"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"

if [ -z "${OPSD_TASK_LIST}" ] || [ ! -f "${OPSD_TASK_LIST}" ]; then
    echo "Set OPSD_TASK_LIST to the saved JSON task list before launching OPSD." >&2
    exit 1
fi

ARGS=()
[ -n "$STUDENT_MODEL_PATH" ] && ARGS+=(--student-model-path "$STUDENT_MODEL_PATH")
[ -n "$DATASET_PATH" ]       && ARGS+=(--dataset-path "$DATASET_PATH")
[ -n "$OUTPUT_DIR" ]         && ARGS+=(--output-dir "$OUTPUT_DIR")
[ -n "$RESUME_FROM_STEP" ]   && ARGS+=(--resume-from-step "$RESUME_FROM_STEP")
[ -n "$RESUME_FROM_PATH" ]   && ARGS+=(--resume-from-path "$RESUME_FROM_PATH")
ARGS+=(--task-list "$OPSD_TASK_LIST")

stop_gpu_guard() {
    local pid waited=0
    while read -r pid; do
        [ -n "${pid}" ] || continue
        echo "Stopping gpu_guard.py before OPSD launch (pid=${pid})"
        kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f '(^|/)python(3)? .*gpu_guard\.py( |$)' || true)
    while pgrep -f '(^|/)python(3)? .*gpu_guard\.py( |$)' >/dev/null; do
        if [ "${waited}" -ge 60 ]; then
            echo "gpu_guard.py did not stop within 60 seconds." >&2
            exit 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

stop_gpu_guard

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    "${ARGS[@]}"
