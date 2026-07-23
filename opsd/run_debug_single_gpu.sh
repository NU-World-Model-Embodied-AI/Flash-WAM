#!/bin/bash
# Single-GPU debug launcher for Flash-WAM OPSD stage-2 training.
# Defaults:
#   weights: hf_assets/FlashWAM-RoboTwin
#   data:    hf_assets/clean_robotwin
#   steps:   video=2, action=2, train=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NGPU=1
export MASTER_PORT="${MASTER_PORT:-29513}"

export STUDENT_PATH="${STUDENT_PATH:-${PROJECT_ROOT}/hf_assets/FlashWAM-RoboTwin}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/hf_assets/clean_robotwin}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/opsd/debug_output}"

export OPSD_VIDEO_STEPS="${OPSD_VIDEO_STEPS:-2}"
export OPSD_ACTION_STEPS="${OPSD_ACTION_STEPS:-2}"
export OPSD_BACKWARD_PER_CHUNK="${OPSD_BACKWARD_PER_CHUNK:-0}"
export OPSD_SKIP_FIRST_CHUNK_LOSS="${OPSD_SKIP_FIRST_CHUNK_LOSS:-1}"
export OPSD_ACTIVATION_CHECKPOINTING="${OPSD_ACTIVATION_CHECKPOINTING:-0}"
export OPSD_USE_SAFE_DATASET="${OPSD_USE_SAFE_DATASET:-0}"
export OPSD_TASK_LIST="${OPSD_TASK_LIST:-${SCRIPT_DIR}/task_lists/selected_12_tasks.json}"

export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export LOAD_WORKER="${LOAD_WORKER:-0}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"

if [ ! -d "${STUDENT_PATH}/transformer" ] && \
   [ ! -d "${STUDENT_PATH}/online_student/transformer" ]; then
    echo "Missing student checkpoint under: ${STUDENT_PATH}" >&2
    echo "Expected either transformer/ or online_student/transformer/." >&2
    exit 1
fi

if [ ! -f "${DATASET_PATH}/empty_emb.pt" ]; then
    echo "Missing dataset empty embedding: ${DATASET_PATH}/empty_emb.pt" >&2
    exit 1
fi

FIRST_INFO="$(find -L "${DATASET_PATH}" -path '*/meta/info.json' -print -quit 2>/dev/null || true)"
if [ -z "${FIRST_INFO}" ]; then
    echo "No LeRobot meta/info.json found under: ${DATASET_PATH}" >&2
    exit 1
fi

echo "Flash-WAM OPSD debug launch"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  STUDENT_PATH=${STUDENT_PATH}"
echo "  DATASET_PATH=${DATASET_PATH}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  OPSD_VIDEO_STEPS=${OPSD_VIDEO_STEPS}"
echo "  OPSD_ACTION_STEPS=${OPSD_ACTION_STEPS}"
echo "  OPSD_BACKWARD_PER_CHUNK=${OPSD_BACKWARD_PER_CHUNK}"
echo "  OPSD_ACTIVATION_CHECKPOINTING=${OPSD_ACTIVATION_CHECKPOINTING}"
echo "  OPSD_USE_SAFE_DATASET=${OPSD_USE_SAFE_DATASET}"
echo "  OPSD_TASK_LIST=${OPSD_TASK_LIST}"
echo "  MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"

bash "${SCRIPT_DIR}/run.sh"
