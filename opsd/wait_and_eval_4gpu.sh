#!/usr/bin/env bash

# Wait for a successful OPSD step-5000 checkpoint, then evaluate it on the
# same externally persisted twelve-task list.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${SCRIPT_DIR}/runs/opsd_4gpu_train}
TASK_LIST=${OPSD_TASK_LIST:-${SCRIPT_DIR}/task_lists/selected_12_tasks.json}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-${SCRIPT_DIR}/runs/robotwin_eval}
POLL_SECONDS=${POLL_SECONDS:-60}
RUN_DATE=${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}
STEP_DIR="${TRAIN_OUTPUT_DIR}/checkpoints/step_5000"
ONLINE_TRANSFORMER="${STEP_DIR}/online_student/transformer"
EVAL_MODEL_DIR="${STEP_DIR}/flashwam_eval_model"

[[ -f "${TASK_LIST}" ]] || { echo "Task list not found: ${TASK_LIST}" >&2; exit 1; }
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_SECONDS must be positive." >&2; exit 1; }

while [[ ! -s "${ONLINE_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]]; do
  if ! pgrep -f '[o]psd/train.py' >/dev/null; then
    echo "step_5000 checkpoint is absent and no OPSD train process is running." >&2
    exit 1
  fi
  echo "$(date '+%F %T') waiting for ${ONLINE_TRANSFORMER}"
  sleep "${POLL_SECONDS}"
done

[[ -f "${ONLINE_TRANSFORMER}/config.json" ]] || { echo "Checkpoint config is missing: ${ONLINE_TRANSFORMER}" >&2; exit 1; }
while pgrep -f '[o]psd/train.py' >/dev/null; do
  echo "$(date '+%F %T') checkpoint is present; waiting for OPSD workers to release GPUs"
  sleep "${POLL_SECONDS}"
done
if ! rg -q 'OPSD training completed!' "${TRAIN_OUTPUT_DIR}/train.log"; then
  echo "Checkpoint exists but the training log does not report successful completion." >&2
  exit 1
fi
bash "${REPO_DIR}/scripts/prepare_flashwam_eval_checkpoint.sh" \
  "${ONLINE_TRANSFORMER}" "${EVAL_MODEL_DIR}"

GPU_IDS=${GPU_IDS:-0,1,2,3} \
ROBOTWIN_TASK_LIST="${TASK_LIST}" \
CHECKPOINT="${EVAL_MODEL_DIR}" \
RUN_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
RUN_DATE="${RUN_DATE}" \
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1} \
ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS:-2} \
EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-50} \
bash "${REPO_DIR}/scripts/eval_flashwam_robotwin_multigpu.sh"

"${ROBOTWIN_PYTHON:-/zsh/miniconda3/envs/robotwin/bin/python}" \
  "${REPO_DIR}/scripts/audit_flashwam_robotwin_eval.py" \
  --task-list "${TASK_LIST}" \
  --checkpoint "${EVAL_MODEL_DIR}" \
  --run-stage-dir "${EVAL_OUTPUT_DIR}/_multi_gpu/${RUN_DATE}" \
  --eval-output-dir "${EVAL_OUTPUT_DIR}"
