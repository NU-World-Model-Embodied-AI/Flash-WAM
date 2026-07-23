#!/usr/bin/env bash

# Multi-GPU Flash-WAM RoboTwin evaluation launcher.
#
# This script only fans out work across GPUs. Per-task retries, timeouts,
# progress resume, and summary.json updates are handled by the RoboTwin
# scheduler at:
#   ${POLICY_REPO}/scripts/eval_flashwam_robotwin.sh

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_flashwam_robotwin_multigpu.sh \
    [task|all|/path/tasks.json] [clean|random] [gpu_ids] [base_port] \
    [-- extra_client_args...]

Examples:
  NUM_INFERENCE_STEPS=1 ACTION_NUM_INFERENCE_STEPS=2 \
  bash scripts/eval_flashwam_robotwin_multigpu.sh

  RUN_DATE=20260722_160000 GPU_IDS=0,1 \
  bash scripts/eval_flashwam_robotwin_multigpu.sh /path/to/tasks.json clean 0,1 29536

Path defaults, override with environment variables when needed:
  POLICY_REPO=/zsh/code/Flash-WAM
  ROBOTWIN_REPO=/zsh/code/RoboTwin
  POLICY_PYTHON=/zsh/miniconda3/envs/linbotva/bin/python
  ROBOTWIN_PYTHON=/zsh/miniconda3/envs/robotwin/bin/python
  CHECKPOINT=<POLICY_REPO>/hf_assets/FlashWAM-RoboTwin
  RUN_OUTPUT_DIR=<ROBOTWIN_REPO>/eval_runs/flashwam_robotwin_multigpu
  ROBOTWIN_EVAL_SCRIPT=<POLICY_REPO>/scripts/eval_flashwam_robotwin.sh

Useful environment overrides:
  RUN_DATE=20260722_160000        Reuse this to resume the same multi-GPU run.
  GPU_IDS=0,1,2,3                GPUs to use if not passed positionally.
  ROBOTWIN_TASK_LIST=/path/tasks.json  Reuse a saved task list without re-sampling.
  TASK_SAMPLE_COUNT=10            Randomly sample this many tasks only when selector is all.
  TASK_SAMPLE_SEED=0              Seed for TASK_SAMPLE_COUNT.
  PER_GPU_WORKERS=1               Must remain 1: each Flash-WAM server has one stateful cache.
  PORT_STRIDE=100                 Port range spacing between GPU shards.
  DRY_RUN=true                    Print commands without launching eval.

Forwarded RoboTwin scheduler defaults:
  NUM_INFERENCE_STEPS=1
  ACTION_NUM_INFERENCE_STEPS=2
  CONTINUE_ON_TASK_TIMEOUT=true
  TASK_MAX_RETRIES=3
  EVAL_NUM_EPISODES=50
EOF
}

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

abs_dir_if_exists() {
  local path=$1

  if [[ -d "${path}" ]]; then
    (cd -- "${path}" && pwd)
  else
    printf '%s\n' "${path}"
  fi
}

abs_file_if_exists() {
  local path=$1
  local dir
  local base

  if [[ -f "${path}" ]]; then
    dir=$(dirname -- "${path}")
    base=$(basename -- "${path}")
    printf '%s/%s\n' "$(cd -- "${dir}" && pwd)" "${base}"
  else
    printf '%s\n' "${path}"
  fi
}

write_json_array() {
  local output=$1
  shift

  "${ROBOTWIN_PYTHON}" - "${output}" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tasks = sys.argv[2:]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ $# -gt 0 && ( "${1}" == "-h" || "${1}" == "--help" ) ]]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_POLICY_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)

POLICY_REPO=${POLICY_REPO:-${DEFAULT_POLICY_REPO}}
ROBOTWIN_REPO=${ROBOTWIN_REPO:-/zsh/code/RoboTwin}
POLICY_PYTHON=${POLICY_PYTHON:-/zsh/miniconda3/envs/linbotva/bin/python}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/zsh/miniconda3/envs/robotwin/bin/python}
TASK_SELECTOR=${ROBOTWIN_TASK_LIST:-${TASK_SELECTOR:-all}}
ENVIRONMENT=${ENVIRONMENT:-clean}
GPU_IDS_RAW=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
BASE_PORT=${BASE_PORT:-29536}
PORT_STRIDE=${PORT_STRIDE:-100}
PER_GPU_WORKERS=${PER_GPU_WORKERS:-${PARA_NUM_PER_GPU:-1}}
RUN_DATE=${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}
DRY_RUN=${DRY_RUN:-false}

POLICY_REPO=$(abs_dir_if_exists "${POLICY_REPO}")
ROBOTWIN_REPO=$(abs_dir_if_exists "${ROBOTWIN_REPO}")
CHECKPOINT=${CHECKPOINT:-${POLICY_REPO}/hf_assets/FlashWAM-RoboTwin}
RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR:-${ROBOTWIN_REPO}/eval_runs/flashwam_robotwin_multigpu}
ROBOTWIN_EVAL_SCRIPT=${ROBOTWIN_EVAL_SCRIPT:-${POLICY_REPO}/scripts/eval_flashwam_robotwin.sh}
ROBOTWIN_TASK_CONFIG_DIR=${ROBOTWIN_TASK_CONFIG_DIR:-${ROBOTWIN_REPO}/task_config}

if [[ $# -gt 0 && "${1}" != "--" ]]; then
  TASK_SELECTOR=$1
  shift
fi
if [[ $# -gt 0 && "${1}" != "--" ]]; then
  ENVIRONMENT=$1
  shift
fi
if [[ $# -gt 0 && "${1}" != "--" ]]; then
  GPU_IDS_RAW=$1
  shift
fi
if [[ $# -gt 0 && "${1}" != "--" ]]; then
  BASE_PORT=$1
  shift
fi
if [[ $# -gt 0 && "${1}" == "--" ]]; then
  shift
fi
EXTRA_CLIENT_ARGS=("$@")

if [[ -f "${TASK_SELECTOR}" ]]; then
  TASK_SELECTOR=$(abs_file_if_exists "${TASK_SELECTOR}")
fi
if [[ -d "${CHECKPOINT}" ]]; then
  CHECKPOINT=$(abs_dir_if_exists "${CHECKPOINT}")
elif [[ -d "${POLICY_REPO}/${CHECKPOINT}" ]]; then
  CHECKPOINT=$(abs_dir_if_exists "${POLICY_REPO}/${CHECKPOINT}")
fi

RUN_OUTPUT_DIR=$(mkdir -p -- "${RUN_OUTPUT_DIR}" && cd -- "${RUN_OUTPUT_DIR}" && pwd)
ROBOTWIN_EVAL_SCRIPT=$(abs_file_if_exists "${ROBOTWIN_EVAL_SCRIPT}")
ROBOTWIN_TASK_CONFIG_DIR=$(abs_dir_if_exists "${ROBOTWIN_TASK_CONFIG_DIR}")

case "${ENVIRONMENT}" in
  clean|random) ;;
  *)
    echo "Unsupported environment: ${ENVIRONMENT}; expected clean or random." >&2
    exit 1
    ;;
esac

if [[ ! -d "${POLICY_REPO}" ]]; then
  echo "Policy repo not found: ${POLICY_REPO}" >&2
  exit 1
fi
if [[ ! -d "${ROBOTWIN_REPO}" ]]; then
  echo "RoboTwin repo not found: ${ROBOTWIN_REPO}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_EVAL_SCRIPT}" ]]; then
  echo "RoboTwin scheduler not found: ${ROBOTWIN_EVAL_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_TASK_CONFIG_DIR}/_eval_step_limit.yml" && "${TASK_SELECTOR}" == "all" ]]; then
  echo "RoboTwin task list not found: ${ROBOTWIN_TASK_CONFIG_DIR}/_eval_step_limit.yml" >&2
  exit 1
fi
if [[ ! -f "${POLICY_PYTHON}" && -z "$(command -v "${POLICY_PYTHON}" 2>/dev/null)" ]]; then
  echo "Policy python not found: ${POLICY_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_PYTHON}" && -z "$(command -v "${ROBOTWIN_PYTHON}" 2>/dev/null)" ]]; then
  echo "RoboTwin python not found: ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi
if ! [[ "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BASE_PORT must be a positive integer: ${BASE_PORT}" >&2
  exit 1
fi
if ! [[ "${PORT_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PORT_STRIDE must be a positive integer: ${PORT_STRIDE}" >&2
  exit 1
fi
if ! [[ "${PER_GPU_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_GPU_WORKERS must be a positive integer: ${PER_GPU_WORKERS}" >&2
  exit 1
fi
if [[ "${PER_GPU_WORKERS}" != "1" ]]; then
  echo "PER_GPU_WORKERS must be 1 because one Flash-WAM server owns one stateful cache." >&2
  exit 1
fi

mapfile -t GPU_ID_LIST < <(printf '%s' "${GPU_IDS_RAW}" | tr ', ' '\n' | awk 'NF {print $0}')
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "No GPU selected. Pass gpu_ids or set GPU_IDS." >&2
  exit 1
fi

RUN_STAGE_DIR="${RUN_OUTPUT_DIR}/_multi_gpu/${RUN_DATE}"
TASK_LIST_DIR="${RUN_STAGE_DIR}/task_lists"
LOG_DIR="${RUN_STAGE_DIR}/logs"
mkdir -p -- "${TASK_LIST_DIR}" "${LOG_DIR}"

SELECTED_TASKS_TXT="${TASK_LIST_DIR}/selected_tasks.txt"
SELECTED_TASKS_JSON="${TASK_LIST_DIR}/selected_tasks.json"
"${ROBOTWIN_PYTHON}" - \
  "${TASK_SELECTOR}" "${ROBOTWIN_TASK_CONFIG_DIR}" \
  "${SELECTED_TASKS_TXT}" "${SELECTED_TASKS_JSON}" <<'PY'
import json
import os
import random
import sys
from pathlib import Path

selector = sys.argv[1]
task_config_dir = Path(sys.argv[2])
txt_path = Path(sys.argv[3])
json_path = Path(sys.argv[4])

if selector == "" or selector == "all":
    import yaml

    task_map_path = task_config_dir / "_eval_step_limit.yml"
    with task_map_path.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    tasks = list(task_map.keys())
elif Path(selector).is_file() and selector.endswith(".json"):
    with Path(selector).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Task list JSON must be an array")
    tasks = []
    for item in data:
        if isinstance(item, str):
            tasks.append(item)
        elif isinstance(item, dict) and "task" in item:
            tasks.append(str(item["task"]))
        else:
            raise ValueError(f"Unsupported task entry: {item!r}")
    fixed_task_list = True
else:
    tasks = [selector]

sample_count = int(os.environ.get("TASK_SAMPLE_COUNT", "0") or "0")
if sample_count < 0:
    raise ValueError("TASK_SAMPLE_COUNT must be non-negative")
if "fixed_task_list" in locals() and sample_count:
    raise ValueError(
        "TASK_SAMPLE_COUNT cannot be used with a saved JSON task list; "
        "reuse the list unchanged."
    )
if sample_count:
    if sample_count > len(tasks):
        raise ValueError(
            f"TASK_SAMPLE_COUNT={sample_count} exceeds loaded task count={len(tasks)}"
        )
    sample_seed = int(os.environ.get("TASK_SAMPLE_SEED", os.environ.get("SEED", "0")) or "0")
    tasks = random.Random(sample_seed).sample(tasks, sample_count)

txt_path.write_text("".join(f"{task}\n" for task in tasks), encoding="utf-8")
json_path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
print(f"Selected {len(tasks)} task(s): {json_path}")
PY

mapfile -t TASKS < "${SELECTED_TASKS_TXT}"
if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No task selected." >&2
  exit 1
fi

export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${FLASHWAM_NUM_INFERENCE_STEPS:-1}}"
export ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS:-${FLASHWAM_ACTION_NUM_INFERENCE_STEPS:-2}}"
export FLASHWAM_NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}"
export FLASHWAM_ACTION_NUM_INFERENCE_STEPS="${ACTION_NUM_INFERENCE_STEPS}"
export CONTINUE_ON_TASK_TIMEOUT="${CONTINUE_ON_TASK_TIMEOUT:-true}"
export TASK_MAX_RETRIES="${TASK_MAX_RETRIES:-3}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-50}"
export ROBOTWIN_TASK_CONFIG_DIR

MANIFEST_TSV="${RUN_STAGE_DIR}/launch_manifest.tsv"
{
  printf 'shard_index\tgpu_id\tbase_port\ttask_count\ttask_list\trun_output_dir\tlog\n'
} > "${MANIFEST_TSV}"

SHARD_PIDS=()
SHARD_LOGS=()
SHARD_GPUS=()
CLEANUP_RUNNING=0

cleanup() {
  if [[ "${CLEANUP_RUNNING}" -eq 1 ]]; then
    return 0
  fi
  CLEANUP_RUNNING=1
  trap - EXIT INT TERM

  for pid in "${SHARD_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${SHARD_PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
}

handle_interrupt() {
  local code=$1
  cleanup
  exit "${code}"
}

if ! is_true "${DRY_RUN}"; then
  trap cleanup EXIT
  trap 'handle_interrupt 130' INT
  trap 'handle_interrupt 143' TERM
fi

total_tasks=${#TASKS[@]}
gpu_count=${#GPU_ID_LIST[@]}
shard_size=$(( (total_tasks + gpu_count - 1) / gpu_count ))
launched=0

echo "Run date: ${RUN_DATE}"
echo "Selected tasks: ${total_tasks}"
echo "GPU shards: ${gpu_count} (${GPU_ID_LIST[*]})"
echo "Task lists: ${TASK_LIST_DIR}"
echo "Logs: ${LOG_DIR}"

stop_gpu_guard() {
  local pid waited=0
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    echo "Stopping gpu_guard.py before RoboTwin evaluation (pid=${pid})"
    kill -TERM "${pid}" 2>/dev/null || true
  done < <(pgrep -f '(^|/)python(3)? .*gpu_guard\.py( |$)' || true)
  while pgrep -f '(^|/)python(3)? .*gpu_guard\.py( |$)' >/dev/null; do
    if [[ "${waited}" -ge 60 ]]; then
      echo "gpu_guard.py did not stop within 60 seconds." >&2
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

if ! is_true "${DRY_RUN}"; then
  stop_gpu_guard
fi

for (( shard_idx = 0; shard_idx < gpu_count; shard_idx++ )); do
  gpu_id=${GPU_ID_LIST[shard_idx]}
  start_index=$(( shard_idx * shard_size ))
  end_index=$(( start_index + shard_size ))
  if (( start_index >= total_tasks )); then
    echo "Skip shard=${shard_idx} gpu=${gpu_id}: no task assigned."
    continue
  fi
  if (( end_index > total_tasks )); then
    end_index=${total_tasks}
  fi
  task_count=$(( end_index - start_index ))
  shard_label="shard_$(printf '%02d' "${shard_idx}")_gpu_$(printf '%s' "${gpu_id}" | tr -c 'A-Za-z0-9_.-' '_')"
  shard_task_file="${TASK_LIST_DIR}/${shard_label}.json"
  shard_log="${LOG_DIR}/${shard_label}.log"
  shard_run_output_dir="${RUN_OUTPUT_DIR}/${shard_label}"
  shard_base_port=$(( BASE_PORT + shard_idx * PORT_STRIDE ))
  shard_tasks=("${TASKS[@]:${start_index}:${task_count}}")

  write_json_array "${shard_task_file}" "${shard_tasks[@]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${shard_idx}" "${gpu_id}" "${shard_base_port}" "${task_count}" \
    "${shard_task_file}" "${shard_run_output_dir}" "${shard_log}" \
    >> "${MANIFEST_TSV}"

  cmd=(
    bash "${ROBOTWIN_EVAL_SCRIPT}"
    "${POLICY_REPO}"
    "${ROBOTWIN_REPO}"
    "${POLICY_PYTHON}"
    "${ROBOTWIN_PYTHON}"
    "${CHECKPOINT}"
    "${shard_run_output_dir}"
    "${shard_task_file}"
    "${ENVIRONMENT}"
    "${gpu_id}"
    "${shard_base_port}"
    --
    "${EXTRA_CLIENT_ARGS[@]}"
  )

  echo "Launch shard=${shard_idx} gpu=${gpu_id} tasks=${task_count} port=${shard_base_port}"
  echo "  task_list=${shard_task_file}"
  echo "  output=${shard_run_output_dir}/${ENVIRONMENT}/${RUN_DATE}"
  echo "  log=${shard_log}"

  if is_true "${DRY_RUN}"; then
    printf '  '
    printf '%q ' env "RUN_DATE=${RUN_DATE}" "PARA_NUM_PER_GPU=${PER_GPU_WORKERS}" "EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}" "${cmd[@]}"
    printf '\n'
    continue
  fi

  env RUN_DATE="${RUN_DATE}" PARA_NUM_PER_GPU="${PER_GPU_WORKERS}" EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES}" \
    "${cmd[@]}" > "${shard_log}" 2>&1 &
  SHARD_PIDS[shard_idx]=$!
  SHARD_LOGS[shard_idx]=${shard_log}
  SHARD_GPUS[shard_idx]=${gpu_id}
  launched=$(( launched + 1 ))
done

echo "Launch manifest: ${MANIFEST_TSV}"

if is_true "${DRY_RUN}"; then
  exit 0
fi

if [[ "${launched}" -eq 0 ]]; then
  echo "No shard launched." >&2
  exit 1
fi

failed=0
for shard_idx in "${!SHARD_PIDS[@]}"; do
  if wait "${SHARD_PIDS[shard_idx]}"; then
    echo "Finished shard=${shard_idx} gpu=${SHARD_GPUS[shard_idx]} log=${SHARD_LOGS[shard_idx]}"
  else
    status=$?
    echo "Failed shard=${shard_idx} gpu=${SHARD_GPUS[shard_idx]} status=${status} log=${SHARD_LOGS[shard_idx]}" >&2
    failed=1
  fi
done

"${ROBOTWIN_PYTHON}" - \
  "${RUN_STAGE_DIR}" "${SELECTED_TASKS_JSON}" "${RUN_OUTPUT_DIR}" "${ENVIRONMENT}" "${RUN_DATE}" <<'PY'
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

stage_dir = Path(sys.argv[1])
tasks = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
run_output_dir = Path(sys.argv[3])
environment = sys.argv[4]
run_date = sys.argv[5]
pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
items, rates = {}, []
for task in tasks:
    matches = list(run_output_dir.glob(f"shard_*/{environment}/{run_date}/{task}/_result_{environment}.txt"))
    if len(matches) > 1:
        raise ValueError(f"Multiple result files found for {task}: {matches}")
    item = {"status": "pending", "result_file": None, "success_rate": None}
    if matches:
        result = matches[0]
        values = pattern.findall(result.read_text(encoding="utf-8", errors="replace"))
        if not values:
            raise ValueError(f"No numeric success rate in {result}")
        item = {"status": "completed", "result_file": str(result), "success_rate": float(values[-1])}
        rates.append(item["success_rate"])
    items[task] = item
summary = {
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "expected_tasks": len(tasks), "completed_tasks": len(rates),
    "average_success_rate": sum(rates) / len(rates) if rates else None,
    "tasks": items,
}
tmp = stage_dir / f".aggregate_summary.json.{os.getpid()}.tmp"
tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, stage_dir / "aggregate_summary.json")
print(f"Aggregate summary: {stage_dir / 'aggregate_summary.json'} ({len(rates)}/{len(tasks)} complete)")
PY

trap - EXIT INT TERM
exit "${failed}"
