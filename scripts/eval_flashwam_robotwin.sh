#!/usr/bin/env bash

# One-GPU Flash-WAM RoboTwin evaluator. The multi-GPU launcher invokes this
# script once per GPU with a disjoint JSON task list.
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 POLICY_REPO ROBOTWIN_REPO POLICY_PYTHON ROBOTWIN_PYTHON CHECKPOINT RUN_OUTPUT_DIR TASKS_JSON clean GPU_ID BASE_PORT" >&2
  exit 2
fi

POLICY_REPO=$1
ROBOTWIN_REPO=$2
POLICY_PYTHON=$3
ROBOTWIN_PYTHON=$4
CHECKPOINT=$5
RUN_OUTPUT_DIR=$6
TASK_LIST=$7
ENVIRONMENT=$8
GPU_ID=$9
BASE_PORT=${10}

if [[ "${ENVIRONMENT}" != "clean" ]]; then
  echo "Only the local clean RoboTwin config is available; set TASK_CONFIG_NAME explicitly after adding a random config." >&2
  exit 2
fi

TASK_CONFIG_NAME=${TASK_CONFIG_NAME:-eval_mv_clean_both}
EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-50}
TASK_MAX_RETRIES=${TASK_MAX_RETRIES:-3}
RUN_DATE=${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}
RESULT_SUFFIX=clean
RUN_DIR="${RUN_OUTPUT_DIR}/${ENVIRONMENT}/${RUN_DATE}"
SERVER_LOG="${RUN_DIR}/server.log"
export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,0.0.0.0${no_proxy:+,${no_proxy}}"
mkdir -p "${RUN_DIR}"

for required in \
  "${CHECKPOINT}/transformer" \
  "${CHECKPOINT}/vae" \
  "${CHECKPOINT}/tokenizer" \
  "${CHECKPOINT}/text_encoder" \
  "${POLICY_REPO}/scripts/serve_flashwam.py" \
  "${POLICY_REPO}/scripts/eval_flashwam_robotwin_client.py" \
  "${ROBOTWIN_REPO}/task_config/${TASK_CONFIG_NAME}.yml"; do
  [[ -e "${required}" ]] || { echo "Required path missing: ${required}" >&2; exit 1; }
done

stop_gpu_guard() {
  local pid
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    echo "Stopping gpu_guard.py before RoboTwin evaluation (pid=${pid})"
    kill -TERM "${pid}" 2>/dev/null || true
  done < <(pgrep -f '(^|/)python(3)? .*gpu_guard\.py( |$)' || true)
}

wait_for_server() {
  "${POLICY_PYTHON}" - "127.0.0.1" "${BASE_PORT}" <<'PY'
import http.client
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        connection = http.client.HTTPConnection(host, port, timeout=2)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        response.read()
        connection.close()
        if response.status == 200:
            raise SystemExit(0)
    except OSError:
        pass
    time.sleep(2)
raise SystemExit("Timed out waiting for Flash-WAM server")
PY
}

update_summary() {
  "${ROBOTWIN_PYTHON}" - "${RUN_DIR}" "${TASK_LIST}" "${CHECKPOINT}" "${TASK_CONFIG_NAME}" <<'PY'
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

run_dir, task_list, checkpoint, task_config = map(Path, sys.argv[1:5])
tasks = json.loads(task_list.read_text(encoding="utf-8"))
if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
    raise ValueError("Task list must be a JSON array of task names")
pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
entries, rates = {}, []
for task in tasks:
    task_dir = run_dir / task
    result = task_dir / "_result_clean.txt"
    progress = task_dir / "_progress_clean.json"
    failed = task_dir / "_timeout_or_failed_clean.txt"
    item = {"status": "pending", "result_file": str(result), "progress_file": str(progress), "timeout_or_failed_file": str(failed), "success_rate": None}
    if result.is_file():
        values = pattern.findall(result.read_text(encoding="utf-8", errors="replace"))
        if not values:
            raise ValueError(f"No numeric success rate in {result}")
        item.update(status="completed", success_rate=float(values[-1]))
        rates.append(item["success_rate"])
    elif failed.is_file():
        item["status"] = "failed_or_timeout"
    elif task_dir.exists():
        item["status"] = "running_or_incomplete"
    entries[task] = item
summary = {
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "run_dir": str(run_dir), "checkpoint": str(checkpoint), "environment": "clean",
    "task_config": str(task_config), "expected_tasks": len(tasks),
    "completed_tasks": len(rates), "average_success_rate": sum(rates) / len(rates) if rates else None,
    "tasks": entries,
}
tmp = run_dir / f".summary.json.{os.getpid()}.tmp"
tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, run_dir / "summary.json")
PY
}

SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

stop_gpu_guard
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${POLICY_PYTHON}" -u "${POLICY_REPO}/scripts/serve_flashwam.py" \
    --checkpoint-dir "${CHECKPOINT}" --host 127.0.0.1 --port "${BASE_PORT}" \
    --local-rank 0 --num-inference-steps "${NUM_INFERENCE_STEPS:-1}" \
    --action-num-inference-steps "${ACTION_NUM_INFERENCE_STEPS:-2}" \
    --save-root "${RUN_DIR}/server_visualization" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
wait_for_server

mapfile -t TASKS < <("${ROBOTWIN_PYTHON}" - "${TASK_LIST}" <<'PY'
import json
import sys
tasks = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(tasks, list) or len(tasks) != len(set(tasks)) or not all(isinstance(task, str) for task in tasks):
    raise ValueError("Task list must contain unique string task names")
print(*tasks, sep="\n")
PY
)
update_summary
for task in "${TASKS[@]}"; do
  task_dir="${RUN_DIR}/${task}"
  result_file="${task_dir}/_result_${RESULT_SUFFIX}.txt"
  progress_file="${task_dir}/_progress_${RESULT_SUFFIX}.json"
  failed_file="${task_dir}/_timeout_or_failed_${RESULT_SUFFIX}.txt"
  if [[ -f "${result_file}" ]]; then
    echo "Skip completed task: ${task}"
    continue
  fi
  mkdir -p "${task_dir}"
  success=false
  for ((attempt=1; attempt<=TASK_MAX_RETRIES+1; attempt++)); do
    echo "Evaluate task=${task} attempt=${attempt}/$((TASK_MAX_RETRIES + 1))"
    if (
      cd "${ROBOTWIN_REPO}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" SAPIEN_RENDER_DEVICE=cuda:0 \
        "${ROBOTWIN_PYTHON}" -u "${POLICY_REPO}/scripts/eval_flashwam_robotwin_client.py" \
          --host 127.0.0.1 --port "${BASE_PORT}" --task-name "${task}" \
          --task-config "${TASK_CONFIG_NAME}" --eval-num-episodes "${EVAL_NUM_EPISODES}" \
          --eval-output-dir "${task_dir}" --result-file-name "$(basename "${result_file}")" \
          --progress-file-name "$(basename "${progress_file}")" \
          --seed "${SEED:-0}" --instruction-type "${INSTRUCTION_TYPE:-unseen}"
    ) > "${task_dir}/client.log" 2>&1 && [[ -f "${result_file}" ]]; then
      success=true
      rm -f "${failed_file}"
      break
    fi
  done
  if [[ "${success}" != true ]]; then
    printf 'Task: %s\nRetries: %s\nSee: %s\n' "${task}" "${TASK_MAX_RETRIES}" "${task_dir}/client.log" > "${failed_file}"
  fi
  update_summary
done
