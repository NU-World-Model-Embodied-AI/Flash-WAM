#!/usr/bin/env python3
"""Validate the final artifacts for one Flash-WAM RoboTwin evaluation run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-list", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-stage-dir", required=True)
    parser.add_argument("--eval-output-dir", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    task_list = Path(args.task_list).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    stage_dir = Path(args.run_stage_dir).resolve()
    eval_output_dir = Path(args.eval_output_dir).resolve()
    tasks = json.loads(task_list.read_text(encoding="utf-8"))
    require(isinstance(tasks, list) and len(tasks) == 12, "Task list must contain exactly 12 tasks.")
    require(len(set(tasks)) == 12 and all(isinstance(task, str) for task in tasks), "Task list must contain unique strings.")
    require((checkpoint / "transformer" / "config.json").is_file(), "Missing evaluation transformer config.")
    require((checkpoint / "transformer" / "diffusion_pytorch_model.safetensors").is_file(), "Missing evaluation transformer weights.")
    for component in ("vae", "tokenizer", "text_encoder"):
        require((checkpoint / component).is_dir(), f"Missing evaluation component: {component}")

    manifest = stage_dir / "launch_manifest.tsv"
    aggregate = stage_dir / "aggregate_summary.json"
    require(manifest.is_file(), f"Missing launch manifest: {manifest}")
    manifest_rows = manifest.read_text(encoding="utf-8").strip().splitlines()
    require(len(manifest_rows) == 5, "Expected exactly four GPU-shard rows in launch manifest.")
    require(
        manifest_rows[0] == "shard_index\tgpu_id\tbase_port\ttask_count\ttask_list\trun_output_dir\tlog",
        "Launch manifest header is invalid.",
    )
    shard_rows = [row.split("\t") for row in manifest_rows[1:]]
    require(all(len(row) == 7 for row in shard_rows), "Launch manifest contains a malformed shard row.")
    require({row[0] for row in shard_rows} == {"0", "1", "2", "3"}, "Expected shard indexes 0 through 3.")
    require(len({row[1] for row in shard_rows}) == 4, "Expected four distinct GPU IDs.")

    manifest_tasks: list[str] = []
    for shard_index, _, _, task_count, task_list_path, shard_output_dir, shard_log in shard_rows:
        require(task_count.isdigit() and int(task_count) > 0, f"Invalid task count in shard {shard_index}.")
        shard_tasks_path = Path(task_list_path)
        require(shard_tasks_path.is_file(), f"Missing shard task list: {shard_tasks_path}")
        shard_tasks = json.loads(shard_tasks_path.read_text(encoding="utf-8"))
        require(
            isinstance(shard_tasks, list)
            and len(shard_tasks) == int(task_count)
            and all(isinstance(task, str) for task in shard_tasks),
            f"Invalid shard task list: {shard_tasks_path}",
        )
        manifest_tasks.extend(shard_tasks)
        require(Path(shard_log).is_file(), f"Missing shard log: {shard_log}")
        summary_paths = list(Path(shard_output_dir).glob(f"*/{stage_dir.name}/summary.json"))
        require(len(summary_paths) == 1, f"Expected one shard summary for shard {shard_index}.")
        shard_summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        require(
            shard_summary.get("expected_tasks") == int(task_count)
            and shard_summary.get("completed_tasks") == int(task_count),
            f"Shard summary is incomplete for shard {shard_index}.",
        )
        require(
            isinstance(shard_summary.get("tasks"), dict)
            and set(shard_summary["tasks"]) == set(shard_tasks),
            f"Shard summary task set differs for shard {shard_index}.",
        )
    require(len(manifest_tasks) == 12 and set(manifest_tasks) == set(tasks), "Shard task lists differ from the selected task list.")

    require(aggregate.is_file(), f"Missing aggregate summary: {aggregate}")
    summary = json.loads(aggregate.read_text(encoding="utf-8"))
    require(summary.get("expected_tasks") == 12, "Aggregate summary expected_tasks is not 12.")
    require(summary.get("completed_tasks") == 12, "Aggregate summary does not report 12 completed tasks.")
    summary_tasks = summary.get("tasks")
    require(isinstance(summary_tasks, dict) and set(summary_tasks) == set(tasks), "Aggregate summary task set differs from task list.")

    task_results = {}
    for task in tasks:
        item = summary_tasks[task]
        result_file = Path(item.get("result_file", ""))
        rate = item.get("success_rate")
        require(item.get("status") == "completed", f"Task is not completed: {task}")
        require(result_file.is_file(), f"Result file missing for {task}: {result_file}")
        require(isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0, f"Invalid success rate for {task}: {rate}")
        task_results[task] = {"result_file": str(result_file), "success_rate": float(rate)}

    audit = {
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete",
        "task_list": str(task_list),
        "checkpoint": str(checkpoint),
        "run_stage_dir": str(stage_dir),
        "eval_output_dir": str(eval_output_dir),
        "gpu_shards": 4,
        "tasks": task_results,
        "average_success_rate": summary.get("average_success_rate"),
    }
    output = stage_dir / "final_audit.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Final audit: {output}")


if __name__ == "__main__":
    main()
