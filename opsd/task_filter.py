"""Select the fixed OPSD task subset without modifying the source dataset."""
from __future__ import annotations

import json
from pathlib import Path


def load_task_list(path: str) -> list[str]:
    """Load the externally saved RoboTwin task list used for this run."""
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError(f"OPSD task list must be a JSON array of strings: {path}")
    if len(tasks) != 12 or len(set(tasks)) != 12:
        raise ValueError(f"OPSD task list must contain 12 unique tasks: {path}")
    return tasks


def task_name_from_dataset_dir(dataset_dir: str) -> str:
    """Extract a RoboTwin task name from a clean-50 dataset directory name."""
    name = Path(dataset_dir).name
    for marker in ("-demo_", "-piper_"):
        if marker in name:
            return name.split(marker, 1)[0]
    raise ValueError(f"Cannot determine task name from dataset directory: {dataset_dir}")


def select_task_dataset_roots(dataset_path: str, tasks: list[str]) -> list[str]:
    """Return only the existing LeRobot sub-datasets for the selected tasks."""
    roots: dict[str, str] = {}
    dataset_root = Path(dataset_path)
    for candidate in dataset_root.glob("*/*"):
        if not (candidate / "meta" / "info.json").is_file():
            continue
        task = task_name_from_dataset_dir(str(candidate))
        if task in tasks:
            if task in roots:
                raise ValueError(f"Multiple training datasets found for task {task!r}")
            roots[task] = str(candidate)

    missing = [task for task in tasks if task not in roots]
    if missing:
        raise FileNotFoundError(
            "Selected tasks have no LeRobot dataset under "
            f"{dataset_path}: {', '.join(missing)}")
    return [roots[task] for task in tasks]
