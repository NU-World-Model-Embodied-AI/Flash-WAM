#!/usr/bin/env python3
"""Evaluate one RoboTwin task through a Flash-WAM websocket policy server."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client
import yaml
from scipy.spatial.transform import Rotation


def _pack_array(value):
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    return value


class FlashWAMClient:
    def __init__(self, host: str, port: int):
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None, ping_interval=None
        )
        self.metadata = self._receive()

    def _receive(self):
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Flash-WAM server error:\n{response}")
        return msgpack.unpackb(response, object_hook=_unpack_array)

    def infer(self, observation: dict) -> dict:
        self._ws.send(msgpack.packb(observation, default=_pack_array))
        return self._receive()

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})

    def close(self) -> None:
        self._ws.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--eval-num-episodes", required=True, type=int)
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--result-file-name", required=True)
    parser.add_argument("--progress-file-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    return parser.parse_args()


def _load_task_args(task_name: str, task_config: str) -> dict:
    from envs import CONFIGS_PATH

    with (Path(CONFIGS_PATH) / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    # The client records machine-readable progress/results. Avoid simulator
    # video encoding here because no ffmpeg pipe is created by this thin
    # websocket adapter.
    args["eval_video_log"] = False

    with (Path(CONFIGS_PATH) / "_embodiment_config.yml").open("r", encoding="utf-8") as f:
        embodiment_configs = yaml.safe_load(f)
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        left_file = right_file = embodiment_configs[embodiment[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_file = embodiment_configs[embodiment[0]]["file_path"]
        right_file = embodiment_configs[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("RoboTwin embodiment must contain one or three entries.")

    for arm, robot_file in (("left", left_file), ("right", right_file)):
        with (Path(robot_file) / "config.yml").open("r", encoding="utf-8") as f:
            args[f"{arm}_embodiment_config"] = yaml.safe_load(f)

    with (Path(CONFIGS_PATH) / "_camera_config.yml").open("r", encoding="utf-8") as f:
        camera_configs = yaml.safe_load(f)
    head_camera = camera_configs[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = head_camera["h"]
    args["head_camera_w"] = head_camera["w"]
    return args


def _flashwam_observation(observation: dict) -> dict:
    images = observation["observation"]
    return {
        "obs": [{
            "observation.images.cam_high": images["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": images["left_camera"]["rgb"],
            "observation.images.cam_right_wrist": images["right_camera"]["rgb"],
            "observation.state": observation["joint_action"]["vector"],
        }],
    }


def _relative_actions_from_response(response: dict) -> np.ndarray:
    actions = np.asarray(response["action"])
    if actions.ndim != 3 or actions.shape[0] != 16:
        raise ValueError(f"Expected Flash-WAM action [16, frames, steps], got {actions.shape}")
    if actions.shape[2] % 4:
        raise ValueError(f"Expected action steps divisible by four, got {actions.shape}")
    return actions


def _add_initial_end_effector_pose(relative_action: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
    def add_arm_pose(relative_arm: np.ndarray, initial_arm: np.ndarray) -> np.ndarray:
        rotation = (Rotation.from_quat(initial_arm[3:7]) * Rotation.from_quat(relative_arm[3:7])).as_quat()
        return np.concatenate((relative_arm[:3] + initial_arm[:3], rotation, relative_arm[7:8]))

    absolute = np.concatenate((
        add_arm_pose(relative_action[:8], initial_pose[:8]),
        add_arm_pose(relative_action[8:], initial_pose[8:]),
    ))
    absolute[3:7] /= np.linalg.norm(absolute[3:7])
    absolute[11:15] /= np.linalg.norm(absolute[11:15])
    return absolute


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_progress(path: Path, task_name: str, task_config: str, seed: int) -> dict:
    if path.is_file():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("task_name") != task_name or progress.get("task_config") != task_config:
            raise ValueError(f"Progress file does not match {task_name}/{task_config}: {path}")
        return progress
    return {
        "task_name": task_name,
        "task_config": task_config,
        "completed_episodes": 0,
        "success_count": 0,
        "next_seed": 10000 * (1 + seed),
    }


def main() -> None:
    args = _parse_args()
    if args.eval_num_episodes <= 0:
        raise ValueError("--eval-num-episodes must be positive.")

    output_dir = Path(args.eval_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / args.progress_file_name
    result_path = output_dir / args.result_file_name
    progress = _load_progress(progress_path, args.task_name, args.task_config, args.seed)
    if progress["completed_episodes"] >= args.eval_num_episodes:
        if not result_path.is_file():
            rate = progress["success_count"] / progress["completed_episodes"]
            result_path.write_text(f"Success rate: {rate}\n", encoding="utf-8")
        return

    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    task_env_class = getattr(importlib.import_module(f"envs.{args.task_name}"), args.task_name)
    task_args = _load_task_args(args.task_name, args.task_config)
    task_env = task_env_class()
    policy = FlashWAMClient(args.host, args.port)
    print(f"Connected to Flash-WAM server: {policy.metadata}", flush=True)

    try:
        while progress["completed_episodes"] < args.eval_num_episodes:
            now_seed = int(progress["next_seed"])
            try:
                task_env.setup_demo(
                    now_ep_num=progress["completed_episodes"], seed=now_seed, is_test=True, **task_args
                )
                episode_info = task_env.play_once()
                task_env.close_env()
            except UnStableError:
                task_env.close_env()
                progress["next_seed"] = now_seed + 1
                _write_json(progress_path, progress)
                continue
            except Exception:
                task_env.close_env()
                progress["next_seed"] = now_seed + 1
                _write_json(progress_path, progress)
                raise

            task_env.setup_demo(
                now_ep_num=progress["completed_episodes"], seed=now_seed, is_test=True, **task_args
            )
            descriptions = generate_episode_descriptions(
                args.task_name, [episode_info["info"]], args.eval_num_episodes
            )
            instruction = random.choice(descriptions[0][args.instruction_type])
            task_env.set_instruction(instruction=instruction)
            policy.reset(instruction)

            success = False
            initial_observation = task_env.get_obs()
            initial_pose = np.asarray(
                initial_observation["endpose"]["left_endpose"]
                + [initial_observation["endpose"]["left_gripper"]]
                + initial_observation["endpose"]["right_endpose"]
                + [initial_observation["endpose"]["right_gripper"]],
                dtype=np.float64,
            )
            first_observation = _flashwam_observation(initial_observation)["obs"][0]
            first = True
            while task_env.take_action_cnt < task_env.step_lim:
                if first:
                    model_input = {"obs": [first_observation]}
                else:
                    model_input = {"obs": None}
                relative_actions = _relative_actions_from_response(policy.infer(model_input))
                key_frames = []
                action_per_frame = relative_actions.shape[2] // 4
                frame_start = 1 if first else 0
                for frame_index in range(frame_start, relative_actions.shape[1]):
                    for action_index in range(relative_actions.shape[2]):
                        action = _add_initial_end_effector_pose(
                            relative_actions[:, frame_index, action_index], initial_pose
                        )
                        task_env.take_action(action, action_type="ee")
                        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                            success = bool(task_env.eval_success)
                            break
                        if (action_index + 1) % action_per_frame == 0:
                            key_frames.append(_flashwam_observation(task_env.get_obs())["obs"][0])
                    if success or task_env.take_action_cnt >= task_env.step_lim:
                        break
                if success or task_env.take_action_cnt >= task_env.step_lim:
                    break
                policy.infer({
                    "obs": key_frames,
                    "compute_kv_cache": True,
                    "state": relative_actions,
                })
                first = False

            task_env.close_env(
                clear_cache=((progress["completed_episodes"] + 1) % task_args["clear_cache_freq"] == 0)
            )
            progress["completed_episodes"] += 1
            progress["success_count"] += int(success)
            progress["next_seed"] = now_seed + 1
            progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _write_json(progress_path, progress)
            print(
                f"{args.task_name}: {progress['success_count']}/{progress['completed_episodes']}",
                flush=True,
            )
    finally:
        policy.close()

    rate = progress["success_count"] / progress["completed_episodes"]
    result_path.write_text(
        f"Task: {args.task_name}\nEpisodes: {progress['completed_episodes']}\n"
        f"Successes: {progress['success_count']}\nSuccess rate: {rate}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
