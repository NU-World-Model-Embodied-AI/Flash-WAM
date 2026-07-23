"""Isolated trainer for Flash-WAM OPSD stage-2 training."""
from __future__ import annotations

import gc
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from distillation.ema import update_ema
from wan_va.distributed.fsdp import apply_ac, shard_model
from wan_va.distributed.util import _configure_model, dist_mean
from dataset.lerobot_latent_dataset import LatentLeRobotDataset
from wan_va.modules.utils import load_transformer, load_vae
from wan_va.utils import FlowMatchScheduler, logger, warmup_constant_lambda

try:
    from .data import OPSDDataMixin, move_batch_to_device, split_batch_into_chunks
    from .rollout import OPSDRolloutMixin
    from .task_filter import load_task_list, select_task_dataset_roots
except ImportError:
    from data import OPSDDataMixin, move_batch_to_device, split_batch_into_chunks
    from rollout import OPSDRolloutMixin
    from task_filter import load_task_list, select_task_dataset_roots


def _resolve_fresh_model_paths(path: str) -> Tuple[str, str]:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root

    online = root / "online_student" / "transformer"
    target = root / "target_student" / "transformer"
    if online.is_dir():
        return str(online), str(target if target.is_dir() else online)

    transformer = root / "transformer"
    if transformer.is_dir():
        return str(transformer), str(transformer)

    if root.is_dir():
        return str(root), str(root)

    raise FileNotFoundError(f"Could not resolve student model path: {path}")


def _resolve_resume_model_paths(path: str) -> Tuple[str, str]:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    online = root / "online_student" / "transformer"
    target = root / "target_student" / "transformer"
    if not online.is_dir() or not target.is_dir():
        raise FileNotFoundError(
            f"Resume path must contain online_student/transformer and "
            f"target_student/transformer: {root}")
    return str(online), str(target)


def _resolve_vae_root(path: str) -> str:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root

    for candidate in (root, *root.parents):
        if (candidate / "vae").is_dir():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve VAE root from {path}; expected a directory containing vae/."
    )


class _SelectedLatentDataset(torch.utils.data.Dataset):
    """Concatenate only the LeRobot sub-datasets selected for this OPSD run."""

    def __init__(self, datasets):
        self._datasets = datasets
        self._starts = []
        start = 0
        for dataset in datasets:
            self._starts.append(start)
            start += len(dataset)
        self._length = start

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for dataset, start in zip(self._datasets, self._starts):
            if index < start + len(dataset):
                return dataset[index - start]
        raise IndexError(index)


class FlashWAMOPSDTrainer(OPSDDataMixin, OPSDRolloutMixin):
    """Trainer for the action-flow OPSD stage."""

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.video_scheduler = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)

        self.wandb = None
        if config.enable_wandb and config.rank == 0:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project="flashwam_opsd",
                    entity=getattr(config, "wandb_entity", None),
                    config=dict(config),
                )
            except ImportError:
                logger.warning("wandb is not installed; OPSD training will not log to wandb.")

        student_path, target_path = self._resolve_model_paths(config)
        if config.rank == 0:
            logger.info("Starting Flash-WAM OPSD stage-2 setup")
            logger.info(f"  Student init: {student_path}")
            logger.info(f"  EMA target:   {target_path}")
            logger.info(
                f"  OPSD steps: video={config.opsd_video_num_inference_steps}, "
                f"action={config.opsd_action_num_inference_steps}")
            logger.info("  Loss: action flow MSE only")
            logger.info(f"  Pred-video detach for action: {config.opsd_pred_video_detach_for_action}")
            logger.info(f"  Backward per chunk: {config.opsd_backward_per_chunk}")

        self._build_train_loader(config)

        logger.info("Loading online student (trainable) ...")
        self.student = load_transformer(
            student_path, torch_dtype=torch.float32, torch_device="cpu")
        if config.opsd_activation_checkpointing:
            apply_ac(self.student)
        elif config.rank == 0:
            logger.info("  Activation checkpointing: disabled")
        self.student = self.student.to(self.dtype)
        self.student = _configure_model(
            model=self.student,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.student.requires_grad_(True)

        logger.info("Loading EMA target student (frozen teacher) ...")
        self.target_student = load_transformer(
            target_path, torch_dtype=self.dtype, torch_device="cpu")
        self.target_student = self.target_student.to(self.dtype)
        self.target_student = _configure_model(
            model=self.target_student,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        self.target_student.requires_grad_(False)
        self.target_student.eval()

        logger.info("Loading VAE for rollout video decoding ...")
        vae_root = _resolve_vae_root(config.student_model_path)
        self.vae = load_vae(
            os.path.join(vae_root, "vae"),
            torch_dtype=self.dtype,
            torch_device="cpu",
        )
        self.video_processor = VideoProcessor(vae_scale_factor=1)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(
                step, warmup_steps=config.warmup_steps),
        )
        if self.step > 0:
            for _ in range(self.step):
                self.lr_scheduler.step()

        self.save_dir = Path(config.output_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_video_dir = Path(config.output_dir) / "rollout_videos"
        self.rollout_video_dir.mkdir(parents=True, exist_ok=True)

    def _build_train_loader(self, config) -> None:
        logger.info("Loading OPSD dataset ...")
        if not config.opsd_task_list:
            raise ValueError("OPSD_TASK_LIST is required for OPSD training.")
        selected_tasks = load_task_list(config.opsd_task_list)
        selected_roots = select_task_dataset_roots(
            config.dataset_path, selected_tasks)
        if config.rank == 0:
            logger.info(
                "  OPSD task list: %s (%d tasks, %d sub-datasets)",
                config.opsd_task_list,
                len(selected_tasks),
                len(selected_roots),
            )
        dataset_init_workers = max(1, config.dataset_init_worker)
        if config.opsd_use_safe_dataset:
            datasets = []
            for dataset_root in selected_roots:
                try:
                    datasets.append(LatentLeRobotDataset(dataset_root, config=config))
                except Exception as exc:
                    logger.warning("Skipping incomplete selected dataset %s: %s", dataset_root, exc)
        else:
            with Pool(dataset_init_workers) as pool:
                datasets = pool.map(
                    partial(LatentLeRobotDataset, config=config), selected_roots)
        if not datasets:
            raise RuntimeError("No selected OPSD datasets could be initialized.")

        train_dataset = _SelectedLatentDataset(datasets)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=config.seed,
            )
            if config.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
        )
        self.train_loader_iter = None

    def _resolve_model_paths(self, config) -> Tuple[str, str]:
        resume_path = getattr(config, "resume_from_path", None)
        resume_step = getattr(config, "resume_from_step", None)
        if resume_path is not None:
            self.step = resume_step if resume_step is not None else 0
            return _resolve_resume_model_paths(resume_path)
        if resume_step is not None:
            self.step = resume_step
            root = Path(config.output_dir) / "checkpoints" / f"step_{resume_step}"
            return _resolve_resume_model_paths(str(root))
        return _resolve_fresh_model_paths(config.student_model_path)

    def convert_batch(self, batch):
        return move_batch_to_device(batch, self.device, self.dtype)

    def split_batch(self, batch):
        return split_batch_into_chunks(batch, self.config.frame_chunk_size)

    def _save_checkpoint(self, which="online_student"):
        model = self.student if which == "online_student" else self.target_student
        try:
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                ckpt_dir = self.save_dir / f"step_{self.step}" / which / "transformer"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, ckpt_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(model.config)
                config_dict.pop("_name_or_path", None)
                with open(ckpt_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"  Saved {which} -> {ckpt_dir}")

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save {which}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def _decode_rollout_video(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            orig_device = next(self.vae.parameters()).device
            vae = self.vae.to(self.device)
            latents = latents.to(device=self.device, dtype=self.vae.dtype)
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                1, vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type="np")[0]
            if orig_device.type == "cpu":
                self.vae = self.vae.to("cpu")
            return video

    def _save_rollout_video(self, latents: torch.Tensor, step: int, batch_idx: int) -> None:
        if self.config.rank != 0:
            return
        decoded_video = self._decode_rollout_video(latents)
        output_path = self.rollout_video_dir / f"step_{step:07d}_batch_{batch_idx:04d}.mp4"
        export_to_video(decoded_video, str(output_path), fps=10)
        logger.info(f"  Saved rollout video -> {output_path}")

    def train(self):
        config = self.config
        logger.info(f"Starting Flash-WAM OPSD training for {config.max_train_steps} steps ...")
        logger.info(f"  action loss weight = {config.opsd_loss_weight}")
        logger.info(f"  EMA decay = {config.ema_decay}")

        self.optimizer.zero_grad()
        acc_losses = []
        acc_opsd_losses = []
        acc_chunks = []
        step_in_acc = 0

        progress_bar = tqdm(
            total=config.max_train_steps,
            desc="OPSD",
            disable=(config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        while self.step < config.max_train_steps:
            batch = self._get_next_batch()
            result = self._train_step_opsd(batch, step_in_acc, self.step)
            acc_losses.append(result["loss"])
            acc_opsd_losses.append(result["opsd_action_flow_loss"])
            acc_chunks.append(result["num_chunks"])
            step_in_acc += 1

            if result["should_sync"]:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), config.max_grad_norm)

                if not torch.isfinite(total_norm):
                    if config.rank == 0:
                        logger.warning(f"[step {self.step}] NaN/Inf grad norm, skipping")
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                update_ema(
                    self.target_student.parameters(),
                    self.student.parameters(),
                    rate=config.ema_decay,
                )

                lr = self.lr_scheduler.get_last_lr()[0]
                avg_loss = dist_mean(torch.stack(acc_losses).sum()).item()
                avg_opsd_loss = dist_mean(torch.stack(acc_opsd_losses).mean()).item()
                avg_chunks = dist_mean(torch.stack(acc_chunks).float().mean()).item()
                acc_losses = []
                acc_opsd_losses = []
                acc_chunks = []
                step_in_acc = 0

                torch.cuda.synchronize()
                if self.step % config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if config.rank == 0:
                    progress_bar.n = self.step + 1
                    postfix = {
                        "loss": f"{avg_opsd_loss:.4f}",
                        "norm": f"{total_norm.item():.2f}",
                        "lr": f"{lr:.2e}",
                    }
                    progress_bar.set_postfix(postfix)
                    log_dict = {
                        "loss/opsd_action_flow": avg_opsd_loss,
                        "loss/total": avg_loss,
                        "train/grad_norm": total_norm.item(),
                        "train/lr": lr,
                        "opsd/video_steps": config.opsd_video_num_inference_steps,
                        "opsd/action_steps": config.opsd_action_num_inference_steps,
                        "opsd/chunks": avg_chunks,
                    }
                    if self.wandb is not None:
                        self.wandb.log(log_dict, step=self.step)

                self.step += 1

                if self.step % config.save_interval == 0:
                    self._save_checkpoint("online_student")
                    self._save_checkpoint("target_student")

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("OPSD training completed!")
        self._save_checkpoint("online_student")
        self._save_checkpoint("target_student")
