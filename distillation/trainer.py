"""FlashWAMDistiller: model/optimizer/dataset setup, training loop, checkpointing."""
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from safetensors.torch import save_file

from distributed.fsdp import shard_model, apply_ac
from distributed.util import _configure_model, dist_mean
from modules.utils import load_transformer
from utils import logger, warmup_constant_lambda, FlowMatchScheduler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data import DataMixin
from step import StepMixin
from ema import update_ema


class FlashWAMDistiller(DataMixin, StepMixin):

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        # k: stride in 1000-step schedule (1000 / 25 = 40)
        self.k = config.num_train_timesteps // config.num_ddim_timesteps
        self.distill_video = getattr(config, 'distill_video', True)
        self.distill_action = getattr(config, 'distill_action', False)
        self.action_distill_mode = getattr(config, 'action_distill_mode', 'consistency')
        self.action_aware = getattr(config, 'action_aware', False)
        self.k_action = config.num_train_timesteps // getattr(
            config, 'num_ddim_timesteps_action', config.num_ddim_timesteps)

        # WandB
        if config.enable_wandb and HAS_WANDB and config.rank == 0:
            wandb.init(
                project="lcm_video_distill_lingbot_va",
                entity=getattr(config, "wandb_entity", None),
                config=dict(config),
            )

        # ==============================================================
        # Schedulers — identical to wan_va/train.py
        # ==============================================================
        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(config.num_train_timesteps, training=True)

        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(config.num_train_timesteps, training=True)

        # Empty text embedding for CFG unconditional pass
        self.empty_emb = torch.load(config.empty_emb_path, map_location="cpu").to(self.device)

        if config.rank == 0:
            logger.info(f"LCM stride k = {self.k} "
                        f"(num_train_timesteps={config.num_train_timesteps}, "
                        f"num_ddim_timesteps={config.num_ddim_timesteps})")
            logger.info(f"Distill video: {self.distill_video}")
            logger.info(f"Distill action: {self.distill_action}")
            logger.info(f"Action aware: {self.action_aware}")
            if self.distill_action:
                logger.info(f"  k_action = {self.k_action}, "
                            f"action_loss_weight = {config.action_loss_weight}, "
                            f"mode = {self.action_distill_mode}")
            if self.action_aware:
                logger.info(f"  action_aware_weight = {config.action_aware_weight}")
            logger.info(f"Empty embedding shape: {self.empty_emb.shape}")

        # ==============================================================
        # Three models
        # ==============================================================
        teacher_path = os.path.join(config.teacher_model_path, "transformer")

        logger.info("Loading teacher (frozen) ...")
        self.teacher = load_transformer(teacher_path, torch_dtype=self.dtype, torch_device="cpu")
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.teacher = self.teacher.to(self.dtype)
        self.teacher = _configure_model(
            model=self.teacher, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=True,
        )

        # Determine student init path (resume from checkpoint or start from teacher)
        resume_path = getattr(config, "resume_from_path", None)
        resume_step = getattr(config, "resume_from_step", None)
        if resume_path is not None:
            student_path = os.path.join(resume_path, "online_student", "transformer")
            target_path = os.path.join(resume_path, "target_student", "transformer")
            self.step = resume_step if resume_step is not None else 0
            if config.rank == 0:
                logger.info(f"Resuming from path: {resume_path}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
                logger.info(f"  Starting step: {self.step}")
        elif resume_step is not None:
            student_path = os.path.join(
                config.output_dir, "checkpoints", f"step_{resume_step}",
                "online_student", "transformer")
            target_path = os.path.join(
                config.output_dir, "checkpoints", f"step_{resume_step}",
                "target_student", "transformer")
            self.step = resume_step
            if config.rank == 0:
                logger.info(f"Resuming from step {resume_step}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
        else:
            student_path = teacher_path
            target_path = teacher_path

        logger.info("Loading online student (trainable) ...")
        self.student = load_transformer(student_path, torch_dtype=torch.float32, torch_device="cpu")
        apply_ac(self.student)
        self.student = self.student.to(self.dtype)
        self.student = _configure_model(
            model=self.student, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=False,
        )
        self.student.train()
        self.student.requires_grad_(True)

        logger.info("Loading target student (EMA, frozen) ...")
        self.target_student = load_transformer(target_path, torch_dtype=self.dtype, torch_device="cpu")
        self.target_student = self.target_student.to(self.dtype)
        self.target_student = _configure_model(
            model=self.target_student, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=False,
        )
        self.target_student.requires_grad_(False)
        self.target_student.eval()

        # ==============================================================
        # Optimizer & LR scheduler
        # ==============================================================
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
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )
        # Fast-forward LR scheduler to the resume step
        if self.step > 0:
            for _ in range(self.step):
                self.lr_scheduler.step()
            if config.rank == 0:
                logger.info(f"LR scheduler fast-forwarded to step {self.step}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}")

        # ==============================================================
        # Dataset — same as native
        # ==============================================================
        logger.info("Loading dataset ...")
        from patches import SafeMultiLatentLeRobotDataset as MultiLatentLeRobotDataset
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=config.world_size,
                               rank=config.rank, shuffle=True, seed=config.seed)
            if config.world_size > 1 else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
        )
        self.save_dir = Path(config.output_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader_iter = None

    # ==================================================================
    # Save checkpoint
    # ==================================================================
    def _save_checkpoint(self, which="online_student"):
        model = self.student if which == "online_student" else self.target_student
        try:
            state_dict = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                ckpt_dir = self.save_dir / f"step_{self.step}" / which / "transformer"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, ckpt_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(model.config)
                config_dict.pop("_name_or_path", None)
                with open(ckpt_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"  Saved {which} → {ckpt_dir}")

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save {which}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    # ==================================================================
    # Main training loop
    # ==================================================================
    def train(self):
        config = self.config
        mode = []
        if self.distill_video:
            mode.append("video")
        if self.distill_action:
            mode.append("action")
        if self.action_aware:
            mode.append("action_aware")
        mode_str = "+".join(mode) if mode else "none"
        logger.info(f"Starting LCM {mode_str} distillation for {config.max_train_steps} steps ...")
        if self.distill_video:
            logger.info(f"  k = {self.k} ({config.num_train_timesteps} / {config.num_ddim_timesteps})")
        if self.distill_action:
            logger.info(f"  k_action = {self.k_action} "
                        f"({config.num_train_timesteps} / {config.num_ddim_timesteps_action})")
            logger.info(f"  action_loss_weight = {config.action_loss_weight}")
            logger.info(f"  action_distill_mode = {self.action_distill_mode}")
        logger.info(f"  Teacher CFG: [{config.cfg_min}, {config.cfg_max}]")
        logger.info(f"  EMA decay: {config.ema_decay}")
        logger.info(f"  Loss: {config.loss_type}")

        self.optimizer.zero_grad()
        acc_losses = []
        acc_video_losses = []
        acc_action_losses = []
        acc_action_aware_losses = []
        step_in_acc = 0

        progress_bar = tqdm(
            total=config.max_train_steps,
            desc="Distill", disable=(config.rank != 0),
            leave=True, dynamic_ncols=True, initial=self.step,
        )

        while self.step < config.max_train_steps:
            batch = self._get_next_batch()
            result = self._train_step(batch, step_in_acc)
            acc_losses.append(result["loss"])
            acc_video_losses.append(result["video_loss"])
            acc_action_losses.append(result["action_loss"])
            acc_action_aware_losses.append(result["action_aware_loss"])
            step_in_acc += 1

            if result["should_sync"]:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), config.max_grad_norm)

                if not torch.isfinite(total_norm):
                    if config.rank == 0:
                        logger.warning(f"[step {self.step}] NaN grad norm, skipping")
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
                avg_video_loss = dist_mean(torch.stack(acc_video_losses).sum()).item()
                avg_action_loss = dist_mean(torch.stack(acc_action_losses).sum()).item()
                avg_action_aware_loss = dist_mean(torch.stack(acc_action_aware_losses).sum()).item()
                acc_losses = []
                acc_video_losses = []
                acc_action_losses = []
                acc_action_aware_losses = []
                step_in_acc = 0

                torch.cuda.synchronize()
                if self.step % config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if config.rank == 0:
                    progress_bar.n = self.step + 1
                    postfix = {
                        "loss": f"{avg_loss:.4f}",
                        "norm": f"{total_norm.item():.2f}",
                        "lr": f"{lr:.2e}",
                    }
                    log_dict = {
                        "loss/total": avg_loss,
                        "train/grad_norm": total_norm.item(),
                        "train/lr": lr,
                    }
                    if self.distill_video:
                        postfix["v"] = f"{avg_video_loss:.4f}"
                        log_dict["loss/video_consistency"] = avg_video_loss
                    if self.distill_action:
                        postfix["a"] = f"{avg_action_loss:.4f}"
                        log_dict["loss/action_consistency"] = avg_action_loss
                    if self.action_aware:
                        postfix["aa"] = f"{avg_action_aware_loss:.4f}"
                        log_dict["loss/action_aware"] = avg_action_aware_loss
                    progress_bar.set_postfix(postfix)
                    if config.enable_wandb and HAS_WANDB:
                        wandb.log(log_dict, step=self.step)

                self.step += 1

                if self.step % config.save_interval == 0:
                    self._save_checkpoint("online_student")
                    self._save_checkpoint("target_student")

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Distillation completed!")
        self._save_checkpoint("online_student")
        self._save_checkpoint("target_student")
