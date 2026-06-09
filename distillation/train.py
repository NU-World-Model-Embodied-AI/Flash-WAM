"""
Flash-WAM: Modality-Aware LCM Distillation for LingBot-VA (joint video+action).

DISTILL_MODE selects the variant: flashwam (the paper's modality-aware method),
joint (naive joint LCM), video (video-only LCM), video_action_aware (video-only
LCM + action regularizer), action (action-only x0).

Design: The training pipeline is nearly identical to wan_va/train.py.
All noise addition, timestep sampling, chunk_size, window_size sampling
are reused from the native training code without modification.

The ONLY differences from native training are:
  1. Three models: Teacher (frozen), Online Student (trainable), Target Student (EMA)
  2. Teacher uses CFG (scale sampled from [2, 10]) with two forward passes
  3. Student and Target Student do NOT use CFG
  4. Loss = Huber on consistency-function output, not MSE on v-prediction
  5. k = num_train_timesteps / num_ddim_timesteps = 1000 / 25 = 40
     (the stride in the 1000-step schedule for constructing training pairs)
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from safetensors.torch import save_file

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))

from patches import install_flash_attn_stub
install_flash_attn_stub()

from distributed.fsdp import shard_model, apply_ac
from distributed.util import _configure_model, init_distributed, dist_mean
from modules.utils import load_transformer
from utils import (
    init_logger, logger,
    get_mesh_id, data_seq_to_patch, sample_timestep_id,
    warmup_constant_lambda,
    FlowMatchScheduler,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# EMA update (handles FSDP DTensor)
# ---------------------------------------------------------------------------
_ema_first_call = True


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.95):
    global _ema_first_call
    for i, (targ, src) in enumerate(zip(target_params, source_params)):
        td, sd = targ.data, src.data
        t_dt = type(td).__name__ == "DTensor"
        s_dt = type(sd).__name__ == "DTensor"

        if _ema_first_call and i < 3:
            t_shape = td._local_tensor.shape if t_dt else td.shape
            s_shape = sd._local_tensor.shape if s_dt else sd.shape
            print(f"[EMA] p#{i} t_dt={t_dt} t={t_shape}  s_dt={s_dt} s={s_shape}", flush=True)

        if t_dt and s_dt:
            td._local_tensor.lerp_(sd._local_tensor, 1.0 - rate)
        elif s_dt:
            td.lerp_(sd.full_tensor(), 1.0 - rate)
        elif t_dt:
            td._local_tensor.lerp_(sd, 1.0 - rate)
        else:
            td.lerp_(sd, 1.0 - rate)
    _ema_first_call = False


def scalings_for_boundary_conditions(sigma, sigma_data=0.5):
    """
    Consistency boundary condition: f(x, 0) = x.
    c_skip(0)=1, c_out(0)=0 → identity at sigma=0.
    """
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LCMVideoDistiller:

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
    # Data iterator — identical to wan_va/train.py
    # ==================================================================
    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        return batch

    # ==================================================================
    # _add_noise — identical to wan_va/train.py, plus returns timestep_ids
    # ==================================================================
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False,
                   action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(
            batch_size=F,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1, f_shift=0,
            action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    # ==================================================================
    # _prepare_input_dict — identical to wan_va/train.py
    # Returns (input_dict, video_timestep_ids)
    # ==================================================================
    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=self.config.noisy_cond_prob,
        )

        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict['actions_mask'],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': self.config.frame_chunk_size,
            'window_size': self.config.attn_window,
        }
        return input_dict

    def convert_input_format(self, input_dict):
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)
        return input_dict

    # ==================================================================
    # Extract video v-prediction from model output → [B, C, F, H, W]
    # ==================================================================
    def _extract_video_v(self, video_pred, ref_shape, batch_size):
        return data_seq_to_patch(
            self.patch_size, video_pred,
            ref_shape[-3], ref_shape[-2], ref_shape[-1],
            batch_size=batch_size,
        )

    # ==================================================================
    # Extract action v-prediction from model output → [B, C, F, N, 1]
    # ==================================================================
    def _extract_action_v(self, action_pred, num_frames):
        return rearrange(action_pred, 'b (f n) c -> b c f n 1', f=num_frames)

    # ==================================================================
    # Consistency function: f(x_t, t) = c_skip * x_t + c_out * pred_x0
    # ==================================================================
    def _consistency_function(self, v_pred, noisy_latent, sigma, sigma_data=None):
        """
        pred_x0 = x_t - sigma * v    (FlowMatch inversion)
        f(x_t, t) = c_skip * x_t + c_out * pred_x0
        """
        if sigma_data is None:
            sigma_data = self.config.sigma_data
        sigma_5d = sigma[None, None, :, None, None].to(v_pred.dtype).to(v_pred.device)
        c_skip, c_out = scalings_for_boundary_conditions(
            sigma_5d, sigma_data=sigma_data)
        pred_x0 = noisy_latent - sigma_5d * v_pred
        return c_skip * noisy_latent + c_out * pred_x0

    # ==================================================================
    # One training step
    # ==================================================================
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape     # [B, C, F, H, W]
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')

        # ---- 1. Prepare input_dict (identical to native training) ----
        input_dict = self._prepare_input_dict(batch)

        # ---- 2. Compute sigma_start and sigma_end for LCM (video) ----
        video_timesteps = input_dict['latent_dict']['timesteps'][0]  # [F]
        sched_ts = self.train_scheduler_latent.timesteps  # [1000]
        video_ts_ids = torch.argmin(
            (sched_ts[:, None] - video_timesteps.cpu()).abs(), dim=0)  # [F]

        sigma_start = self.train_scheduler_latent.sigmas[video_ts_ids].to(self.device)
        end_ids = (video_ts_ids + self.k).clamp(max=self.config.num_train_timesteps - 1)
        sigma_end = self.train_scheduler_latent.sigmas[end_ids].to(self.device)
        timesteps_end = self.train_scheduler_latent.timesteps[end_ids].to(self.device)

        # ---- 2b. Compute sigma pairs for actions ----
        if self.distill_action:
            action_timesteps = input_dict['action_dict']['timesteps'][0]  # [F]
            sched_ts_a = self.train_scheduler_action.timesteps
            action_ts_ids = torch.argmin(
                (sched_ts_a[:, None] - action_timesteps.cpu()).abs(), dim=0)

            sigma_start_action = self.train_scheduler_action.sigmas[action_ts_ids].to(self.device)
            end_ids_action = (action_ts_ids + self.k_action).clamp(
                max=self.config.num_train_timesteps - 1)
            sigma_end_action = self.train_scheduler_action.sigmas[end_ids_action].to(self.device)
            timesteps_end_action = self.train_scheduler_action.timesteps[end_ids_action].to(self.device)

        # ---- 3. Teacher CFG Euler step ----
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        with torch.no_grad():
            # Conditioned forward
            video_v_cond, action_v_cond = self.teacher(input_dict, train_mode=True)

            # Unconditioned forward (replace text_emb with empty)
            B_emb = input_dict['latent_dict']['text_emb'].shape[0]
            empty_emb = self.empty_emb.expand(B_emb, -1, -1)
            input_dict_uncond = {
                'latent_dict': {**input_dict['latent_dict'], 'text_emb': empty_emb},
                'action_dict': {**input_dict['action_dict'], 'text_emb': empty_emb},
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            video_v_uncond, _ = self.teacher(input_dict_uncond, train_mode=True)

            # CFG combination (video only — action_guidance_scale=1, no CFG)
            video_v_cfg = video_v_uncond + cfg_scale * (video_v_cond - video_v_uncond)

            # Video Euler step → x_prev
            video_v_cfg_5d = self._extract_video_v(video_v_cfg, ref_shape, B)
            sigma_s = sigma_start[None, None, :, None, None].to(video_v_cfg_5d)
            sigma_e = sigma_end[None, None, :, None, None].to(video_v_cfg_5d)
            x_prev = input_dict['latent_dict']['noisy_latents'] + \
                     video_v_cfg_5d * (sigma_e - sigma_s)

            # Action Euler step → x_prev_action
            if self.distill_action:
                action_v_5d = self._extract_action_v(action_v_cond, num_frames)
                sigma_s_a = sigma_start_action[None, None, :, None, None].to(action_v_5d)
                sigma_e_a = sigma_end_action[None, None, :, None, None].to(action_v_5d)
                x_prev_action = input_dict['action_dict']['noisy_latents'] + \
                                action_v_5d * (sigma_e_a - sigma_s_a)

        # ---- 4. Online student consistency prediction at sigma_start ----
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.student.set_requires_gradient_sync(False)
        else:
            self.student.set_requires_gradient_sync(True)

        student_video_v_seq, student_action_v_seq = self.student(input_dict, train_mode=True)

        # ---- 4a. Video consistency prediction ----
        if self.distill_video:
            student_video_v = self._extract_video_v(student_video_v_seq, ref_shape, B)
            student_video_pred = self._consistency_function(
                student_video_v,
                input_dict['latent_dict']['noisy_latents'],
                sigma_start,
            )

        # ---- 4b. Action prediction ----
        if self.distill_action:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            if self.action_distill_mode == "x0":
                # Direct x_0 prediction: x_0 = x_σ - σ · v
                sigma_s_a_5d = sigma_start_action[None, None, :, None, None].to(student_action_v)
                student_action_pred = input_dict['action_dict']['noisy_latents'] - \
                                      sigma_s_a_5d * student_action_v
            else:
                student_action_pred = self._consistency_function(
                    student_action_v,
                    input_dict['action_dict']['noisy_latents'],
                    sigma_start_action,
                )

        # ---- 5. Target student prediction at sigma_end on x_prev ----
        input_dict_end = {
            'latent_dict': {
                **input_dict['latent_dict'],
                'noisy_latents': x_prev.detach(),
                'timesteps': timesteps_end[None].repeat(B, 1),
            },
            'action_dict': {**input_dict['action_dict']},
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if self.distill_action:
            input_dict_end['action_dict'] = {
                **input_dict['action_dict'],
                'noisy_latents': x_prev_action.detach(),
                'timesteps': timesteps_end_action[None].repeat(B, 1),
            }

        with torch.no_grad():
            target_video_v_seq, target_action_v_seq = self.target_student(
                input_dict_end, train_mode=True)

            if self.distill_video:
                target_video_v = self._extract_video_v(target_video_v_seq, ref_shape, B)
                target_video_pred = self._consistency_function(
                    target_video_v, x_prev, sigma_end,
                )

            if self.distill_action:
                target_action_v = self._extract_action_v(target_action_v_seq, num_frames)
                if self.action_distill_mode == "x0":
                    sigma_e_a_5d = sigma_end_action[None, None, :, None, None].to(target_action_v)
                    target_action_pred = x_prev_action - sigma_e_a_5d * target_action_v
                else:
                    target_action_pred = self._consistency_function(
                        target_action_v, x_prev_action, sigma_end_action,
                    )

        # ---- 6. Loss ----
        video_loss = torch.tensor(0.0, device=self.device)
        if self.distill_video:
            if self.config.loss_type == "huber":
                c = self.config.huber_c
                video_diff = student_video_pred.float() - target_video_pred.detach().float()
                video_loss = torch.mean(torch.sqrt(video_diff ** 2 + c ** 2) - c)
            else:
                video_loss = F.mse_loss(
                    student_video_pred.float(), target_video_pred.detach().float())

        action_loss = torch.tensor(0.0, device=self.device)
        if self.distill_action:
            mask = actions_mask.float()
            if self.config.loss_type == "huber":
                c = self.config.huber_c
                action_diff = (student_action_pred.float() * mask) - \
                              (target_action_pred.detach().float() * mask)
                action_loss = (torch.sqrt(action_diff ** 2 + c ** 2) - c).sum() / \
                              mask.sum().clamp(min=1)
            else:
                action_diff = (student_action_pred.float() * mask) - \
                              (target_action_pred.detach().float() * mask)
                action_loss = (action_diff ** 2).sum() / mask.sum().clamp(min=1)

        # ---- 6b. Action-aware regularizer (native flow matching MSE) ----
        action_aware_loss = torch.tensor(0.0, device=self.device)
        if self.action_aware:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            action_targets = input_dict['action_dict']['targets']
            mask = actions_mask.float()
            aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
            action_aware_loss = (aa_diff ** 2).sum() / mask.sum().clamp(min=1)

        loss = video_loss + self.config.action_loss_weight * action_loss \
               + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss

        loss = loss / self.gradient_accumulation_steps

        if not torch.isfinite(loss):
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            loss = torch.zeros(1, device=self.device, requires_grad=True)

        loss.backward()

        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach() if self.distill_action else action_loss,
            "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
            "should_sync": should_sync,
        }

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(args):
    # Pick a config module via CONFIG_FILE env var so per-task configs
    # (config_g1_alltasks.py, etc.) can coexist with the canonical
    # robotwin config.py without ever being overwritten.
    # Default preserves the original behaviour.
    import importlib
    _config_mod = os.environ.get("CONFIG_FILE", "config")
    config = importlib.import_module(_config_mod).cfg

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.teacher_model_path is not None:
        config.teacher_model_path = args.teacher_model_path
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step
    if args.resume_from_path is not None:
        config.resume_from_path = args.resume_from_path

    if rank == 0:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Teacher: {config.teacher_model_path}")
        logger.info(f"Dataset: {config.dataset_path}")
        logger.info(f"Output:  {config.output_dir}")

    trainer = LCMVideoDistiller(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="LCM Video Distillation v2")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--resume-from-step", type=int, default=None,
                        help="Resume training from this checkpoint step")
    parser.add_argument("--resume-from-path", type=str, default=None,
                        help="Resume from explicit checkpoint directory (overrides --resume-from-step path)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
