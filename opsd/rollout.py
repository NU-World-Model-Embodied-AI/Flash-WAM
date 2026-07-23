"""KV-cache rollout and OPSD loss for Flash-WAM stage-2 training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from wan_va.utils import logger
import torch.nn.functional as F
from einops import rearrange

from wan_va.utils import data_seq_to_patch, get_mesh_id


def _call_model_method(model, name: str, *args, **kwargs):
    """Call a custom method on either a plain module or a wrapped module."""
    if hasattr(model, name):
        return getattr(model, name)(*args, **kwargs)
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, name):
        return getattr(module, name)(*args, **kwargs)
    raise AttributeError(f"Model does not expose method {name!r}.")


def _make_step_timesteps(scheduler, num_steps: int) -> torch.Tensor:
    if num_steps not in {1, 2}:
        raise ValueError("OPSD rollout currently supports 1 or 2 denoising steps.")
    scheduler.set_timesteps(num_steps)
    return F.pad(scheduler.timesteps, (0, 1), mode="constant", value=0)


def _expand_timesteps(timestep, batch_size: int, num_frames: int, device) -> torch.Tensor:
    return torch.ones(
        [batch_size, num_frames],
        dtype=torch.float32,
        device=device,
    ) * float(timestep)


def _video_flow_from_seq(video_seq: torch.Tensor, ref: torch.Tensor, patch_size) -> torch.Tensor:
    return data_seq_to_patch(
        patch_size,
        video_seq,
        ref.shape[-3],
        ref.shape[-2],
        ref.shape[-1],
        batch_size=ref.shape[0],
    )


def _action_flow_from_seq(action_seq: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return rearrange(action_seq, "b (f n) c -> b c f n 1", f=ref.shape[2])


@dataclass
class OpsdCacheManager:
    # Inputs:
    #   model: WanTransformer3DModel or FSDP-wrapped equivalent
    #   cache_name: attention cache namespace
    #   config: OPSD config with frame_chunk_size, patch_size, attn_window
    model: object
    cache_name: str
    config: object

    def clear_cache(self) -> None:
        _call_model_method(self.model, "clear_cache", self.cache_name)

    def clear_pred_cache(self) -> None:
        _call_model_method(self.model, "clear_pred_cache", self.cache_name)

    def create_empty_cache(self, batch: dict) -> None:
        # Inputs:
        #   batch["latents"]: Tensor[B,C,F,H,W] for shape inference
        #   batch["actions"]: Tensor[B,C,F,N,1] for shape inference
        # Outputs:
        #   None; initializes attention caches in the wrapped model
        latents = batch["latents"]
        actions = batch["actions"]
        patch_f, patch_h, patch_w = self.config.patch_size
        latent_token_per_chunk = (
            self.config.frame_chunk_size
            * latents.shape[-2]
            * latents.shape[-1]
        ) // (patch_f * patch_h * patch_w) # 240 = 2 * 240 * 20 // (2 * 2) 
        action_token_per_chunk = self.config.frame_chunk_size * actions.shape[-2] # 32 = 2 * 16
        _call_model_method(
            self.model,
            "clear_cache",
            self.cache_name,
        )
        _call_model_method(
            self.model,
            "create_empty_cache",
            self.cache_name,
            self.config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,  # use with `attn_window ` to get total token 9792 for kv init
            latents.device,
            latents.dtype,
            latents.shape[0],
        )  

    def video_grid(self, video: torch.Tensor, frame_start: int) -> torch.Tensor:
        patch_f, patch_h, patch_w = self.config.patch_size
        grid = get_mesh_id(
            video.shape[-3] // patch_f,
            video.shape[-2] // patch_h,
            video.shape[-1] // patch_w,
            0,
            1,
            frame_start,
        ).to(video.device)
        return grid[None].repeat(video.shape[0], 1, 1)

    def action_grid(self, action: torch.Tensor, frame_start: int) -> torch.Tensor:
        grid = get_mesh_id(
            action.shape[-3],
            action.shape[-2],
            action.shape[-1],
            1,
            1,
            frame_start,
            action=True,
        ).to(action.device)
        return grid[None].repeat(action.shape[0], 1, 1)

    def make_video_input(
        self,
        video: torch.Tensor,
        text_emb: torch.Tensor,
        timestep,
        frame_start: int,
    ) -> dict:
        return {
            "noisy_latents": video,
            "timesteps": _expand_timesteps(
                timestep, video.shape[0], video.shape[2], video.device), # [B, F]
            "grid_id": self.video_grid(video, frame_start),
            "text_emb": text_emb,
        }

    def make_action_input(
        self,
        action: torch.Tensor,
        text_emb: torch.Tensor,
        timestep,
        frame_start: int,
    ) -> dict:
        return {
            "noisy_latents": action,
            "timesteps": _expand_timesteps(
                timestep, action.shape[0], action.shape[2], action.device),
            "grid_id": self.action_grid(action, frame_start),
            "text_emb": text_emb,
        }

    @torch.no_grad()
    def write_video_to_cache(
        self,
        video: torch.Tensor,
        text_emb: torch.Tensor,
        frame_start: int,
        *,
        update_cache: int = 2,
    ) -> None:
        input_dict = self.make_video_input(video, text_emb, 0, frame_start)
        self.model(
            input_dict,
            update_cache=update_cache,
            cache_name=self.cache_name,
            action_mode=False,
        )

    @torch.no_grad()
    def write_action_to_cache(
        self,
        action: torch.Tensor,
        text_emb: torch.Tensor,
        frame_start: int,
        *,
        update_cache: int = 2,
    ) -> None:
        input_dict = self.make_action_input(action, text_emb, 0, frame_start)
        self.model(
            input_dict,
            update_cache=update_cache,
            cache_name=self.cache_name,
            action_mode=True,
        )

    @torch.no_grad()
    def write_gt_chunk_to_cache(self, chunk: dict) -> None:
        self.write_video_to_cache(
            chunk["video"], chunk["text_emb"], chunk["frame_start"], update_cache=2)
        self.write_action_to_cache(
            chunk["action"], chunk["text_emb"], chunk["frame_start"], update_cache=2)


def sample_video_chunk(
    model,
    cache: OpsdCacheManager,
    noise: torch.Tensor,
    *,
    text_emb: torch.Tensor,
    frame_start: int,
    scheduler,
    num_steps: int,
    config,
) -> torch.Tensor:
    # Inputs:
    #   noise: Tensor[B,C,F,H,W] initial video noise
    # Outputs:
    #   Tensor[B,C,F,H,W]: final predicted video latent written to pred cache
    timesteps = _make_step_timesteps(scheduler, num_steps) # num_steps = 2 -> [1000, 833, 0]
    video = noise
    with torch.no_grad():
        for step_idx, timestep in enumerate(timesteps):
            last_step = step_idx == len(timesteps) - 1
            input_dict = cache.make_video_input(video, text_emb, timestep, frame_start)
            video_flow_seq = model(
                input_dict,
                update_cache=1 if last_step else 0,
                cache_name=cache.cache_name,
                action_mode=False,
            ) # [1, 960, 48]
            if not last_step:
                video_flow = _video_flow_from_seq(video_flow_seq, video, config.patch_size) # [1, 48, 2, 24, 20]
                video = scheduler.step(video_flow, timestep, video, return_dict=False)
    return video.detach()


def sample_student_action_trajectory(
    model,
    cache: OpsdCacheManager,
    noise: torch.Tensor,
    *,
    text_emb: torch.Tensor,
    frame_start: int,
    scheduler,
    num_steps: int,
    action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    # Inputs:
    #   noise: Tensor[B,C,F,N,1] shared initial action noise
    # Outputs:
    #   final_action: Tensor[B,C,F,N,1]
    #   flows: list[Tensor[B,C,F,N,1]] with gradients
    #   states: list[Tensor[B,C,F,N,1]] student-visited action states
    #   final_cache_timestep: scalar terminal timestep for cache write
    timesteps = _make_step_timesteps(scheduler, num_steps)
    action = noise * action_mask.to(noise.dtype) # [1, 30, 2, 16, 1] 其中 动作的第2 维度的30个元素中前 14 个元素和最后2个元素是 noise 其他的都是0
    flows = []
    states = []

    for step_idx, timestep in enumerate(timesteps[:-1]):
        states.append(action.detach()) # 每一步 forward 前使用 action.detach()，第二步去噪不会回传到第一步 
        input_dict = cache.make_action_input(action.detach(), text_emb, timestep, frame_start)
        action_flow_seq = model(
            input_dict,
            update_cache=0,
            cache_name=cache.cache_name,
            action_mode=True,
        ) # [1, 32, 30] = [B, T, D]
        action_flow = _action_flow_from_seq(action_flow_seq, action) # [1, 30, 2, 16, 1]
        flows.append(action_flow)
        with torch.no_grad():
            action = scheduler.step(
                action_flow.detach(), timestep, action.detach(), return_dict=False)
            action = action * action_mask.to(action.dtype)

    return action.detach(), flows, states, timesteps[-1]


def evaluate_teacher_action_flows_on_states(
    model,
    cache: OpsdCacheManager,
    states: Sequence[torch.Tensor],
    *,
    text_emb: torch.Tensor,
    frame_start: int,
    scheduler,
    num_steps: int,
) -> List[torch.Tensor]:
    # Inputs:
    #   states: student-visited action states before each denoising update
    # Outputs:
    #   list[Tensor[B,C,F,N,1]] detached EMA teacher action flows
    timesteps = _make_step_timesteps(scheduler, num_steps)
    if len(states) != len(timesteps) - 1:
        raise ValueError("Teacher states must match the number of denoising steps.")

    flows = []
    with torch.no_grad():
        for action, timestep in zip(states, timesteps[:-1]):
            input_dict = cache.make_action_input(action, text_emb, timestep, frame_start)
            action_flow_seq = model(
                input_dict,
                update_cache=0,
                cache_name=cache.cache_name,
                action_mode=True,
            )
            flows.append(_action_flow_from_seq(action_flow_seq, action).detach())
    return flows


def cache_final_student_action(
    model,
    cache: OpsdCacheManager,
    action: torch.Tensor,
    *,
    text_emb: torch.Tensor,
    frame_start: int,
    terminal_timestep,
) -> None:
    with torch.no_grad():
        input_dict = cache.make_action_input(
            action, text_emb, terminal_timestep, frame_start)
        model(
            input_dict,
            update_cache=1,
            cache_name=cache.cache_name,
            action_mode=True,
        )


def masked_flow_mse(
    student_flows: Sequence[torch.Tensor],
    teacher_flows: Sequence[torch.Tensor],
    action_mask: torch.Tensor,
) -> torch.Tensor:
    # Inputs:
    #   student_flows: list of Tensor[B,C,F,N,1], requires grad
    #   teacher_flows: list of Tensor[B,C,F,N,1], detached
    #   action_mask: BoolTensor[B,C,F,N,1]
    # Outputs:
    #   scalar Tensor: mean masked flow MSE over denoising steps
    if len(student_flows) != len(teacher_flows):
        raise ValueError("student_flows and teacher_flows must have the same length.")
    if not student_flows:
        raise ValueError("At least one flow pair is required.")

    mask = action_mask.float()
    losses = []
    denom = mask.sum().clamp(min=1)
    for student_flow, teacher_flow in zip(student_flows, teacher_flows):
        diff = (student_flow.float() - teacher_flow.detach().float()) * mask.float()
        losses.append((diff ** 2).sum() / denom)
    return torch.stack(losses).mean()


class OPSDRolloutMixin:
    """Mixin implementing one OPSD training microstep."""

    def _train_step_opsd(self, batch: dict, batch_idx: int, global_step: int) -> dict:
        batch = self.convert_batch(batch) # ['latents'(1, 48, 16, 24, 20), 'text_emb'(1, 512, 4096), 'actions'(1, 30, 16, 16, 1), 'actions_mask'(1, 30, 16, 16, 1)] # 30维中的14维是true
        chunks = self.split_batch(batch)  # List of 8 * ['video'(1, 48, 2, 24, 20), 'action', 'action_mask', 'frame_start'(0->2->4), 'text_emb']
        if not chunks:
            raise RuntimeError("OPSD batch produced no chunks.")

        local_chunk_count = len(chunks)
        if torch.distributed.is_initialized():
            shared_chunk_count = torch.tensor(
                local_chunk_count, device=self.device, dtype=torch.int64)
            torch.distributed.all_reduce(
                shared_chunk_count, op=torch.distributed.ReduceOp.MIN)
            chunks = chunks[:shared_chunk_count.item()]
        if not chunks:
            raise RuntimeError("No common OPSD chunks across distributed ranks.")
        if self.config.rank == 0 and len(chunks) != local_chunk_count:
            logger.info(
                "OPSD chunk count aligned across ranks: local=%d shared=%d",
                local_chunk_count,
                len(chunks),
            )

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(self.student, "set_requires_gradient_sync"):
            self.student.set_requires_gradient_sync(should_sync)

        self.student.eval()
        self.target_student.eval()

        student_cache = OpsdCacheManager(
            self.student, self.config.opsd_cache_name, self.config)
        teacher_cache = OpsdCacheManager(
            self.target_student, self.config.opsd_cache_name, self.config)
        student_cache.create_empty_cache(batch)
        teacher_cache.create_empty_cache(batch)

        loss_terms = []
        logged_loss = torch.tensor(0.0, device=self.device)
        supervised_chunks = 0
        save_rollout_video = (
            self.config.rank == 0
            and self.config.opsd_rollout_video_interval > 0
            and (global_step + 1) % self.config.opsd_rollout_video_interval == 0
        )
        rollout_video_chunks = [] if save_rollout_video else None

        start_idx = 0
        if self.config.opsd_skip_first_chunk_loss and len(chunks) > 1:
            with torch.no_grad():
                student_cache.write_gt_chunk_to_cache(chunks[0])
                teacher_cache.write_gt_chunk_to_cache(chunks[0])
            if rollout_video_chunks is not None:
                rollout_video_chunks.append(chunks[0]["video"].detach().cpu())
            start_idx = 1

        for chunk_idx in range(start_idx, len(chunks)):
            chunk = chunks[chunk_idx]
            video_noise = torch.randn_like(chunk["video"])
            action_noise = torch.randn_like(chunk["action"])

            student_video = sample_video_chunk(
                self.student,
                student_cache,
                video_noise,
                text_emb=chunk["text_emb"],
                frame_start=chunk["frame_start"],
                scheduler=self.video_scheduler,
                num_steps=self.config.opsd_video_num_inference_steps,
                config=self.config,
            )
            if rollout_video_chunks is not None:
                rollout_video_chunks.append(student_video.detach().cpu())

            if not self.config.opsd_pred_video_detach_for_action:
                raise NotImplementedError(
                    "The current Wan KV-cache stores detached K/V tensors, so "
                    "OPSD_PRED_VIDEO_DETACH_FOR_ACTION=0 is not supported.")
            del student_video

            teacher_cache.write_video_to_cache(
                chunk["video"], chunk["text_emb"], chunk["frame_start"], update_cache=2)

            student_action, student_flows, student_states, terminal_timestep = (
                sample_student_action_trajectory(
                    self.student,
                    student_cache,
                    action_noise,
                    text_emb=chunk["text_emb"],
                    frame_start=chunk["frame_start"],
                    scheduler=self.action_scheduler,
                    num_steps=self.config.opsd_action_num_inference_steps,
                    action_mask=chunk["action_mask"],
                )
            )
            teacher_flows = evaluate_teacher_action_flows_on_states(
                self.target_student,
                teacher_cache,
                student_states,
                text_emb=chunk["text_emb"],
                frame_start=chunk["frame_start"],
                scheduler=self.action_scheduler,
                num_steps=self.config.opsd_action_num_inference_steps,
            )

            chunk_loss = masked_flow_mse(
                student_flows, teacher_flows, chunk["action_mask"])
            if not torch.isfinite(chunk_loss):
                chunk_loss = torch.zeros(1, device=self.device, requires_grad=True).sum()

            if self.config.opsd_backward_per_chunk:
                num_loss_chunks = max(len(chunks) - start_idx, 1)
                (
                    chunk_loss
                    * self.config.opsd_loss_weight
                    / float(num_loss_chunks)
                    / float(self.gradient_accumulation_steps)
                ).backward()
            else:
                loss_terms.append(chunk_loss)
            logged_loss = logged_loss + chunk_loss.detach()
            supervised_chunks += 1

            student_cache.clear_pred_cache()
            teacher_cache.clear_pred_cache()
            student_cache.write_gt_chunk_to_cache(chunk)
            teacher_cache.write_action_to_cache(
                chunk["action"], chunk["text_emb"], chunk["frame_start"], update_cache=2)

        if supervised_chunks == 0:
            raise RuntimeError("No OPSD loss terms were accumulated.")

        opsd_loss = logged_loss / float(supervised_chunks) * self.config.opsd_loss_weight
        loss = opsd_loss / self.gradient_accumulation_steps

        if not self.config.opsd_backward_per_chunk:
            opsd_loss = torch.stack(loss_terms).mean() * self.config.opsd_loss_weight
            loss = opsd_loss / self.gradient_accumulation_steps

            if not torch.isfinite(loss):
                loss = torch.zeros(1, device=self.device, requires_grad=True)

            loss.backward()

        if rollout_video_chunks:
            rollout_latents = torch.cat(rollout_video_chunks, dim=2)
            self._save_rollout_video(rollout_latents, global_step + 1, batch_idx)

        return {
            "loss": loss.detach(),
            "opsd_action_flow_loss": (logged_loss / max(supervised_chunks, 1)).detach(),
            "should_sync": should_sync,
            "num_chunks": torch.tensor(len(chunks), device=self.device),
        }
