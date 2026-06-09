"""Consistency / loss training step (StepMixin)."""
import torch
import torch.nn.functional as F
from einops import rearrange

from utils import data_seq_to_patch, logger
from consistency import scalings_for_boundary_conditions


class StepMixin:
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
                # Action consistency function: f = x_σ - σ · v
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
