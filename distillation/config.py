"""
Config for Flash-WAM: modality-aware LCM distillation of LingBot-VA (v2).

Key design: 25-step teacher schedule is just 25 anchor points sub-sampled
from the original 1000-step FlowMatch schedule. All noise addition and
timestep embedding use the native 1000-step system.
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: LCM Video Distillation v2")

# ============================================================
# Paths
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH", os.path.join(_project_root, "checkpoints", "lingbot-va-posttrain-robotwin"))
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR", os.path.join(_this_dir, "output"))
cfg.dataset_path = os.environ.get(
    "DATASET_PATH", os.path.join(_project_root, "training_data", "lerobot_robotwin_eef_aug_500"))
cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# Model Architecture (from va_robotwin_cfg)
# ============================================================
cfg.patch_size = (1, 2, 2)
cfg.param_dtype = torch.bfloat16
cfg.env_type = "robotwin_tshape"
cfg.height = 256
cfg.width = 320
cfg.action_dim = 30
cfg.action_per_frame = 16
cfg.frame_chunk_size = 2
cfg.attn_window = 72

cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
_inv = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inv[_j] = _i
cfg.inverse_used_action_channel_ids = _inv

cfg.action_norm_method = "quantiles"
cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1,
    ] + [0.0] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1,
    ] + [0.0] * 14 + [1.0, 1.0],
}

# ============================================================
# FlowMatch Scheduler (same as native training)
# ============================================================
cfg.snr_shift = 5.0
cfg.action_snr_shift = 1.0
cfg.num_train_timesteps = 1000

# ============================================================
# LCM Distillation
# ============================================================
cfg.num_ddim_timesteps = 2        # 2 anchor points → k=500 stride (target: 2-step generation)

# Distillation mode: "flashwam" | "joint" | "video" | "video_action_aware" | "action"
#   flashwam           — Flash-WAM (the paper's method): modality-aware joint
#                        distillation = video consistency + action consistency
#                        + action MSE regularizer
#   joint              — naive joint LCM: video consistency + action consistency (ablation)
#   video              — video-only LCM consistency loss (ablation)
#   video_action_aware — video-only LCM + small action MSE regularizer (ablation)
#   action             — action-only consistency, student init = video-LCM checkpoint
cfg.distill_mode = os.environ.get("DISTILL_MODE", "flashwam")

_mode = cfg.distill_mode
cfg.distill_video = _mode in ("video", "joint", "video_action_aware", "flashwam")
cfg.distill_action = _mode in ("action", "joint", "flashwam")
cfg.action_aware = _mode in ("video_action_aware", "flashwam")
cfg.num_ddim_timesteps_action = 2        # k_action = 1000/2 = 500
cfg.action_loss_weight = 1.0
cfg.action_distill_mode = "x0"          # action consistency function parametrization
cfg.action_aware_weight = 0.01          # small weight for action-aware regularizer

# ============================================================
# LCM Hyperparameters
# ============================================================
cfg.ema_decay = 0.995
cfg.loss_type = "huber"           # "l2" or "huber"
cfg.huber_c = 0.001
cfg.sigma_data = 0.5              # for boundary condition scaling
cfg.cfg_min = 2.0                 # teacher CFG scale range
cfg.cfg_max = 10.0

# ============================================================
# Training
# ============================================================
cfg.learning_rate = 5e-6
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = 0.0
cfg.max_grad_norm = 2.0
cfg.warmup_steps = 100
cfg.max_train_steps = 10000
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 8
cfg.load_worker = 0
cfg.noisy_cond_prob = 0.0         # no noisy condition augmentation during distillation
cfg.cfg_prob = 0.0                # no random CFG dropout — teacher handles CFG explicitly

# ============================================================
# Checkpointing & Logging
# ============================================================
cfg.save_interval = 1000
cfg.gc_interval = 50
cfg.enable_wandb = True
cfg.wandb_entity = None
cfg.seed = 42
