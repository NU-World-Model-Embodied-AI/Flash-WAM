"""Config for Flash-WAM OPSD stage-2 training.

This config intentionally keeps OPSD separate from the first-stage
consistency distillation config. Shared architecture/data defaults are copied
from distillation.config, while the active loss and rollout controls are
defined here.
"""
import copy
import os

from distillation.config import cfg as _base_cfg


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


cfg = copy.deepcopy(_base_cfg)

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

cfg.__name__ = "Config: Flash-WAM OPSD Stage-2"
cfg.stage = "opsd"

# Paths. STUDENT_PATH is the stage-1 Flash-WAM checkpoint root. For convenience,
# TEACHER_PATH remains accepted as an alias because distillation/run.sh used it.
cfg.student_model_path = os.environ.get(
    "STUDENT_PATH",
    os.environ.get("TEACHER_PATH", cfg.teacher_model_path),
)
cfg.output_dir = os.environ.get("OUTPUT_DIR", os.path.join(_this_dir, "output"))
cfg.dataset_path = os.environ.get(
    "DATASET_PATH",
    os.path.join(_project_root, "training_data", "lerobot_robotwin_eef_aug_500"),
)
cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
cfg.opsd_task_list = os.environ.get("OPSD_TASK_LIST")

# OPSD rollout. v1 supports the requested 1/1, 1/2, and 2/2 settings.
cfg.opsd_video_num_inference_steps = _env_int("OPSD_VIDEO_STEPS", 2)
cfg.opsd_action_num_inference_steps = _env_int("OPSD_ACTION_STEPS", 2)
_valid_step_pairs = {(1, 1), (1, 2), (2, 2)}
if (
    cfg.opsd_video_num_inference_steps,
    cfg.opsd_action_num_inference_steps,
) not in _valid_step_pairs:
    raise ValueError(
        "OPSD step pair must be one of (1,1), (1,2), or (2,2).")

cfg.opsd_loss_type = "flow"
cfg.opsd_loss_step_mode = "all"
cfg.opsd_cache_name = os.environ.get("OPSD_CACHE_NAME", "opsd")
cfg.opsd_pred_video_detach_for_action = _env_bool(
    "OPSD_PRED_VIDEO_DETACH_FOR_ACTION", True)
cfg.opsd_skip_first_chunk_loss = _env_bool("OPSD_SKIP_FIRST_CHUNK_LOSS", True)
cfg.opsd_loss_weight = _env_float("OPSD_LOSS_WEIGHT", 1.0)
cfg.opsd_debug_cache = _env_bool("OPSD_DEBUG_CACHE", False)
cfg.opsd_backward_per_chunk = _env_bool("OPSD_BACKWARD_PER_CHUNK", False)
cfg.opsd_activation_checkpointing = _env_bool("OPSD_ACTIVATION_CHECKPOINTING", True)
cfg.opsd_use_safe_dataset = _env_bool("OPSD_USE_SAFE_DATASET", False)
cfg.opsd_rollout_video_interval = _env_int(
    "OPSD_ROLLOUT_VIDEO_INTERVAL", 0)

# Disable first-stage distillation switches in this stage. These are kept only
# to make accidental use obvious in logs/config dumps.
cfg.distill_mode = "opsd"
cfg.distill_video = False
cfg.distill_action = False
cfg.action_aware = False
cfg.cfg_prob = 0.0
cfg.noisy_cond_prob = 0.0

# Common train-loop env overrides useful for smoke tests.
cfg.max_train_steps = _env_int("MAX_TRAIN_STEPS", cfg.max_train_steps)
cfg.batch_size = _env_int("BATCH_SIZE", cfg.batch_size)
cfg.dataset_init_worker = _env_int(
    "DATASET_INIT_WORKER",
    min(8, os.cpu_count() or 1),
)
cfg.gradient_accumulation_steps = _env_int(
    "GRADIENT_ACCUMULATION_STEPS", cfg.gradient_accumulation_steps)
cfg.save_interval = _env_int("SAVE_INTERVAL", cfg.save_interval)
cfg.gc_interval = _env_int("GC_INTERVAL", cfg.gc_interval)
cfg.load_worker = _env_int("LOAD_WORKER", cfg.load_worker)
cfg.enable_wandb = _env_bool("ENABLE_WANDB", cfg.enable_wandb)
cfg.learning_rate = _env_float("LEARNING_RATE", cfg.learning_rate)
cfg.ema_decay = _env_float("EMA_DECAY", cfg.ema_decay)
