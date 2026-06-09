"""
Flash-WAM: Modality-Aware LCM Distillation for LingBot-VA (joint video+action).

DISTILL_MODE selects the variant: flashwam (the paper's modality-aware method),
joint (naive joint LCM), video (video-only LCM), video_action_aware (video-only
LCM + action regularizer), action (action-only consistency).

Design: The training pipeline is nearly identical to wan_va/train.py.
All noise addition, timestep sampling, chunk_size, window_size sampling
are reused from the native training code without modification.

The ONLY differences from native training are:
  1. Three models: Teacher (frozen), Online Student (trainable), Target Student (EMA)
  2. Teacher uses CFG (scale sampled from [2, 10]) with two forward passes
  3. Student and Target Student do NOT use CFG
  4. Loss = Huber on consistency-function output, not MSE on v-prediction
  5. k = num_train_timesteps / num_ddim_timesteps = 1000 / 2 = 500
     (the stride in the 1000-step schedule for constructing training pairs)
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))

from patches import install_flash_attn_stub
install_flash_attn_stub()

from distributed.util import init_distributed
from utils import init_logger, logger
from trainer import FlashWAMDistiller


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

    trainer = FlashWAMDistiller(config)
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
