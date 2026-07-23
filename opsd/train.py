"""Entrypoint for Flash-WAM OPSD stage-2 training."""
import argparse
import importlib
import os
import sys

import torch


OPSD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(OPSD_DIR)
WAN_VA_DIR = os.path.join(PROJECT_ROOT, "wan_va")
DISTILLATION_DIR = os.path.join(PROJECT_ROOT, "distillation")

def _install_paths():
    for path in (PROJECT_ROOT, WAN_VA_DIR, DISTILLATION_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


def run(args):
    _install_paths()
    from distillation.patches import install_flash_attn_stub
    install_flash_attn_stub()

    from opsd.trainer import FlashWAMOPSDTrainer
    from wan_va.distributed.util import init_distributed
    from wan_va.utils import init_logger, logger

    init_logger()

    config_mod = os.environ.get("CONFIG_FILE", "opsd.config")
    config = importlib.import_module(config_mod).cfg

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    else:
        torch.cuda.set_device(local_rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.student_model_path is not None:
        config.student_model_path = args.student_model_path
    elif args.teacher_model_path is not None:
        config.student_model_path = args.teacher_model_path
        if rank == 0:
            logger.warning("--teacher-model-path is deprecated for OPSD; use --student-model-path.")
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step
    if args.resume_from_path is not None:
        config.resume_from_path = args.resume_from_path
    if args.task_list is not None:
        config.opsd_task_list = args.task_list

    if rank == 0:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Student: {config.student_model_path}")
        logger.info(f"Dataset: {config.dataset_path}")
        logger.info(f"Output:  {config.output_dir}")
        logger.info(
            f"OPSD steps: video={config.opsd_video_num_inference_steps}, "
            f"action={config.opsd_action_num_inference_steps}")

    trainer = FlashWAMOPSDTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Flash-WAM OPSD stage-2 training")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--student-model-path", type=str, default=None)
    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--task-list",
        type=str,
        default=None,
        help="Saved JSON array of the 12 RoboTwin tasks used for this OPSD run.",
    )
    parser.add_argument(
        "--resume-from-step",
        type=int,
        default=None,
        help="Resume OPSD training from this checkpoint step.",
    )
    parser.add_argument(
        "--resume-from-path",
        type=str,
        default=None,
        help="Resume from explicit OPSD checkpoint directory.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
