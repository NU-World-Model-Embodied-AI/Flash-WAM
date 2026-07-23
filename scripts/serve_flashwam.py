import argparse
import copy
import dataclasses
import logging
import os
import socket
from pathlib import Path


@dataclasses.dataclass
class Args:
    """Serve Flash-WAM with the LingBot-VA websocket protocol."""

    checkpoint_dir: str = "hf_assets/FlashWAM-RoboTwin"
    config_name: str = "robotwin"
    host: str = "0.0.0.0"
    port: int = 29536
    save_root: str = "visualization"
    local_rank: int = 0
    num_inference_steps: int | None = None
    action_num_inference_steps: int | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Serve Flash-WAM with the LingBot-VA websocket protocol."
    )
    parser.add_argument("--checkpoint-dir", default=Args.checkpoint_dir)
    parser.add_argument("--config-name", default=Args.config_name)
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--save-root", default=Args.save_root)
    parser.add_argument("--local-rank", type=int, default=Args.local_rank)
    parser.add_argument("--num-inference-steps", type=int, default=Args.num_inference_steps)
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=Args.action_num_inference_steps,
    )
    return Args(**vars(parser.parse_args()))


def _resolve_checkpoint_dir(path: str) -> str:
    checkpoint_dir = Path(path).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir
    if not (checkpoint_dir / "transformer").is_dir():
        raise FileNotFoundError(
            f"{checkpoint_dir}/transformer not found. Pass --checkpoint-dir "
            "pointing to a Flash-WAM checkpoint snapshot."
        )
    return str(checkpoint_dir)


def create_flashwam_policy(args: Args):
    import torch

    from wan_va.configs import VA_CONFIGS
    from wan_va.wan_va_server import VA_Server

    if args.config_name not in VA_CONFIGS:
        valid = ", ".join(sorted(VA_CONFIGS))
        raise ValueError(f"Unknown config {args.config_name!r}. Valid configs: {valid}")

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.wan22_pretrained_model_name_or_path = _resolve_checkpoint_dir(
        args.checkpoint_dir
    )
    config.host = args.host
    config.port = args.port
    config.save_root = args.save_root
    config.infer_mode = "server"
    config.rank = 0
    config.local_rank = args.local_rank
    config.world_size = 1
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    if args.action_num_inference_steps is not None:
        config.action_num_inference_steps = args.action_num_inference_steps

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    return VA_Server(config)


def main(args: Args) -> None:
    from wan_va.utils import init_logger
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import (
        WebsocketPolicyServer,
    )

    init_logger()
    policy = create_flashwam_policy(args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logging.info("Creating Flash-WAM server (host: %s, ip: %s)", hostname, local_ip)

    metadata = {
        "model": "flashwam",
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "config_name": args.config_name,
        "num_inference_steps": args.num_inference_steps,
        "action_num_inference_steps": args.action_num_inference_steps,
    }
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(parse_args())
