"""Data helpers for Flash-WAM OPSD training."""
from __future__ import annotations

from typing import Dict, List

import torch


def move_batch_to_device(batch: Dict[str, torch.Tensor], device, dtype) -> Dict[str, torch.Tensor]:
    """Move tensors to the training device while preserving boolean masks."""
    out = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.dtype == torch.bool:
            out[key] = value.to(device)
        elif torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device)
    return out


def split_batch_into_chunks(batch: Dict[str, torch.Tensor], frame_chunk_size: int) -> List[dict]:
    # Inputs:
    #   batch["latents"]: Tensor[B,C,F,H,W] GT video latents
    #   batch["actions"]: Tensor[B,C,F,N,1] GT normalized actions
    #   batch["actions_mask"]: BoolTensor[B,C,F,N,1]
    #   batch["text_emb"]: Tensor[B,L,D]
    # Outputs:
    #   list[dict]: one dict per chunk with video/action/action_mask/text_emb
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive.")

    required = ("latents", "actions", "actions_mask", "text_emb")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing OPSD batch keys: {missing}")

    latents = batch["latents"]
    actions = batch["actions"]
    actions_mask = batch["actions_mask"]
    text_emb = batch["text_emb"]

    if latents.ndim != 5:
        raise ValueError(f"latents must be 5D [B,C,F,H,W], got {latents.shape}")
    if actions.ndim != 5:
        raise ValueError(f"actions must be 5D [B,C,F,N,1], got {actions.shape}")
    if actions_mask.shape != actions.shape:
        raise ValueError(
            f"actions_mask shape {actions_mask.shape} must match actions {actions.shape}")
    if latents.shape[0] != actions.shape[0]:
        raise ValueError("latents/actions batch sizes must match.")
    if latents.shape[2] != actions.shape[2]:
        raise ValueError("latents/actions frame counts must match.")

    chunks = []
    num_frames = latents.shape[2]
    for frame_start in range(0, num_frames, frame_chunk_size):
        frame_end = min(frame_start + frame_chunk_size, num_frames)
        chunks.append({
            "video": latents[:, :, frame_start:frame_end],
            "action": actions[:, :, frame_start:frame_end],
            "action_mask": actions_mask[:, :, frame_start:frame_end],
            "frame_start": frame_start,
            "text_emb": text_emb,
        })
    return chunks


class OPSDDataMixin:
    """Small local copy of the existing distillation data-iterator pattern."""

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        return batch
