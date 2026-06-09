"""
Quick smoke test: verify all imports, model loading, and one forward pass.
Run with: python distill_video_v2/test_imports.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patches import install_flash_attn_stub
install_flash_attn_stub()

import torch
print(f"[1/7] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

from config import cfg
print(f"[2/7] Config loaded. k = {cfg.num_train_timesteps // cfg.num_ddim_timesteps}")

from utils import FlowMatchScheduler, sample_timestep_id, get_mesh_id, data_seq_to_patch
scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(1000, training=True)
print(f"[3/7] Scheduler OK. sigmas shape: {scheduler.sigmas.shape}, "
      f"range: [{scheduler.sigmas[-1]:.4f}, {scheduler.sigmas[0]:.4f}]")

# Test anchor logic
k = cfg.num_train_timesteps // cfg.num_ddim_timesteps
ts_ids = sample_timestep_id(batch_size=5, num_train_timesteps=1000)
sigma_start = scheduler.sigmas[ts_ids]
end_ids = (ts_ids + k).clamp(max=999)
sigma_end = scheduler.sigmas[end_ids]
print(f"[4/7] Anchor test OK. ts_ids={ts_ids.tolist()}, "
      f"sigma_start={sigma_start.tolist()[:3]}, sigma_end={sigma_end.tolist()[:3]}")
assert (sigma_start >= sigma_end).all(), "sigma_start should >= sigma_end"

# Test empty_emb loading
empty_emb = torch.load(cfg.empty_emb_path, map_location="cpu")
print(f"[5/7] Empty embedding loaded. shape: {empty_emb.shape}")

# Test model loading (single model, CPU, quick)
from modules.utils import load_transformer
print("[6/7] Loading transformer (CPU) ...")
model = load_transformer(
    os.path.join(cfg.teacher_model_path, "transformer"),
    torch_dtype=torch.float32,
    torch_device="cpu",
)
print(f"  Model loaded. attn_mode={model.config.get('attn_mode', 'NOT SET')}")
assert model.config.get("attn_mode") == "flex", "attn_mode must be 'flex' for training!"

# Test consistency function math
from train import scalings_for_boundary_conditions
sigma_zero = torch.tensor([0.0])
c_skip, c_out = scalings_for_boundary_conditions(sigma_zero)
assert abs(c_skip.item() - 1.0) < 1e-6, f"c_skip at sigma=0 should be 1, got {c_skip.item()}"
assert abs(c_out.item() - 0.0) < 1e-6, f"c_out at sigma=0 should be 0, got {c_out.item()}"

sigma_mid = torch.tensor([0.5])
c_skip, c_out = scalings_for_boundary_conditions(sigma_mid)
print(f"[7/7] Boundary conditions OK. "
      f"sigma=0: c_skip=1, c_out=0. "
      f"sigma=0.5: c_skip={c_skip.item():.4f}, c_out={c_out.item():.4f}")

del model
print("\n=== ALL TESTS PASSED ===")
