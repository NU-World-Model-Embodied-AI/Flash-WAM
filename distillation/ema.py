"""Exponential moving average update for the target student."""
import torch


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
