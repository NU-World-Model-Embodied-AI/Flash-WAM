<h1 align="center">⚡ Flash-WAM: Modality-Aware Distillation for World Action Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.05254"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a>
  <a href="https://flashwam.github.io"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

Flash-WAM is a modality-aware step-distillation framework for joint video–action world models. It distills each modality with a consistency function matched to its noise regime — a linear-gradient-scaling choice for the low-noise action stream and a variance-preserving choice for the high-noise video stream — compressing LingBot-VA inference to a single step per modality. On RoboTwin 2.0 this yields up to a **23× speedup** (8.1 s → 348 ms per chunk) while preserving teacher-level task success.

This repository releases the Flash-WAM distillation code (the modality-aware joint method plus its LCM ablations) for RoboTwin.

## 📰 News

- **[2026-06]** Flash-WAM paper released on [arXiv](https://arxiv.org/abs/2606.05254).
- **[2026-06]** Distillation code (Flash-WAM + ablations) released.

## ✅ Checklist

- [x] Flash-WAM distillation code (modality-aware joint method)
- [x] LCM ablation modes (naive joint, video-only, video-only + reg)
- [ ] Distilled Flash-WAM checkpoints
- [ ] LIBERO distillation configs

## 📦 Model Checkpoints

| Model | Repository | Description |
| :--- | :--- | :--- |
| LingBot-VA teacher (posttrain-robotwin) | [🤗 robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | Teacher checkpoint to distill from |
| Flash-WAM distilled (RoboTwin) | _coming soon_ | Single-step distilled student |

Post-training dataset: [🤗 robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot).

## 🚀 Quick Start

Flash-WAM builds on **LingBot-VA**. For **environment installation** and **evaluation**, follow the [LingBot-VA repository](https://github.com/Robbyant/lingbot-va) — Flash-WAM uses the same environment and the same RoboTwin server/client evaluation pipeline. Once the LingBot-VA environment is set up, this repository runs in it directly.

## 🔬 Distillation

Point the distiller at the teacher checkpoint and dataset, then select the method via `DISTILL_MODE`:

```bash
export TEACHER_PATH=/path/to/lingbot-va-posttrain-robotwin
export DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot

# Flash-WAM (the paper's modality-aware joint method)
DISTILL_MODE=flashwam bash distill_video_v2/run.sh

# LCM ablations from the paper
DISTILL_MODE=joint              bash distill_video_v2/run.sh   # naive joint LCM
DISTILL_MODE=video              bash distill_video_v2/run.sh   # video-only LCM
DISTILL_MODE=video_action_aware bash distill_video_v2/run.sh   # video-only LCM + reg
```

Key knobs (see `distill_video_v2/config.py`): `NGPU`, `OUTPUT_DIR`, `num_ddim_timesteps` (student video steps), `num_ddim_timesteps_action` (student action steps), `lcm_skip_k`, `cfg_min`/`cfg_max` (teacher CFG range).

## 📝 Citation

```bibtex
@misc{akbari2026flashwammodalityawaredistillationworld,
      title={Flash-WAM: Modality-Aware Distillation for World Action Models}, 
      author={Arman Akbari and Ci Zhang and Arash Akbari and Lin Zhao and Yixiao Chen and Weiwei Chen and Xuan Zhang and Geng Yuan and Yanzhi Wang},
      year={2026},
      eprint={2606.05254},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.05254}, 
}
```

## Acknowledgements

Built on [LingBot-VA](https://github.com/Robbyant/lingbot-va) and evaluated on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Licensed under Apache-2.0.
