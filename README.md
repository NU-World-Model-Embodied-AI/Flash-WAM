<h1 align="center">⚡ Flash-WAM: Modality-Aware Distillation for World Action Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.05254"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a>
  <a href="https://flashwam.github.io"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

Flash-WAM is a modality-aware step-distillation framework for joint video–action world models. It distills each modality with a consistency function matched to its noise regime — a linear-gradient-scaling choice for the low-noise action stream and a variance-preserving choice for the high-noise video stream — compressing LingBot-VA inference to a single step per modality. On RoboTwin 2.0 this yields up to a **23× speedup** (8.1 s → 348 ms per chunk) while preserving teacher-level task success.

This repository provides the Flash-WAM distillation code (the modality-aware method plus its LCM ablations), the model code, and the distilled RoboTwin checkpoint.

## 📰 News

- **[2026-06]** Flash-WAM RoboTwin checkpoint released on [🤗 HuggingFace](https://huggingface.co/NU-World-Model-Embodied-AI/FlashWAM-RoboTwin).
- **[2026-06]** Flash-WAM paper released on [arXiv](https://arxiv.org/abs/2606.05254).

## ✅ Checklist

- [x] Flash-WAM RoboTwin checkpoint
- [x] Distillation code (Flash-WAM + LCM ablations)
- [ ] Real-world deployment setup on Unitree G1 humanoid

## 📦 Model Checkpoints

| Model | Repository | Description |
| :--- | :--- | :--- |
| LingBot-VA teacher (posttrain-robotwin) | [🤗 robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | Teacher checkpoint to distill from |
| Flash-WAM distilled (RoboTwin) | [🤗 NU-World-Model-Embodied-AI/FlashWAM-RoboTwin](https://huggingface.co/NU-World-Model-Embodied-AI/FlashWAM-RoboTwin) | Single-step distilled student (complete, with encoders) |

Post-training dataset: [🤗 robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot).

## 🚀 Quick Start

Flash-WAM builds on **LingBot-VA**. For **environment installation** and **evaluation**, follow the [LingBot-VA repository](https://github.com/Robbyant/lingbot-va) — Flash-WAM uses the same environment and the same RoboTwin server/client evaluation pipeline. Once the LingBot-VA environment is set up, this repository runs in it directly.

## 🔬 Distillation

Point the distiller at the teacher checkpoint and dataset, then select the method via `DISTILL_MODE`:

```bash
export TEACHER_PATH=/path/to/lingbot-va-posttrain-robotwin
export DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot

# Flash-WAM (the paper's modality-aware joint method)
DISTILL_MODE=flashwam bash distillation/run.sh

# LCM ablations from the paper
DISTILL_MODE=joint              bash distillation/run.sh   # naive joint LCM
DISTILL_MODE=video              bash distillation/run.sh   # video-only LCM
DISTILL_MODE=video_action_aware bash distillation/run.sh   # video-only LCM + reg
```

Key knobs (see `distillation/config.py`): `NGPU`, `OUTPUT_DIR`, `num_ddim_timesteps` (student video steps), `num_ddim_timesteps_action` (student action steps), `cfg_min`/`cfg_max` (teacher CFG range).

## 📊 Results

### Ablation — RoboTwin 2.0 (success rate %, 50 tasks, by task horizon)

Flash-WAM is the only LCM-based strategy that preserves teacher-level accuracy across both NFE budgets and all horizons: naive joint LCM collapses, and the video-only variants trail.

| Method | Nᵥ | Nₐ | H1 Clean | H1 Rand | H2 Clean | H2 Rand | H3 Clean | H3 Rand | Avg Clean | Avg Rand |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| LingBot-VA (teacher) | 25 | 50 | 94.18 | 93.56 | 90.35 | 86.95 | 93.22 | 93.28 | 92.93 | 91.55 |
| Video-only LCM | 1 | 2 | 87.10 | 82.73 | 73.13 | 68.19 | 62.50 | 68.25 | 80.66 | 76.92 |
| Video-only LCM + reg. | 1 | 2 | 91.53 | 88.50 | 83.00 | 74.69 | 68.00 | 62.75 | 86.92 | 82.02 |
| Naive Joint LCM | 1 | 2 | 41.00 | 35.13 | 4.00 | 3.13 | 0.00 | 0.00 | 25.88 | 20.08 |
| **Flash-WAM** | 1 | 2 | **92.30** | **88.47** | **84.88** | **76.63** | **73.50** | **63.25** | **88.42** | **82.66** |
| Video-only LCM | 1 | 1 | 85.57 | 78.17 | 72.06 | 61.81 | 43.75 | 34.75 | 77.90 | 69.46 |
| Video-only LCM + reg. | 1 | 1 | 66.87 | 61.07 | 39.19 | 35.56 | 10.25 | 4.75 | 53.48 | 48.40 |
| Naive Joint LCM | 1 | 1 | 54.63 | 46.00 | 21.56 | 15.63 | 0.00 | 0.00 | 39.68 | 32.96 |
| **Flash-WAM** | 1 | 1 | **87.30** | **86.93** | **78.44** | **72.63** | **63.50** | **60.75** | **82.56** | **80.26** |

### Real-World — Unitree G1 (success rate %, 3 tasks × 10 rollouts)

T1: open pot lid and place a potato inside · T2: pick the red bottle (with a yellow distractor) · T3: pick the pink object and place it on the target. See [`demo/`](demo/) for rollout videos.

| Method | Nᵥ / Nₐ | T1 | T2 | T3 | Average |
| :--- | :-: | :-: | :-: | :-: | :-: |
| LingBot-VA | 3 / 10 | 50 | 70 | 80 | 66.7 |
| LingBot-VA (reduced NFE) | 1 / 2 | 30 | 30 | 60 | 40.0 |
| LingBot-VA + Video-only LCM | 1 / 2 | 30 | 50 | 50 | 43.3 |
| **Flash-WAM** | 1 / 2 | **50** | **60** | **70** | **60.0** |
| LingBot-VA (reduced NFE) | 1 / 1 | 10 | 30 | 30 | 23.3 |
| LingBot-VA + Video-only LCM | 1 / 1 | 20 | 40 | 40 | 33.3 |
| **Flash-WAM** | 1 / 1 | **40** | **50** | **60** | **50.0** |

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

Built on [LingBot-VA](https://github.com/Robbyant/lingbot-va) (Apache-2.0) and evaluated on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Flash-WAM's own code, documentation, and demos are released under [CC BY-NC 4.0](LICENSE.txt) (non-commercial use only); the bundled LingBot-VA components in `wan_va/` remain under Apache-2.0.
