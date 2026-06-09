# Real-World Demos (Unitree G1)

Real-world rollouts from the [project page](https://flashwam.github.io), grouped by method/NFE setting. Each folder holds three task videos.

| Folder | Setting | Result |
| :--- | :--- | :--- |
| `LingBot-VA-3v10a/` | LingBot-VA teacher (3 video / 10 action steps) | reference |
| `No-Distillation-1v1a/` | LingBot-VA at reduced NFE (1 video / 1 action step), no distillation | fails |
| `Flash-WAM-1v1a/` | Flash-WAM (1 video / 1 action step) | succeeds |

Flash-WAM matches the teacher's behavior at a single denoising step per modality, where naively reducing the step count without distillation collapses.
