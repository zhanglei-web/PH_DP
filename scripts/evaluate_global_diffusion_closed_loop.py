#!/usr/bin/env python3
"""Run smoke and formal autonomous Global Diffusion evaluation."""

from pathlib import Path

from mujoco_shared_control.rss2023.global_evaluation import run_evaluation


CHECKPOINT = Path(
    "outputs/global_diffusion/global_diffusion_final_expert_20260816T180000Z/best.pt"
)
NORMALIZATION = Path(
    "outputs/global_diffusion/global_diffusion_final_expert_20260816T180000Z/normalization_stats.npz"
)
OUTPUT = Path(
    "outputs/global_diffusion/global_diffusion_final_expert_20260816T180000Z/closed_loop_300"
)


if __name__ == "__main__":
    report = run_evaluation(CHECKPOINT, NORMALIZATION, OUTPUT)
    print(report["summary"])
