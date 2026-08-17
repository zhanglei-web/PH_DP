#!/usr/bin/env python3
"""Train Stage TCN V1 from the frozen Stage Dataset V1 split."""

from pathlib import Path

from mujoco_shared_control.stage.train import train_stage_tcn


if __name__ == "__main__":
    checkpoint = train_stage_tcn(
        Path("outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z"),
        Path("outputs/stage_tcn/stage_tcn_v1_hysteresis_20260817T070000Z"),
    )
    print(f"best checkpoint: {checkpoint}")
