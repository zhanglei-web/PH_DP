#!/usr/bin/env python3
"""Train Oracle Stage-Conditioned Diffusion V1 on the frozen successful dataset."""

from pathlib import Path
import argparse

from mujoco_shared_control.rss2023.oracle_stage_train import train_oracle


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z"
OUTPUT = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-steps", type=int, default=500)
    args = parser.parse_args()
    print(train_oracle(args.dataset_dir, args.output_dir, device_name=args.device, smoke_steps=args.smoke_steps))


if __name__ == "__main__":
    main()
