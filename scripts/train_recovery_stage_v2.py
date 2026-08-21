#!/usr/bin/env python3
"""Train Recovery Stage V2 (independent stage/physical fusion), CUDA only."""
from pathlib import Path
import argparse
from mujoco_shared_control.rss2023.recovery_stage_train import DATASET_DEFAULT, train

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=ROOT / DATASET_DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(train("v2", a.dataset, a.output, steps=a.steps, smoke=a.smoke))

if __name__ == "__main__":
    main()
