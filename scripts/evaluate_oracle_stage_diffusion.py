#!/usr/bin/env python3
"""Evaluate Oracle Stage Diffusion checkpoints on the Global dynamics seed bank."""

from pathlib import Path
import argparse
import csv
import json

from mujoco_shared_control.rss2023.oracle_stage_evaluation import run_evaluation


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=TRAINING)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-seed", type=int, default=2_000_000)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    root = args.training_dir
    files = [root / "checkpoints" / f"step_{step:08d}.pt" for step in range(10_000, 80_001, 10_000)] + [root / "best.pt"]
    rows = []
    for checkpoint in files:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report = run_evaluation(checkpoint, root / "normalization_stats.npz", root / "checkpoint_sweep_n100" / checkpoint.stem, formal_seeds=range(args.start_seed, args.start_seed + args.count), device=args.device)
        summary = report["summary"]
        rows.append({"checkpoint": checkpoint.name, "step": 80_000 if checkpoint.name == "best.pt" else int(checkpoint.stem.split("_")[-1]), "success": summary["success"]["count"], "grasp": summary["grasp"]["count"], "lift": summary["lift"]["count"], "transport": summary["transport"]["count"], "place": summary["place"]["count"], "release": summary["release"]["count"], "retreat": summary["retreat"]["count"], "illegal_drop": summary["illegal_drop"]["count"], "ik_failure": summary["ik_failure"]["count"], "timeout": summary["timeout"]["count"], "mean_episode_length": summary["episode_length"]["mean"], "average_return": summary["average_return"]})
        print(rows[-1], flush=True)
    with (root / "checkpoint_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "evaluation_seed_manifest.json").write_text(json.dumps({"environment_seeds": [args.start_seed, args.start_seed + args.count - 1], "sampling_seed_rule": "8_000_000 + environment_seed", "N": args.count}, indent=2) + "\n")


if __name__ == "__main__":
    main()
