#!/usr/bin/env python3
"""Derive AWAC Reward V1 labels without modifying source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mujoco_shared_control.awac.reward import derive_awac_reward_v1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("manifests/rule_expert_v1_formal.json"),
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("outputs/awac_dataset/awac_v1_formal_rule"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/awac_dataset/awac_reward_v1_formal_rule"),
    )
    args = parser.parse_args()
    report = derive_awac_reward_v1(args.manifest, args.source_dir, args.output_dir)
    print(json.dumps({
        "version": report["version"],
        "output_dataset_dir": report["output_dataset_dir"],
        "transition_count_by_split": report["transition_count_by_split"],
        "terminal_reconstruction": report["terminal_reconstruction"],
        "filtered_tail_summary": report["filtered_tail_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
