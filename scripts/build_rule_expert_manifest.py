#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_shared_control.collection.manifest import build_formal_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen Rule Expert v1 manifest")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("datasets/pick_box/expert_rule"))
    parser.add_argument("--output", type=Path,
                        default=Path("manifests/rule_expert_v1_formal.json"))
    parser.add_argument("--split-seed", type=int, default=20260812)
    args = parser.parse_args()
    manifest = build_formal_manifest(args.dataset_root, args.output,
                                     split_seed=args.split_seed)
    print(f"manifest={args.output.resolve()}")
    print(f"content_sha256={manifest['content_sha256']}")
    print(f"episodes={manifest['episode_count']} transitions={manifest['transition_count']}")
    print(f"split_counts={manifest['split']['counts']}")


if __name__ == "__main__":
    main()
