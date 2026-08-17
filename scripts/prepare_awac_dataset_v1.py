#!/usr/bin/env python3
"""Prepare the read-only formal Rule Expert corpus for offline AWAC-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mujoco_shared_control.awac import convert_formal_rule_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert manifest-listed Rule Expert HDF5 into AWAC-v1 NPZ files"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/rule_expert_v1_formal.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/awac_dataset/awac_v1_formal_rule"),
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip per-HDF5 SHA-256 verification (not recommended for formal output)",
    )
    args = parser.parse_args()
    report = convert_formal_rule_dataset(
        args.manifest,
        args.output,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
