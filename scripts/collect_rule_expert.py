#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mujoco_shared_control.collection import AutomaticCollector, CollectionConfig
from mujoco_shared_control.collection.types import CollectionVariant


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect image-free Rule Expert episodes")
    parser.add_argument("--output", type=Path, default=Path("datasets/pick_box/expert_rule"))
    parser.add_argument("--nominal-success-target", type=int, default=2)
    parser.add_argument("--perturbed-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("v1 implements one worker; IDs already reserve worker_id")
    if args.nominal_success_target < 0 or args.perturbed_episodes < 0:
        parser.error("episode counts cannot be negative")
    config = CollectionConfig(dataset_root=str(args.output), max_steps=args.max_steps)
    collector = AutomaticCollector(config)
    results = []
    try:
        index = 0
        nominal_successes = 0
        while nominal_successes < args.nominal_success_target:
            if index >= args.max_attempts:
                raise RuntimeError(
                    f"max attempts reached with {nominal_successes}/"
                    f"{args.nominal_success_target} nominal successes"
                )
            result = collector.collect_episode(
                worker_episode_index=index, environment_seed=args.seed + index,
                variant=CollectionVariant.NOMINAL,
            )
            results.append(asdict(result))
            print(json.dumps(asdict(result), default=lambda value: value.value, ensure_ascii=False))
            nominal_successes += int(result.outcome.value == "success")
            index += 1
        for _ in range(args.perturbed_episodes):
            if index >= args.max_attempts:
                raise RuntimeError("max attempts reached before perturbed campaign completed")
            result = collector.collect_episode(
                worker_episode_index=index, environment_seed=args.seed + index,
                variant=CollectionVariant.PERTURBED,
            )
            results.append(asdict(result))
            print(json.dumps(asdict(result), default=lambda value: value.value, ensure_ascii=False))
            index += 1
    finally:
        collector.close()
    successes = sum(result["outcome"].value in {"success", "recovered"} for result in results)
    print(f"completed={len(results)} successful={successes} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
