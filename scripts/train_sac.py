#!/usr/bin/env python3
"""Formal entrypoint for SAC v1 pipeline and explicitly bounded sanity runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from mujoco_shared_control.sac.agent import SACCoreConfig
from mujoco_shared_control.sac.trainer import (
    EntropyWarmStartConfig,
    MediumHorizonReleaseConfig,
    MeanPolicyImprovementConfig,
    SACTrainer,
    TrainingProtocol,
)
from mujoco_shared_control.sac.policy_anchor import (
    InitialPolicyTrustRegionConfig,
    PolicyAnchorConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--critic-learning-starts", type=int, default=10_000)
    parser.add_argument("--actor-learning-starts", type=int, default=10_000)
    parser.add_argument(
        "--policy-anchor", action="store_true",
        help="enable the frozen-initial-policy KL anchor (v2 behavior-preserving schedule)",
    )
    parser.add_argument(
        "--medium-horizon", action="store_true",
        help="resume the validated 30k run with the fixed R0-R3 release schedule",
    )
    parser.add_argument(
        "--mean-policy-improvement", action="store_true",
        help="freeze entropy axis, mix deterministic/stochastic episodes, and use a movable stable anchor",
    )
    parser.add_argument("--policy-anchor-weight", type=float, default=0.1)
    parser.add_argument(
        "--safe-warm-start", action="store_true",
        help=(
            "10k deterministic collection, then stochastic Critic-only warmup, "
            "plus calibrated entropy and the initial-policy trust region"
        ),
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("sac_v1_%Y%m%dT%H%M%SZ")
    run_directory = Path("outputs/sac_training") / run_id
    actor_learning_starts = 20_000 if args.safe_warm_start else args.actor_learning_starts
    evaluation_steps = (
        (40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000)
        if args.medium_horizon else (
            (40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000)
            if args.mean_policy_improvement else TrainingProtocol().evaluation_steps
        )
    )
    protocol = TrainingProtocol(
        total_env_steps=args.steps,
        evaluation_steps=evaluation_steps,
        checkpoint_frequency=20_000 if args.medium_horizon else 50_000,
        critic_learning_starts=args.critic_learning_starts,
        actor_learning_starts=actor_learning_starts,
        alpha_learning_starts=actor_learning_starts,
        # Deterministic BC interaction populates Replay with successful behavior.
        # Critic-only warmup then switches to stochastic actions so Replay also
        # covers the same local support queried by the SAC target policy.
        deterministic_collection_until=(
            args.critic_learning_starts if args.safe_warm_start else 0
        ),
    )
    anchor_enabled = args.policy_anchor or args.safe_warm_start
    anchor_weight = 1.0 if args.safe_warm_start else args.policy_anchor_weight
    trainer = SACTrainer(
        run_directory,
        protocol,
        core_config=SACCoreConfig(alpha_init=0.0025) if args.safe_warm_start else SACCoreConfig(),
        device=args.device,
        policy_anchor_config=PolicyAnchorConfig(
            enabled=anchor_enabled, weight=anchor_weight
        ),
        policy_trust_region_config=InitialPolicyTrustRegionConfig(
            enabled=args.safe_warm_start,
            # This contact-rich task is exceptionally sensitive around the
            # grasp/release boundaries.  A 1e-2 empirical KL still changed all
            # six initialization successes into failures; projecting the same
            # checkpoint to 1e-5 restored 6/20 without touching reward or env.
            max_kl=1e-5,
            max_parameter_relative_radius=1e-4,
        ),
        entropy_warm_start_config=EntropyWarmStartConfig(enabled=args.safe_warm_start),
        medium_horizon_release_config=MediumHorizonReleaseConfig(
            enabled=args.medium_horizon
        ),
        mean_policy_improvement_config=MeanPolicyImprovementConfig(
            enabled=args.mean_policy_improvement
        ),
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    if trainer.global_env_steps == 0:
        initialization = trainer.evaluate()
        print(json.dumps({"initialization_evaluation": initialization}, indent=2))
    elif (args.medium_horizon or args.mean_policy_improvement) and trainer.global_env_steps == 30_000:
        reference = trainer.evaluate(100 if args.mean_policy_improvement else None)
        if args.mean_policy_improvement:
            trainer.stable_evaluation = reference
            trainer.stable_anchor_promotions = 1
            trainer.stable_checkpoint_path = str(
                trainer.checkpoint_directory / "stable.pt"
            )
            trainer.save_checkpoint("stable.pt")
        print(json.dumps({"medium_horizon_30k_reference": reference}, indent=2))
    result = trainer.train(args.steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
