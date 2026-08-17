#!/usr/bin/env python3
"""Geometric 48-D Offline-to-Online Hybrid AWAC, through Online 20k."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.evaluation import evaluate_policy
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.milestones import MilestoneTracker
from mujoco_shared_control.awac.online import (
    UnifiedHybridReplay, low_noise_behavior_action, restore_hybrid_awac_trainer,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig
from mujoco_shared_control.sac.diagnostics import restore_environment, snapshot_environment


OFFLINE_CHECKPOINT = Path(
    "outputs/awac_training/awac_v3_geometric_milestone_state_offline25k_20260814T160000Z/checkpoint_best.pt"
)
OFFLINE_DATASET = Path("outputs/awac_dataset/awac_v3_geometric_milestone_state/train.npz")
EVALUATION_STEPS = (
    0, 1_000, 2_500, 5_000, 7_500, 10_000,
    12_500, 15_000, 17_500, 20_000,
)
VALIDATION_SEEDS = list(range(300_000, 300_100))
TRAINING_SEED_START = 900_000
REPLAY_CAPACITY = 1_000_000
SANITY_EPISODES = 0
ONLINE_UPDATES = 20_000
EXPLORATION_STD_SCALE = 0.25


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(value, temporary); temporary.replace(path)


def save_online_log(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    arrays: dict[str, np.ndarray] = {
        "obs": np.asarray([row["obs"] for row in rows], np.float32).reshape(-1, 48),
        "continuous_action": np.asarray([row["continuous_action"] for row in rows], np.float32).reshape(-1, 6),
        "gripper_action": np.asarray([row["gripper_action"] for row in rows], np.float32),
        "reward": np.asarray([row["reward"] for row in rows], np.float32),
        "next_obs": np.asarray([row["next_obs"] for row in rows], np.float32).reshape(-1, 48),
        "terminated": np.asarray([row["terminated"] for row in rows], bool),
        "truncated": np.asarray([row["truncated"] for row in rows], bool),
        "episode_id": np.asarray([row["episode_id"] for row in rows]),
        "step": np.asarray([row["step"] for row in rows], np.int64),
        "task_success": np.asarray([row["task_success"] for row in rows], bool),
        "milestones": np.asarray([row["milestones"] for row in rows], np.uint8).reshape(-1, 5),
        "termination_reason": np.asarray([row["termination_reason"] for row in rows]),
        "gripper_state": np.asarray([row["gripper_state"] for row in rows]),
        "object_grasped": np.asarray([row["object_grasped"] for row in rows], bool),
        "online_policy_step": np.asarray([row["online_policy_step"] for row in rows], np.int64),
        "collection_step": np.asarray([row["collection_step"] for row in rows], np.int64),
        "collection_phase": np.asarray([row["collection_phase"] for row in rows]),
        "requested_continuous_action": np.asarray([row["requested_continuous_action"] for row in rows], np.float32).reshape(-1, 6),
        "requested_gripper_action": np.asarray([row["requested_gripper_action"] for row in rows], np.float32),
        "continuous_policy_std_mean": np.asarray([row["continuous_policy_std_mean"] for row in rows], np.float32),
        "continuous_effective_std_mean": np.asarray([row["continuous_effective_std_mean"] for row in rows], np.float32),
        "behavior_std_scale": np.asarray([row["behavior_std_scale"] for row in rows], np.float32),
        "release_retreat_state": np.asarray([row["release_retreat_state"] for row in rows], bool),
        "transport_done_state": np.asarray([row["transport_done_state"] for row in rows], bool),
        "retreat_target_distance": np.asarray([row["retreat_target_distance"] for row in rows], np.float32),
        "gripper_close_probability": np.asarray([row["gripper_close_probability"] for row in rows], np.float32),
        "action_clipped": np.asarray([row["action_clipped"] for row in rows], bool),
        "fallback_used": np.asarray([row["fallback_used"] for row in rows], bool),
        "rejection_reason": np.asarray([row["rejection_reason"] for row in rows]),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_online_log(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        # NPZ members are compressed. Indexing ``arrays[name]`` inside the row
        # loop decompresses the complete member every time and becomes
        # prohibitively expensive as the resumable online log grows. Materialize
        # every member exactly once; row conversion below is otherwise identical.
        loaded = {name: arrays[name] for name in arrays.files}
    return [
        {
            name: value[index].copy() if value[index].ndim else value[index].item()
            for name, value in loaded.items()
        }
        for index in range(len(loaded["reward"]))
    ]


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluation_summary(value: dict[str, Any]) -> dict[str, Any]:
    place = value["place_success"]["count"]
    return {
        "success": value["task_success"], "success_rate": value["task_success_rate"],
        "grasp": value["grasp_success"]["count"], "lift": value["lift_success"]["count"],
        "transport": value["transport_success"]["count"], "place": place,
        "release": place, "retreat": value["retreat_success"]["count"],
        "illegal_drop": value["illegal_drop"]["count"],
        "ik_failure": value["ik_failure"]["count"], "timeout": value["timeout"]["count"],
        "average_return": value["average_episode_return"],
        "place_to_success_conversion": value["task_success"] / max(place, 1),
    }


def load_release_retreat_validation(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "observation": torch.from_numpy(np.asarray(data["obs"], np.float32)),
            "continuous": torch.from_numpy(np.asarray(data["continuous_action"], np.float32)),
            "gripper": torch.from_numpy(
                np.asarray(data["gripper_action"], np.float32)).unsqueeze(1),
        }


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    array = value.detach().cpu().numpy().astype(np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


@torch.no_grad()
def release_retreat_diagnostic(trainer, arrays: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    observation = arrays["observation"].to(trainer.device)
    normalized = trainer.normalize(observation)
    continuous = arrays["continuous"].to(trainer.device)
    gripper = arrays["gripper"].to(trainer.device)
    mask = (observation[:, 46] > .5) & (observation[:, 47] <= .5)
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        sampled_continuous, sampled_gripper, _ = trainer.actor.sample(normalized)
    dataset_q = torch.minimum(
        trainer.q1(normalized, continuous, gripper),
        trainer.q2(normalized, continuous, gripper),
    ).squeeze(1)
    policy_q = torch.minimum(
        trainer.q1(normalized, sampled_continuous, sampled_gripper),
        trainer.q2(normalized, sampled_continuous, sampled_gripper),
    ).squeeze(1)
    action, _binary, close_probability = trainer.actor.deterministic_action(normalized)
    return {
        "transitions": int(mask.sum()),
        "actor_dz_mean": float(action[mask, 2].mean()),
        "actor_dz_positive_ratio": float((action[mask, 2] > 0).float().mean()),
        "gripper_close_probability": tensor_stats(close_probability[mask].squeeze(1)),
        "dataset_q": tensor_stats(dataset_q[mask]),
        "advantage": tensor_stats((dataset_q - policy_q)[mask]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_online"))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.stop_after_updates is not None and not 0 <= args.stop_after_updates <= ONLINE_UPDATES:
        parser.error("--stop-after-updates must be within [0, 20000]")
    resume_checkpoint = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
    run = (
        resume_checkpoint.parent.parent if resume_checkpoint else
        (args.output_root / f"online_awac_v3_geometric_hybrid_{args.run_id}").resolve()
    )
    checkpoint_dir = run / "checkpoints"; evaluation_dir = run / "closed_loop"; replay_dir = run / "online_replay"
    if resume_checkpoint is None:
        checkpoint_dir.mkdir(parents=True, exist_ok=False); evaluation_dir.mkdir(); replay_dir.mkdir()
    elif not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    device = torch.device(
        "cuda" if args.device == "cuda"
        else "cpu" if args.device == "cpu"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    offline_checkpoint = OFFLINE_CHECKPOINT.resolve(); offline_dataset = OFFLINE_DATASET.resolve()
    restore_path = resume_checkpoint or offline_checkpoint
    trainer, restored_payload = restore_hybrid_awac_trainer(restore_path, device=device)
    reward_config = AWACRewardV1Config(**restored_payload["reward_config"])
    if asdict(reward_config) != asdict(AWACRewardV1Config()):
        raise RuntimeError("Online run refused: Reward V1 differs from frozen definition")
    replay = UnifiedHybridReplay(offline_dataset, capacity=REPLAY_CAPACITY, device=device)
    diagnostic_validation = load_release_retreat_validation(
        offline_dataset.parent / "validation.npz")
    expected_step = 20_000 + int(restored_payload.get("online_awac_update_step", 0))
    if (
        replay.offline_count != 135_237 or replay.observation_dim != 48
        or trainer.step != expected_step or trainer.config.observation_dim != 48
        or bool(restored_payload.get("legacy_rule_milestones_used", False))
    ):
        raise RuntimeError("Online run is not continuous with frozen Offline AWAC")
    if resume_checkpoint is None:
        source_rng = restored_payload.get("rng_state")
        if source_rng is None:
            raise RuntimeError("Offline 20k checkpoint must contain RNG state")
        random.setstate(source_rng["python"])
        np.random.set_state(source_rng["numpy"])
        torch.set_rng_state(source_rng["torch"])
        trainer.generator.set_state(source_rng["trainer_generator"])
    spec = ExpertActionSpec(); rule = RuleExpertConfig()
    normalized_open = float(spec.normalize(np.r_[np.zeros(6), rule.open_gripper_m])[6])
    normalized_close = float(spec.normalize(np.r_[np.zeros(6), rule.close_gripper_m])[6])
    gripper_threshold = 0.5 * (normalized_open + normalized_close)
    if not np.isclose(gripper_threshold, 0.375):
        raise RuntimeError("frozen gripper mapping threshold changed")
    config = CollectionConfig()
    transition_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    release_retreat_evaluations: dict[str, Any] = {}
    protection: dict[str, Any] | None = None
    env: PickPlaceEnv | None = None; adapter: ExpertCommandAdapter | None = None
    observation: dict[str, Any] | None = None; reward_protocol: AWACRewardV1Online | None = None
    episode_index = -1; episode_step = 0; episode_return = 0.0; consecutive_ik = 0
    episode_requested_close = 0; episode_actual_close = 0
    episode_release_step: int | None = None; episode_retreat_step: int | None = None

    if resume_checkpoint is not None:
        transition_rows = load_online_log(Path(restored_payload["online_transition_log"]))
        for row in transition_rows:
            replay.append(
                row["obs"], row["continuous_action"], float(row["gripper_action"]),
                float(row["reward"]), row["next_obs"], bool(row["terminated"]),
                bool(row["truncated"]),
            )
        episode_rows = read_json_lines(run / "online_episodes.jsonl")
        update_rows = read_json_lines(run / "training_metrics.jsonl")
        for step in EVALUATION_STEPS:
            evaluation_path = evaluation_dir / f"online_step_{step:05d}.json"
            if evaluation_path.exists():
                evaluations[str(step)] = json.loads(evaluation_path.read_text())
        rng = restored_payload["rng_state"]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"]); trainer.generator.set_state(rng["trainer_generator"])
        rollout = restored_payload.get("rollout_state")
        if rollout is not None:
            episode_index = int(rollout["episode_index"])
            episode_step = int(rollout["episode_step"])
            episode_return = float(rollout["episode_return"])
            episode_requested_close = int(rollout["episode_requested_close"])
            episode_actual_close = int(rollout["episode_actual_close"])
            episode_release_step = rollout.get("episode_release_step")
            episode_retreat_step = rollout.get("episode_retreat_step")
            env = PickPlaceEnv(
                render_mode=None, control_timestep=config.control_timestep_s,
                max_episode_steps=config.max_steps, enable_camera=False,
            )
            adapter = ExpertCommandAdapter(env.ik_controller, spec)
            reset_observation, _ = env.reset(seed=TRAINING_SEED_START + episode_index, options={
                "randomize_arm": config.randomize_arm,
                "arm_joint_noise_scale": config.arm_joint_noise_scale,
                "randomize_object": config.randomize_object,
                "randomize_goal": config.randomize_goal,
            })
            adapter.reset(reset_observation["ee_pose"], reset_observation["q_obs"])
            consecutive_ik = restore_environment(env, adapter, rollout["environment"])
            observation = rollout["observation"]
            initial_state_43 = np.r_[
                env.get_policy_observation(observation),
                np.float32(bool(observation["object_grasped"])),
            ].astype(np.float32)
            reward_protocol = AWACRewardV1Online(initial_state_43, reward_config)
            reward_protocol.load_state_dict(rollout["reward_state"])

    base_metadata = {
        "offline_checkpoint": str(offline_checkpoint), "offline_checkpoint_sha256": sha(offline_checkpoint),
        "offline_dataset": str(offline_dataset), "offline_dataset_sha256": sha(offline_dataset),
        "dataset_report_sha256": restored_payload["dataset_report_sha256"],
        "dataset_files_sha256": restored_payload["dataset_files_sha256"],
        "state_definition": restored_payload["state_definition"],
        "legacy_rule_milestones_used": False,
        "reward_version": "awac_reward_v1", "reward_config": asdict(reward_config),
        "online_protocol": {
            "replay": "single uniform offline+online buffer", "capacity": REPLAY_CAPACITY,
            "updates_per_environment_step": 1, "extra_exploration_noise": False,
            "actor_objective": "unchanged Hybrid AWAC joint-log-prob objective",
            "sanity_episodes": SANITY_EPISODES,
            "sanity_updates": 0,
            "training_environment_transitions": ONLINE_UPDATES,
            "online_updates": ONLINE_UPDATES,
            "behavior_continuous_std_scale_before_transport": EXPLORATION_STD_SCALE,
            "behavior_continuous_std_scale_after_transport": 0.0,
            "behavior_gripper": "deterministic probability threshold at 0.5",
            "evaluation_steps": list(EVALUATION_STEPS), "validation_seeds": [300000, 300099],
        },
    }

    def checkpoint(online_step: int, *, label: str | None = None) -> Path:
        stem = label or f"online_step_{online_step:05d}"
        replay_stem = label or f"step_{online_step:05d}"
        replay_log = replay_dir / f"online_transitions_{replay_stem}.npz"
        save_online_log(replay_log, transition_rows)
        rollout_state = None
        if env is not None and adapter is not None and observation is not None and reward_protocol is not None:
            rollout_state = {
                "environment": snapshot_environment(env, adapter, consecutive_ik),
                "observation": observation, "reward_state": reward_protocol.state_dict(),
                "episode_index": episode_index, "episode_step": episode_step,
                "episode_return": episode_return,
                "episode_requested_close": episode_requested_close,
                "episode_actual_close": episode_actual_close,
                "episode_release_step": episode_release_step,
                "episode_retreat_step": episode_retreat_step,
            }
        metadata = {
            **base_metadata, "format_version": "online_awac_v3_geometric_milestone_state",
            "online_environment_step": replay.online_count,
            "online_awac_update_step": online_step,
            "offline_pretrain_updates": 20_000,
            "offline_optimizer_step": 20_000,
            "sanity_transition_count": sanity_complete_step,
            "replay_metadata": replay.metadata(), "online_transition_log": str(replay_log),
            "online_transition_log_sha256": sha(replay_log),
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "trainer_generator": trainer.generator.get_state(),
            },
            "training_seed_start": TRAINING_SEED_START, "rollout_state": rollout_state,
        }
        path = checkpoint_dir / f"{stem}.pt"
        atomic_save(trainer.checkpoint(metadata), path)
        return path

    stream_mode = "a" if resume_checkpoint is not None else "w"
    metrics_stream = (run / "training_metrics.jsonl").open(stream_mode)
    episodes_stream = (run / "online_episodes.jsonl").open(stream_mode)
    sanity_complete_step: int | None = (
        restored_payload.get("sanity_transition_count") if resume_checkpoint else 0
    )
    online_step = int(restored_payload.get("online_awac_update_step", 0)) if resume_checkpoint else 0
    try:
        if resume_checkpoint is None:
            step0 = checkpoint(0)
            print("evaluating online step 0", flush=True)
            evaluations["0"] = evaluate_policy(HybridCheckpointPredictor(step0), VALIDATION_SEEDS, reward_config)
            release_retreat_evaluations["0"] = release_retreat_diagnostic(
                trainer, diagnostic_validation, trainer.config.seed + 20_000)
            (evaluation_dir / "online_step_00000.json").write_text(json.dumps(evaluations["0"], indent=2) + "\n")
            print(json.dumps({"step": 0, **evaluation_summary(evaluations["0"])}), flush=True)

        # Five complete sanity episodes have variable length. The generous bound
        # is unreachable during a healthy run but prevents an accidental infinite loop.
        for collection_step in range(replay.online_count + 1, 5_000 * SANITY_EPISODES + ONLINE_UPDATES + 1):
            if online_step >= ONLINE_UPDATES:
                break
            collection_phase = "sanity" if sanity_complete_step is None else "training"
            if env is None:
                episode_index += 1; episode_step = 0; episode_return = 0.0; consecutive_ik = 0
                episode_requested_close = 0; episode_actual_close = 0
                episode_release_step = None; episode_retreat_step = None
                env = PickPlaceEnv(
                    render_mode=None, control_timestep=config.control_timestep_s,
                    max_episode_steps=config.max_steps, enable_camera=False,
                )
                adapter = ExpertCommandAdapter(env.ik_controller, spec)
                observation, _ = env.reset(seed=TRAINING_SEED_START + episode_index, options={
                    "randomize_arm": config.randomize_arm,
                    "arm_joint_noise_scale": config.arm_joint_noise_scale,
                    "randomize_object": config.randomize_object,
                    "randomize_goal": config.randomize_goal,
                })
                adapter.reset(observation["ee_pose"], observation["q_obs"])
                initial_state_43 = np.r_[env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"]))].astype(np.float32)
                reward_protocol = AWACRewardV1Online(initial_state_43, reward_config)
            assert adapter is not None and observation is not None and reward_protocol is not None
            episode_step += 1
            state_43 = np.r_[env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"]))].astype(np.float32)
            state = np.r_[state_43, reward_protocol.tracker.current.astype(np.float32)].astype(np.float32)
            transport_done_state = bool(state[45] > .5)
            release_retreat_state = bool(state[46] > .5 and state[47] <= .5)
            behavior_std_scale = 0.0 if transport_done_state else EXPLORATION_STD_SCALE
            retreat_target_distance = (
                float(np.linalg.norm(
                    state_43[14:17] - reward_protocol.tracker.retreat_target(state_43)))
                if release_retreat_state else np.nan
            )
            normalized = trainer.normalize(torch.from_numpy(state).to(device)).unsqueeze(0)
            trainer.actor.eval()
            with torch.no_grad():
                (
                    requested_continuous_tensor, requested_gripper_tensor,
                    policy_std_tensor, effective_std_tensor, close_probability_tensor,
                ) = low_noise_behavior_action(
                    trainer.actor, normalized,
                    exploration_std_scale=behavior_std_scale,
                )
            trainer.actor.train()
            requested_continuous = requested_continuous_tensor.squeeze(0).cpu().numpy().astype(np.float64)
            requested_close = bool(requested_gripper_tensor.item())
            policy_std_mean = float(policy_std_tensor.mean())
            effective_std_mean = float(effective_std_tensor.mean())
            close_probability = float(close_probability_tensor.item())
            episode_requested_close += int(requested_close)
            requested_gripper = normalized_close if requested_close else normalized_open
            requested_action = np.r_[requested_continuous, requested_gripper]
            adapted = adapter.adapt(spec.denormalize(requested_action))
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            next_observation, _env_reward, _env_terminated, _env_truncated, _env_info = env.step(adapted.joint_target)
            next_state_43 = np.r_[env.get_policy_observation(next_observation), np.float32(bool(next_observation["object_grasped"]))].astype(np.float32)
            if adapted.accepted:
                replay_continuous = np.asarray(adapted.normalized[:6], np.float32)
            else:
                # IK fallback holds the prior Cartesian target; zero delta is the
                # action that physically caused next_state.
                replay_continuous = np.zeros(6, np.float32)
            actual_gripper_normalized = float(spec.normalize(np.r_[np.zeros(6), adapted.joint_target[7]])[6])
            replay_gripper = float(actual_gripper_normalized < gripper_threshold)
            if (
                replay_continuous.shape != (6,)
                or not np.isfinite(replay_continuous).all()
                or np.any(np.abs(replay_continuous) > 1.0 + 1e-6)
                or replay_gripper not in (0.0, 1.0)
            ):
                raise FloatingPointError("actual executed policy action is invalid")
            if adapted.accepted and not np.allclose(
                    replay_continuous, adapted.normalized[:6], atol=1e-7):
                raise RuntimeError("replay action is not the adapter-executed action")
            episode_actual_close += int(replay_gripper)
            reward_step = reward_protocol.step(
                state_43, next_state_43,
                ik_failure=consecutive_ik >= config.max_consecutive_ik_failures,
                time_limit=episode_step >= config.max_steps,
            )
            next_state = np.r_[next_state_43, reward_protocol.tracker.current.astype(np.float32)].astype(np.float32)
            replay.append(
                state, replay_continuous, replay_gripper, reward_step.reward,
                next_state, reward_step.terminated, reward_step.truncated,
            )
            transition_rows.append({
                "obs": state, "continuous_action": replay_continuous,
                "gripper_action": replay_gripper, "reward": reward_step.reward,
                "next_obs": next_state, "terminated": reward_step.terminated,
                "truncated": reward_step.truncated,
                "episode_id": f"online_{episode_index:06d}", "step": episode_step - 1,
                "task_success": reward_step.task_success,
                "milestones": reward_step.milestones.astype(np.uint8),
                "termination_reason": reward_step.termination_reason,
                "gripper_state": "CLOSE" if replay_gripper else "OPEN",
                "object_grasped": bool(next_observation["object_grasped"]),
                "online_policy_step": online_step,
                "collection_step": collection_step,
                "collection_phase": collection_phase,
                "requested_continuous_action": requested_continuous,
                "requested_gripper_action": float(requested_close),
                "continuous_policy_std_mean": policy_std_mean,
                "continuous_effective_std_mean": effective_std_mean,
                "behavior_std_scale": behavior_std_scale,
                "release_retreat_state": release_retreat_state,
                "transport_done_state": transport_done_state,
                "retreat_target_distance": retreat_target_distance,
                "gripper_close_probability": close_probability,
                "action_clipped": bool(adapted.action_clipped),
                "fallback_used": bool(adapted.fallback_used),
                "rejection_reason": adapted.rejection_reason,
            })
            if episode_release_step is None and bool(reward_step.milestones[3]):
                episode_release_step = episode_step
            if episode_retreat_step is None and bool(reward_step.milestones[4]):
                episode_retreat_step = episode_step
            update: dict[str, Any] | None = None
            if collection_phase == "training":
                online_step += 1
                update = trainer.update(replay.sample(trainer.config.batch_size, trainer.generator))
                completed = len(episode_rows)
                update.update({
                    "online_environment_step": collection_step,
                    "online_awac_update_step": online_step,
                    "replay_size": len(replay), "offline_transition_count": replay.offline_count,
                    "online_transition_count": replay.online_count,
                    "sampled_online_count": replay.last_sample_online_count,
                    "sampled_online_fraction": replay.last_sample_online_count / trainer.config.batch_size,
                    "rollout_completed_episodes": completed,
                    "rollout_success_rate": (
                        float(np.mean([row["task_success"] for row in episode_rows])) if completed else 0.0
                    ),
                    "rollout_average_episode_return": (
                        float(np.mean([row["episode_return"] for row in episode_rows])) if completed else 0.0
                    ),
                    "requested_gripper_close": int(requested_close),
                    "actual_gripper_close": int(replay_gripper),
                    "continuous_policy_std_mean": policy_std_mean,
                    "continuous_effective_std_mean": effective_std_mean,
                    "behavior_std_scale": behavior_std_scale,
                    "release_retreat_state": int(release_retreat_state),
                    "retreat_target_distance": (
                        retreat_target_distance if np.isfinite(retreat_target_distance) else 0.0),
                    "gripper_close_probability": close_probability,
                })
                update_rows.append(update); metrics_stream.write(json.dumps(update) + "\n")
            episode_return += reward_step.reward; observation = next_observation
            if reward_step.terminated or reward_step.truncated:
                episode_record = {
                    "episode_id": f"online_{episode_index:06d}",
                    "environment_seed": TRAINING_SEED_START + episode_index,
                    "collection_end_step": collection_step,
                    "online_end_step": online_step, "episode_length": episode_step,
                    "completion_phase": collection_phase,
                    "episode_return": episode_return, "task_success": reward_step.task_success,
                    "milestones": reward_step.milestones.astype(int).tolist(),
                    "termination_reason": reward_step.termination_reason,
                    "requested_close_ratio": episode_requested_close / episode_step,
                    "actual_close_ratio": episode_actual_close / episode_step,
                    "release_step": episode_release_step,
                    "retreat_step": episode_retreat_step,
                    "release_to_retreat_steps": (
                        episode_retreat_step - episode_release_step
                        if episode_release_step is not None and episode_retreat_step is not None else None
                    ),
                }
                episode_rows.append(episode_record); episodes_stream.write(json.dumps(episode_record) + "\n")
                env.close(); env = None; adapter = None; observation = None; reward_protocol = None
            if collection_step % 100 == 0:
                metrics_stream.flush(); episodes_stream.flush()
            sanity_episodes = [
                row for row in episode_rows if row["completion_phase"] == "sanity"
            ]
            if collection_phase == "sanity" and len(sanity_episodes) == SANITY_EPISODES:
                sanity_complete_step = collection_step
                checkpoint(0, label=f"online_sanity_{SANITY_EPISODES:02d}ep")
                sanity_success = int(sum(row["task_success"] for row in sanity_episodes))
                long_post_release_timeout = int(sum(
                    row["termination_reason"] == "timeout"
                    and row["release_step"] is not None
                    and row["retreat_step"] is None
                    for row in sanity_episodes
                ))
                print(json.dumps({
                    "sanity_transitions": sanity_complete_step,
                    "optimizer_step": trainer.step,
                    "completed_episodes": len(sanity_episodes),
                    "success": sanity_success,
                    "long_post_release_timeout": long_post_release_timeout,
                }), flush=True)
                if sanity_success < 4 or long_post_release_timeout:
                    protection = {
                        "step": 0,
                        "reason": "5-episode milestone-conditioned behavior sanity failed",
                        "completed_episodes": len(sanity_episodes),
                        "success": sanity_success,
                        "long_post_release_timeout": long_post_release_timeout,
                    }
                    break
                if args.stop_after_updates == 0:
                    return
            if protection is not None:
                checkpoint(online_step, label=f"protective_stop_{online_step:05d}")
                break
            if (
                collection_phase == "training"
                and args.stop_after_updates is not None
                and online_step == args.stop_after_updates
                and online_step < ONLINE_UPDATES
            ):
                path = checkpoint(online_step, label=f"online_progress_{online_step:05d}")
                print(json.dumps({
                    "segment_complete": online_step,
                    "checkpoint": str(path),
                    "online_transitions": replay.online_count,
                    "optimizer_step": trainer.step,
                }), flush=True)
                return
            if collection_phase == "training" and online_step in EVALUATION_STEPS[1:]:
                path = checkpoint(online_step)
                print(f"evaluating online step {online_step}", flush=True)
                evaluation = evaluate_policy(HybridCheckpointPredictor(path), VALIDATION_SEEDS, reward_config)
                evaluations[str(online_step)] = evaluation
                release_retreat_evaluations[str(online_step)] = release_retreat_diagnostic(
                    trainer, diagnostic_validation, trainer.config.seed + trainer.step)
                (evaluation_dir / f"online_step_{online_step:05d}.json").write_text(json.dumps(evaluation, indent=2) + "\n")
                summary = evaluation_summary(evaluation)
                print(json.dumps({"step": online_step, **summary}), flush=True)
                assert update is not None
                numeric_bad = any(not np.isfinite(value) for value in update.values() if isinstance(value, (int, float)))
                q_bad = abs(update["q1_mean"]) > 1_000 or abs(update["q2_mean"]) > 1_000
                if numeric_bad or q_bad:
                    protection = {"step": online_step, "reason": "non-finite metric or Q explosion"}
                elif online_step > 0 and summary["success"] < 80:
                    protection = {"step": online_step, "reason": "validation success below 80/100", "summary": summary}
                elif online_step > 0 and summary["retreat"] < 80:
                    protection = {"step": online_step, "reason": "validation retreat structurally regressed", "summary": summary}
                elif online_step > 0 and summary["timeout"] > 15:
                    protection = {"step": online_step, "reason": "validation timeout structurally increased", "summary": summary}
                elif online_step > 0 and summary["place_to_success_conversion"] < .85:
                    protection = {"step": online_step, "reason": "Place-to-Success structurally regressed", "summary": summary}
                if protection is not None:
                    break
    except FloatingPointError as error:
        protection = {"step": len(update_rows), "reason": str(error)}
        checkpoint(len(update_rows), label=f"protective_stop_{len(update_rows):05d}")
    finally:
        metrics_stream.close(); episodes_stream.close()
        if env is not None: env.close()

    evaluated = {step: evaluation_summary(value) for step, value in evaluations.items()}
    best_step = max(evaluated, key=lambda value: (
        evaluated[value]["success"], evaluated[value]["place_to_success_conversion"], -int(value)
    ))
    # The mature Offline 20k policy remains the protected best unless Online is
    # strictly better on the primary fixed-seed Success metric.
    offline_success = evaluated["0"]["success"]
    if int(best_step) == 0 or evaluated[best_step]["success"] <= offline_success:
        best_step = "0"
        best_source = offline_checkpoint
    else:
        best_source = checkpoint_dir / f"online_step_{int(best_step):05d}.pt"
    shutil.copy2(best_source, run / "checkpoint_best.pt")
    # Fresh-seed model comparison is deliberately deferred until the complete
    # Online +10k curve has been reviewed.
    confirmation = None
    diagnostic_names = [key for key in update_rows[0] if key not in {
        "step", "online_environment_step", "online_awac_update_step",
        "replay_size", "offline_transition_count",
        "online_transition_count", "sampled_online_count", "rollout_completed_episodes",
        "requested_gripper_close", "actual_gripper_close",
    }] if update_rows else []
    sanity_episodes = [
        row for row in episode_rows if row["completion_phase"] == "sanity"
    ]
    training_episodes = [
        row for row in episode_rows if row["completion_phase"] == "training"
    ]
    released_episodes = [row for row in episode_rows if row["release_step"] is not None]
    release_to_retreat = [
        row["release_to_retreat_steps"] for row in released_episodes
        if row["release_to_retreat_steps"] is not None
    ]
    collection_behavior = {
        "continuous_policy_std_mean": float(np.mean([
            row["continuous_policy_std_mean"] for row in transition_rows
        ])),
        "continuous_effective_std_mean": float(np.mean([
            row["continuous_effective_std_mean"] for row in transition_rows
        ])),
        "exploration_std_scale": EXPLORATION_STD_SCALE,
        "gripper_close_probability_mean": float(np.mean([
            row["gripper_close_probability"] for row in transition_rows
        ])),
        "requested_close_ratio": float(np.mean([
            row["requested_gripper_action"] for row in transition_rows
        ])),
        "executed_close_ratio": float(np.mean([
            row["gripper_action"] for row in transition_rows
        ])),
    }
    release_retreat_transitions = [
        row for row in transition_rows if row["release_retreat_state"]
    ]
    pre_transport_transitions = [
        row for row in transition_rows if not row["transport_done_state"]
    ]
    post_transport_transitions = [
        row for row in transition_rows if row["transport_done_state"]
    ]
    retreat_distances = np.asarray([
        row["retreat_target_distance"] for row in release_retreat_transitions
    ], np.float64)
    release_retreat_behavior = {
        "pre_transport_stochastic_steps": len(pre_transport_transitions),
        "post_transport_deterministic_steps": len(post_transport_transitions),
        "post_transport_effective_std_max": float(max(
            (row["continuous_effective_std_mean"] for row in post_transport_transitions),
            default=0.0,
        )),
        "release_to_retreat_deterministic_steps": len(release_retreat_transitions),
        "continuous_action_mean": (
            np.mean([row["continuous_action"] for row in release_retreat_transitions], axis=0).astype(float).tolist()
            if release_retreat_transitions else None
        ),
        "dz_mean": float(np.mean([row["continuous_action"][2] for row in release_retreat_transitions]))
        if release_retreat_transitions else None,
        "dz_positive_ratio": float(np.mean([
            row["continuous_action"][2] > 0 for row in release_retreat_transitions
        ])) if release_retreat_transitions else None,
        "policy_std_mean": float(np.mean([
            row["continuous_policy_std_mean"] for row in release_retreat_transitions
        ])) if release_retreat_transitions else None,
        "effective_std_mean": float(np.mean([
            row["continuous_effective_std_mean"] for row in release_retreat_transitions
        ])) if release_retreat_transitions else None,
        "gripper_close_probability_mean": float(np.mean([
            row["gripper_close_probability"] for row in release_retreat_transitions
        ])) if release_retreat_transitions else None,
        "retreat_target_distance": {
            "mean": float(retreat_distances.mean()), "min": float(retreat_distances.min()),
            "p50": float(np.percentile(retreat_distances, 50)),
            "p90": float(np.percentile(retreat_distances, 90)),
        } if len(retreat_distances) else None,
    }
    def rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        grasped = int(sum(bool(row["milestones"][0]) for row in rows))
        lifted = int(sum(bool(row["milestones"][1]) for row in rows))
        transported = int(sum(bool(row["milestones"][2]) for row in rows))
        placed = int(sum(bool(row["milestones"][3]) for row in rows))
        retreated = int(sum(bool(row["milestones"][4]) for row in rows))
        success = int(sum(row["task_success"] for row in rows))
        return {
            "episodes": len(rows),
            "success": success,
            "success_rate": float(np.mean([row["task_success"] for row in rows])) if rows else 0.0,
            "grasp": grasped, "lift": lifted, "transport": transported,
            "place": placed, "release": placed, "retreat": retreated,
            "place_to_success": float(success / max(placed, 1)),
            "release_to_retreat": float(retreated / max(placed, 1)),
            "average_return": float(np.mean([row["episode_return"] for row in rows])) if rows else 0.0,
            "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
            "timeout": int(sum(row["termination_reason"] == "timeout" for row in rows)),
            "illegal_drop": int(sum(row["termination_reason"] == "illegal_drop" for row in rows)),
            "ik_failure": int(sum(row["termination_reason"] == "ik_failure_limit" for row in rows)),
        }
    diagnostics = {
        "updates": len(update_rows), "replay_initial": 135_237,
        "replay_final": replay.metadata(),
        "replay_final_online_fraction": replay.online_count / max(len(replay), 1),
        "metrics": {name: {
            "mean": float(np.mean([row[name] for row in update_rows])),
            "std": float(np.std([row[name] for row in update_rows])),
            "min": float(np.min([row[name] for row in update_rows])),
            "max": float(np.max([row[name] for row in update_rows])),
        } for name in diagnostic_names},
        "sanity_rollout": rollout_summary(sanity_episodes),
        "training_rollout": rollout_summary(training_episodes),
        "all_online_rollout": rollout_summary(episode_rows),
        "collection_behavior": collection_behavior,
        "release_retreat_behavior": release_retreat_behavior,
        "release_to_retreat": {
            "released_episodes": len(released_episodes),
            "retreated_episodes": len(release_to_retreat),
            "average_steps": float(np.mean(release_to_retreat)) if release_to_retreat else None,
            "post_release_timeout": int(sum(
                row["termination_reason"] == "timeout" for row in released_episodes
            )),
            "post_release_success": int(sum(row["task_success"] for row in released_episodes)),
        },
    }
    result = {
        "status": "protective_stop" if protection else "complete_20k",
        "protection": protection, "offline_checkpoint": base_metadata,
        "checkpoint_continuity": {
            "offline_optimizer_step": 20_000,
            "sanity_episodes": SANITY_EPISODES,
            "sanity_transitions": sanity_complete_step,
            "online_updates": len(update_rows),
            "final_optimizer_step": trainer.step, "optimizers_restored": True,
            "normalization_reestimated": False, "actor_or_critic_reinitialized": False,
        },
        "replay": replay.metadata(), "evaluations": evaluations,
        "evaluation_summary": evaluated,
        "release_retreat_evaluation_diagnostics": release_retreat_evaluations,
        "best_step": int(best_step),
        "best_checkpoint": str((run / "checkpoint_best.pt").resolve()),
        "fresh_seed_confirmation": confirmation,
        "diagnostics": diagnostics,
        "reward_reused_from": str(Path("src/mujoco_shared_control/awac/reward.py").resolve()),
        "online_awac_only": True,
        "algorithm_modifications": [],
        "behavior_policy_modifications": {
            "continuous_std_scale_before_transport": EXPLORATION_STD_SCALE,
            "continuous_std_scale_after_transport": 0.0,
            "gripper_sampling": "deterministic threshold",
        },
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (run / "final_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "run": str(run), "status": result["status"], "replay": result["replay"],
        "evaluations": evaluated, "best_step": result["best_step"],
    }, indent=2))


if __name__ == "__main__": main()
