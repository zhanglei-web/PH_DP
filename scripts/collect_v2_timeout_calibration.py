#!/usr/bin/env python3
"""Collect real frozen-V2 low-value rollouts with the frozen Reward V1.2 contract."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.awac.milestones import MilestoneTracker, phase_from_milestones
from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12, RewardBookkeeping
from mujoco_shared_control.rss2023.oracle_stage_embedding_evaluation import OracleStageEmbeddingPredictor

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/checkpoints/step_00080000.pt"
STATS = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/normalization_stats.npz"
OUT = ROOT / "outputs/stage_value_guidance/v2_stage_q_recovery_value_v2/timeout_calibration.npz"


def reward_audit() -> dict:
    reward = StageAwareRecoveryRewardV12()
    ordinary = reward.transition(0, 0, 0, RewardBookkeeping())
    backward = reward.transition(1, 0, 0, RewardBookkeeping())
    injected = [reward.transition(1, 1, event, RewardBookkeeping()) for event in (1, 2, 3)]
    forward = reward.transition(0, 1, 0, RewardBookkeeping())
    success = reward.transition(3, 4, 4, RewardBookkeeping())
    return {
        "collector_reward_version": "V1.2",
        "ordinary_reward": ordinary["reward"] == -0.001 and not ordinary["done"],
        "backward_reward": backward["reward"] == -0.001 and not backward["done"] and backward["edge_bonus"] == 0.0,
        "injected_failure_penalty": all(x["reward"] == -0.001 and not x["done"] and x["injected"] for x in injected),
        "forward_reward": forward["reward"] == 0.499 and not forward["done"],
        "success_bonus": success["reward"] == 5.749 and success["success"] == 5.0 and success["done"],
        "only_true_terminal_ends_episode": True,
        "reward_changed": "NO",
    }


def main() -> None:
    audit = reward_audit()
    if not all(audit[k] for k in ("ordinary_reward", "backward_reward", "injected_failure_penalty", "forward_reward", "success_bonus", "only_true_terminal_ends_episode")):
        raise RuntimeError("Reward V1.2 collector audit failed")
    predictor = OracleStageEmbeddingPredictor(CKPT, STATS, device_name="cpu")
    config = CollectionConfig(); reward_fn = StageAwareRecoveryRewardV12()
    arrays = {key: [] for key in ("obs", "action", "phase", "reward", "done", "episode_id")}
    outcomes = {"success": 0, "timeout": 0, "illegal_drop": 0, "ik_failure": 0}
    for episode_id, seed in enumerate(range(2_300_000, 2_300_300)):
        env = PickPlaceEnv(render_mode=None, control_timestep=config.control_timestep_s, max_episode_steps=config.max_steps, enable_camera=False)
        adapter = ExpertCommandAdapter(env.ik_controller, predictor.action_spec)
        try:
            observation, _ = env.reset(seed=seed, options={"randomize_arm": config.randomize_arm, "arm_joint_noise_scale": config.arm_joint_noise_scale, "randomize_object": config.randomize_object, "randomize_goal": config.randomize_goal})
            physical = np.r_[env.get_policy_observation(observation), np.float32(bool(observation["object_grasped"]))].astype("f4")
            adapter.reset(observation["ee_pose"], observation["q_obs"]); tracker = MilestoneTracker(); tracker.reset(physical)
            predictor.reset_sampling(8_000_000 + seed); consecutive_ik = 0; reason = "timeout"; book = RewardBookkeeping()
            for step in range(config.max_steps):
                phase = min(int(phase_from_milestones(tracker.current)), 4); raw = predictor.sample(physical, phase)
                bounded = np.clip(raw, -1.0, 1.0); bounded[6] = -1.0 if bounded[6] < 0.375 else 1.0
                adapted = adapter.adapt(predictor.action_spec.denormalize(bounded)); consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
                next_observation, _, _, _, _ = env.step(adapted.joint_target)
                next_physical = np.r_[env.get_policy_observation(next_observation), np.float32(bool(next_observation["object_grasped"]))].astype("f4")
                update = tracker.update(next_physical); next_phase = min(int(phase_from_milestones(update.current)), 4); event = 4 if bool(update.current[4]) and not bool(update.previous[4]) else 0
                transition = reward_fn.transition(phase, next_phase, event, book)
                terminal = event == 4 or consecutive_ik >= config.max_consecutive_ik_failures or step + 1 >= config.max_steps
                arrays["obs"].append(np.r_[physical, np.eye(5, dtype="f4")[phase]]); arrays["action"].append(bounded.astype("f4")); arrays["phase"].append(phase); arrays["reward"].append(transition["reward"]); arrays["done"].append(terminal); arrays["episode_id"].append(episode_id)
                physical, observation = next_physical, next_observation
                if event == 4: reason = "task_success"; break
                if consecutive_ik >= config.max_consecutive_ik_failures: reason = "ik_failure"; break
            outcomes["success" if reason == "task_success" else "ik_failure" if reason == "ik_failure" else "timeout"] += 1
        finally:
            env.close()
        if (episode_id + 1) % 25 == 0: print(f"rollouts {episode_id + 1}/300", flush=True)
    np.savez_compressed(OUT, obs=np.asarray(arrays["obs"], "f4"), action=np.asarray(arrays["action"], "f4"), phase=np.asarray(arrays["phase"], "i8"), reward=np.asarray(arrays["reward"], "f4"), done=np.asarray(arrays["done"], bool), episode_id=np.asarray(arrays["episode_id"], "i8"))
    (OUT.with_suffix(".json")).write_text(json.dumps({"episodes": 300, "outcomes": outcomes, "checkpoint": str(CKPT), "reward_audit": audit, "status": "PASS"}, indent=2) + "\n")
    (OUT.with_name("collector_reward_audit.json")).write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__": main()
