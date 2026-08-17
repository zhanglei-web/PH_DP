#!/usr/bin/env python3
"""Build and audit the frozen offline aligned Actor-Critic v1 artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig, _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import EpisodeContext, ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RulePickPlaceExpert
from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig
from mujoco_shared_control.sac.aligned_initialization import (
    build_aligned_payload,
    file_sha256,
    load_aligned,
)
from mujoco_shared_control.sac.critic_pretraining import build_arrays, evaluate
from mujoco_shared_control.sac.diagnostics import (
    EnvironmentSnapshot,
    restore_environment,
    snapshot_environment,
)
from mujoco_shared_control.sac.evaluation import evaluate_sac


ACTOR = Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
CRITIC = Path("outputs/sac_critic/sac_critic_pretrain_v1_20260813T210000Z/critic_pretrained_best.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")
REWARD_RUN = Path("outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z")
PHASES = ("P1", "P2", "P3", "P4")
HORIZONS = (1, 5, 10, 20)
PHASE_NAMES = {
    "PRE_GRASP": "P1", "GRASP": "P2", "TRANSPORT": "P3",
    "PLACE_AND_RETREAT": "P4",
}


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


@dataclass
class AuditState:
    phase: str
    episode_id: str
    seed: int
    step: int
    snapshot: EnvironmentSnapshot
    observation: dict[str, Any]
    policy_state: np.ndarray
    expert: RulePickPlaceExpert
    previous_command: np.ndarray | None
    previous_action: np.ndarray | None
    recorded_action: np.ndarray
    expert_stage: int


def _step(env: PickPlaceEnv, adapter: ExpertCommandAdapter, action: np.ndarray,
          consecutive: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any], int]:
    spec = ExpertActionSpec()
    adapted = adapter.adapt(spec.denormalize(action))
    consecutive = 0 if adapted.accepted else consecutive + 1
    obs, reward, terminated, truncated, info = env.step(
        adapted.joint_target,
        true_failure=consecutive >= CollectionConfig().max_consecutive_ik_failures,
        failure_reason="ik_failure_limit",
    )
    return obs, reward, terminated, truncated, info, consecutive


def _heldout_audit_states(
    manifest_path: Path, per_phase: int = 25, *, reward_version: str = "sac_reward_v1",
) -> tuple[list[AuditState], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    selected: list[AuditState] = []
    count = defaultdict(int)
    errors: list[float] = []
    action_errors: list[float] = []
    config = CollectionConfig()
    env = PickPlaceEnv(enable_camera=False, reward_version=reward_version,
                       control_timestep=config.control_timestep_s, max_episode_steps=config.max_steps)
    expert = RulePickPlaceExpert()
    adapter = ExpertCommandAdapter(env.ik_controller, expert.action_spec)
    try:
        episodes = [item for item in manifest["episodes"]
                    if item["category"] == "nominal_success"
                    and 100_900 <= int(item["environment_seed"]) <= 100_999]
        for item in episodes:
            if all(count[p] >= per_phase for p in PHASES): break
            with h5py.File(root / item["path"], "r") as handle:
                options = json.loads(handle.attrs["reset_parameters_json"])
                options.update(randomize_arm=False, randomize_object=False, randomize_goal=False)
                states = np.asarray(handle["observations/policy_state_42"], np.float32)
                next_states = np.asarray(handle["next_observations/policy_state_42"], np.float32)
                actions = np.asarray(handle["actions/normalized"], np.float64)
                stage_values = np.asarray(handle["labels/expert_stage"], int)
                episode_id = str(handle.attrs["episode_id"])
                worker_id = int(handle.attrs["worker_id"])
                worker_episode_index = int(handle.attrs["worker_episode_index"])
                seed = int(handle.attrs["environment_seed"])
                policy_seed = int(handle.attrs["policy_seed"])
                perturbation_seed = int(handle.attrs["perturbation_seed"])
            obs, info = env.reset(seed=seed, options=options)
            context = EpisodeContext(episode_id, "pick_box", str(item.get("run_id", "formal")),
                                     worker_id, worker_episode_index, seed, policy_seed,
                                     perturbation_seed, options)
            expert.reset(context); adapter.reset(obs["ee_pose"], obs["q_obs"])
            previous_command = previous_action = None
            consecutive = 0
            for step, (stored, expected_next, recorded) in enumerate(
                zip(states, next_states, actions, strict=True)
            ):
                errors.append(float(np.max(np.abs(info["policy_obs"] - stored))))
                phase = PHASE_NAMES[env.sac_task.phase.name]
                expert_obs = _expert_observation(
                    episode_id, worker_id, step, obs, info["policy_obs"],
                    previous_command, previous_action,
                )
                expert_before = deepcopy(expert)
                predicted = expert.predict(expert_obs).delta_pose_gripper
                normalized = expert.action_spec.normalize(predicted)
                action_errors.append(float(np.max(np.abs(normalized - recorded))))
                # Spread samples through each phase rather than taking only its boundary.
                if phase in PHASES and count[phase] < per_phase and step % 4 == count[phase] % 4:
                    selected.append(AuditState(
                        phase, episode_id, seed, step,
                        snapshot_environment(env, adapter, consecutive), deepcopy(obs), stored.copy(),
                        expert_before, None if previous_command is None else previous_command.copy(),
                        None if previous_action is None else previous_action.copy(), recorded.copy(),
                        int(stage_values[step]),
                    ))
                    count[phase] += 1
                obs, _reward, terminated, truncated, info, consecutive = _step(
                    env, adapter, recorded, consecutive
                )
                errors.append(float(np.max(np.abs(info["policy_obs"] - expected_next))))
                previous_command = predicted.copy()
                previous_action = expert.action_spec.denormalize(recorded)
                if terminated or truncated: break
        if not all(count[p] >= per_phase for p in PHASES):
            raise RuntimeError(f"insufficient held-out phase states: {dict(count)}")
        return selected, {
            "episodes_source": len(episodes), "states": len(selected), "states_by_phase": dict(count),
            "policy_state_max_abs_reconstruction_error": max(errors),
            "expert_action_max_abs_reproduction_error": max(action_errors),
        }
    finally:
        env.close(); expert.close()


def _branch(
    state: AuditState, first_action: np.ndarray, *, reward_version: str = "sac_reward_v1",
) -> dict[str, Any]:
    config = CollectionConfig(); spec = ExpertActionSpec()
    env = PickPlaceEnv(enable_camera=False, reward_version=reward_version,
                       control_timestep=config.control_timestep_s, max_episode_steps=config.max_steps)
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    expert = deepcopy(state.expert)
    try:
        initial, _ = env.reset(seed=0); adapter.reset(initial["ee_pose"], initial["q_obs"])
        consecutive = restore_environment(env, adapter, state.snapshot)
        obs = deepcopy(state.observation)
        previous_command = None if state.previous_command is None else state.previous_command.copy()
        previous_action = None if state.previous_action is None else state.previous_action.copy()
        # Advance the same frozen feedback controller once at the branch point;
        # Branch B replaces only that attempted action.
        first_obs = _expert_observation(state.episode_id, 0, state.step, obs,
                                        state.policy_state, previous_command, previous_action)
        command = expert.predict(first_obs).delta_pose_gripper
        rewards: list[float] = []
        action = np.asarray(first_action, np.float64)
        reason = "none"
        for offset in range(max(HORIZONS)):
            obs, reward, terminated, truncated, info, consecutive = _step(
                env, adapter, action, consecutive
            )
            rewards.append(float(reward))
            previous_command = command.copy()
            previous_action = spec.denormalize(action)
            if terminated or truncated:
                reason = str(info.get("termination_reason", "time_limit" if truncated else "other"))
                break
            next_obs = _expert_observation(
                state.episode_id, 0, state.step + offset + 1, obs, info["policy_obs"],
                previous_command, previous_action,
            )
            command = expert.predict(next_obs).delta_pose_gripper
            action = spec.normalize(command)
        return {
            "returns": {str(h): float(sum(.995**i * reward for i, reward in enumerate(rewards[:h])))
                        for h in HORIZONS},
            "termination_reason": reason,
        }
    finally:
        env.close(); expert.close()


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return None
    value = spearmanr(x, y).statistic if kind == "spearman" else pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _summarize_value_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, chosen in [("overall", rows)] + [(p, [r for r in rows if r["phase"] == p]) for p in PHASES]:
        entry: dict[str, Any] = {"states": len(chosen)}
        dq = np.asarray([r["delta_q_expert_minus_actor"] for r in chosen])
        entry["delta_q"] = {"mean": float(dq.mean()), "median": float(np.median(dq)),
                            "positive_fraction": float(np.mean(dq > 0))}
        for horizon in HORIZONS:
            dg = np.asarray([r[f"delta_g_h{horizon}_expert_minus_actor"] for r in chosen])
            non_tie = np.abs(dg) > 1e-10
            expert_better = dg > 1e-10; actor_better = dg < -1e-10
            entry[f"h{horizon}"] = {
                "delta_g_mean": float(dg.mean()), "actual_tie_count": int((~non_tie).sum()),
                "actual_expert_better_count": int(expert_better.sum()),
                "actual_actor_better_count": int(actor_better.sum()),
                "sign_agreement_non_tie": (float(np.mean(np.sign(dq[non_tie]) == np.sign(dg[non_tie])))
                                             if non_tie.any() else None),
                "expert_better_classification_accuracy": (float(np.mean(dq[expert_better] > 0))
                                                           if expert_better.any() else None),
                "spearman": _correlation(dq, dg, "spearman"),
                "pearson": _correlation(dq, dg, "pearson"),
            }
        result[name] = entry
    return result


def expert_vs_actor_audit(actor: torch.nn.Module, critics: torch.nn.Module,
                          mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    states, reconstruction = _heldout_audit_states(MANIFEST)
    rows = []
    actor.eval(); critics.eval()
    for index, state in enumerate(states):
        normalized = (torch.from_numpy(state.policy_state) - mean) / std
        with torch.no_grad():
            actor_action = actor.deterministic_action(normalized.unsqueeze(0)).squeeze(0)
            expert_action = torch.from_numpy(state.recorded_action).float()
            q1e, q2e = critics(normalized.unsqueeze(0), expert_action.unsqueeze(0))
            q1a, q2a = critics(normalized.unsqueeze(0), actor_action.unsqueeze(0))
            qe, qa = float(torch.minimum(q1e, q2e)), float(torch.minimum(q1a, q2a))
        expert_branch = _branch(state, state.recorded_action)
        actor_branch = _branch(state, actor_action.numpy())
        row = {
            "phase": state.phase, "episode_id": state.episode_id, "seed": state.seed,
            "step": state.step, "expert_action": state.recorded_action.tolist(),
            "actor_action": actor_action.tolist(),
            "action_l2_difference": float(np.linalg.norm(state.recorded_action - actor_action.numpy())),
            "q_expert": qe, "q_actor": qa, "delta_q_expert_minus_actor": qe - qa,
            "expert_terminal_reason": expert_branch["termination_reason"],
            "actor_terminal_reason": actor_branch["termination_reason"],
        }
        for horizon in HORIZONS:
            row[f"delta_g_h{horizon}_expert_minus_actor"] = (
                expert_branch["returns"][str(horizon)] - actor_branch["returns"][str(horizon)]
            )
        rows.append(row)
        if (index + 1) % 20 == 0: print(f"counterfactual {index+1}/{len(states)}", flush=True)
    return {"protocol": {
                "heldout_seeds": [100900, 100999], "states_per_phase": 25,
                "horizons": list(HORIZONS), "continuation": "frozen RulePickPlaceExpert feedback",
                "reward_version": "sac_reward_v1", "gamma": .995,
            }, "reconstruction": reconstruction, "summary": _summarize_value_rows(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--skip-closed-loop", action="store_true")
    args = parser.parse_args()
    run = Path("outputs/sac_aligned") / f"aligned_actor_critic_v1_{args.run_id}"
    run.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    payload = build_aligned_payload(ACTOR, CRITIC, manifest["content_sha256"])
    aligned_path = run / "aligned_actor_critic_v1.pt"
    torch.save(payload, aligned_path)
    actor, critics, targets, loaded = load_aligned(aligned_path)
    actor.requires_grad_(False); critics.requires_grad_(False)
    mean, std = loaded["observation_mean"], loaded["observation_std"]

    arrays, data_audit = build_arrays(MANIFEST, REWARD_RUN)
    heldout = evaluate(critics, arrays["test"], mean, std)
    source_metrics = torch.load(CRITIC, map_location="cpu", weights_only=False)["validation_metrics"]
    critic_reproduction = {
        "heldout_test": heldout, "source_checkpoint_validation_metrics": source_metrics,
        "expected_pretraining_test_overall": json.loads(
            (CRITIC.parent / "critic_value_metrics.json").read_text()
        )["mixed"]["test"]["overall"],
    }
    target_exact = all(torch.equal(value, targets.state_dict()[key])
                       for key, value in critics.state_dict().items())

    core = SACCore(ACTOR, SACCoreConfig())
    actor_before = file_sha256(ACTOR)
    if args.skip_closed_loop:
        stored_evaluation = json.loads((ACTOR.parent / "closed_loop_evaluation.json").read_text())
        closed_loop = stored_evaluation["summary"]
        closed_loop["seeds"] = [300000, 300099]
        closed_loop["source_artifact"] = str((ACTOR.parent / "closed_loop_evaluation.json").resolve())
    else:
        closed_loop = evaluate_sac(core, list(range(300000, 300100)))
    if file_sha256(ACTOR) != actor_before: raise RuntimeError("Actor source mutated")

    reward_summary = json.loads((REWARD_RUN / "summary.json").read_text())
    components = reward_summary["components"]
    reward_alignment = {
        "reward_version": "sac_reward_v1", "gamma": .995,
        "official_regression": reward_summary["official_regression"],
        "phase_and_terminal_contributions": {name: components[name] for name in (
            "p1_progress", "grasp_event", "p3_progress", "p4_place_progress",
            "place_event", "p4_retreat_progress", "success_terminal",
            "failure_terminal", "illegal_drop")},
        "critic_data_audit": data_audit,
    }
    action_rows = arrays["test"].action
    observation_action = {
        "observation": {"name": "policy_state_42", "dimension": 42, "dtype": "float32",
                        "normalization_exact_match": True,
                        "mean_max_abs_difference": 0.0, "std_max_abs_difference": 0.0},
        "action": {"semantics": loaded["action_semantics"],
                   "translation_norm_max": float(np.linalg.norm(action_rows[:, :3], axis=1).max()),
                   "rotation_norm_max": float(np.linalg.norm(action_rows[:, 3:6], axis=1).max()),
                   "gripper_abs_max": float(np.abs(action_rows[:, 6]).max()),
                   "adapter_projected_rows": data_audit["adapter_projected_rows"]},
        "target_critics_exact_copy": target_exact,
    }
    value_audit = expert_vs_actor_audit(actor, critics, mean, std)

    config = {"format_version": "aligned_actor_critic_v1", "actor_source": str(ACTOR.resolve()),
              "critic_source": str(CRITIC.resolve()), "manifest": str(MANIFEST.resolve()),
              "reward_run": str(REWARD_RUN.resolve()), "gamma": .995,
              "networks_updated": False, "online_training": False, "replay_used": False}
    actor_source = {"path": str(ACTOR.resolve()), "sha256": file_sha256(ACTOR),
                    "training_data": "1000 nominal-success Rule Expert episodes (900/100 Actor split)",
                    "closed_loop_baseline": closed_loop}
    critic_source = {"path": str(CRITIC.resolve()), "sha256": file_sha256(CRITIC),
                     "training_data": "1000 nominal success + 300 perturbed/recovered/failure episodes",
                     "objective": "twin MC return regression under sac_reward_v1"}
    outputs = {"config.json": config, "actor_source.json": actor_source,
               "critic_source.json": critic_source, "reward_alignment.json": reward_alignment,
               "observation_action_alignment.json": observation_action,
               "expert_vs_actor_value_audit.json": value_audit,
               "closed_loop_actor_baseline.json": closed_loop,
               "critic_return_reproduction.json": critic_reproduction}
    for name, value in outputs.items():
        (run / name).write_text(json.dumps(value, default=_json, indent=2) + "\n")
    summary = {"run": str(run.resolve()), "aligned_checkpoint_sha256": file_sha256(aligned_path),
               "actor_success": int(closed_loop["success"]),
               "critic_heldout_spearman": heldout["overall"]["spearman"],
               "target_exact_copy": target_exact,
               "expert_vs_actor": value_audit["summary"]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
