#!/usr/bin/env python3
"""Read-only exploration-scale and local Critic-guidance audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig
from mujoco_shared_control.sac.diagnostics import (
    EnvironmentSnapshot, constrained_local_step, q_action_gradient,
    restore_environment, sample_with_log_std_override, snapshot_environment,
)
from mujoco_shared_control.tasks.sac_reward import SACPhase


CHECKPOINT = Path(
    "outputs/sac_training/sac_v2_final_safe_trust_20260813T100000Z/checkpoints/latest.pt"
)
MANIFEST = Path("manifests/rule_expert_v1_formal.json")
SEEDS = list(range(420_000, 420_100))
LOG_STDS: tuple[float | None, ...] = (None, -5.0, -4.5, -4.0, -3.5, -3.0)
PHASES = tuple(phase.name for phase in SACPhase)
MILESTONES = ("grasped", "lifted", "transported", "released", "retreated")


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_core(path: Path) -> tuple[SACCore, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["global_env_steps"]) != 30_000:
        raise ValueError("audit requires the validated 30k checkpoint")
    core = SACCore(payload["actor_artifact"], SACCoreConfig(**payload["core_config"]))
    core.load_training_state_dict(payload["core"])
    core.actor.eval(); core.actor.requires_grad_(False)
    core.critics.eval(); core.critics.requires_grad_(False)
    core.target_critics.eval(); core.target_critics.requires_grad_(False)
    return core, payload


def _act(core: SACCore, state: np.ndarray, log_std: float | None,
         generator: torch.Generator) -> np.ndarray:
    if log_std is None:
        return core.select_action(state, deterministic=True)
    normalized = core.normalize_observation(torch.as_tensor(state)).unsqueeze(0)
    return sample_with_log_std_override(core.actor, normalized, log_std, generator).squeeze(0).numpy()


def _rollout(core: SACCore, seed: int, log_std: float | None) -> dict[str, Any]:
    config, spec = CollectionConfig(), ExpertActionSpec(**core.action_spec)
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1",
                       control_timestep=config.control_timestep_s,
                       max_episode_steps=config.max_steps)
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    # Common random numbers across every std setting: a paired seed changes
    # only the scale, never the direction/sequence of Gaussian perturbations.
    generator = torch.Generator().manual_seed(seed + 9_000_000)
    try:
        obs, info = env.reset(seed=seed, options={
            "randomize_arm": config.randomize_arm,
            "arm_joint_noise_scale": config.arm_joint_noise_scale,
            "randomize_object": config.randomize_object,
            "randomize_goal": config.randomize_goal,
        })
        adapter.reset(obs["ee_pose"], obs["q_obs"])
        initial_z = float(obs["object_pose"][2, 3]); milestones = np.zeros(5, bool)
        episode_return = discounted = 0.0; consecutive_ik = 0
        reason = "time_limit"; phases: list[str] = []; fallback_count = 0
        for step in range(config.max_steps):
            phase_before = env.sac_task.phase.name
            action = _act(core, info["policy_obs"], log_std, generator)
            adapted = adapter.adapt(spec.denormalize(action))
            fallback_count += int(adapted.fallback_used)
            consecutive_ik = 0 if adapted.accepted else consecutive_ik + 1
            safety = consecutive_ik >= config.max_consecutive_ik_failures
            next_obs, reward, terminated, truncated, next_info = env.step(
                adapted.joint_target, true_failure=safety,
                failure_reason="ik_failure_limit",
            )
            phases.append(phase_before); episode_return += reward
            discounted += core.config.gamma ** step * reward
            grasped = bool(next_obs["object_grasped"])
            obj, goal, ee = (next_obs[name][:3, 3] for name in ("object_pose", "goal_pose", "ee_pose"))
            milestones[0] |= grasped
            milestones[1] |= bool(milestones[0] and grasped and obj[2] - initial_z >= .10)
            milestones[2] |= bool(milestones[1] and grasped and np.linalg.norm(obj[:2]-goal[:2]) < .055)
            milestones[3] |= bool(next_info.get("successful_release", False))
            milestones[4] |= bool(milestones[3] and np.linalg.norm(ee-(goal+[0, 0, .16])) <= .008)
            obs, info = next_obs, next_info
            if terminated or truncated:
                reason = next_info.get("termination_reason", "time_limit" if truncated else "other_failure")
                break
        return {
            "seed": seed, "success": reason == "task_success", "termination_reason": reason,
            "episode_return": episode_return, "discounted_return": discounted,
            "episode_length": step + 1, "last_phase": phases[-1],
            "ik_fallback_count": fallback_count,
            **{name: bool(milestones[i]) for i, name in enumerate(MILESTONES)},
        }
    finally:
        env.close()


def _summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows), "success": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "milestone_rates": {m: float(np.mean([r[m] for r in rows])) for m in MILESTONES},
        "termination_reason_counts": dict(Counter(r["termination_reason"] for r in rows)),
        "mean_return": float(np.mean([r["episode_return"] for r in rows])),
        "median_return": float(np.median([r["episode_return"] for r in rows])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])),
        "rows": rows,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {"mean": float(values.mean()), **{
        f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)
    }, "max": float(values.max())}


def exploration_audit(core: SACCore) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    settings: dict[str, Any] = {}
    for value in LOG_STDS:
        name = "deterministic" if value is None else f"log_std_{value:g}"
        rows = [_rollout(core, seed, value) for seed in SEEDS]
        settings[name] = {"log_std": value, "std": 0.0 if value is None else float(np.exp(value)),
                          **_summarize_rollouts(rows)}
        print(f"exploration {name}: {settings[name]['success']}/100", flush=True)
    reference = {row["seed"]: row for row in settings["deterministic"]["rows"]}
    paired, phase_failures = {}, {}
    for name, result in settings.items():
        if name == "deterministic": continue
        counts = Counter(); failures = Counter()
        for row in result["rows"]:
            det = reference[row["seed"]]["success"]; sto = row["success"]
            counts[("det_success" if det else "det_failure") + "__" +
                   ("stochastic_success" if sto else "stochastic_failure")] += 1
            if det and not sto:
                failures[row["last_phase"]] += 1
                failures["reason:" + row["termination_reason"]] += 1
        paired[name] = dict(counts); phase_failures[name] = dict(failures)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    states = torch.as_tensor(checkpoint["replay"]["observation"][:4096])
    normalized = core.normalize_observation(states)
    with torch.no_grad(): deterministic = core.actor.deterministic_action(normalized)
    perturbation = {}
    for value in LOG_STDS[1:]:
        gen = torch.Generator().manual_seed(86_000 + int((value + 6) * 1000))
        samples = torch.cat([sample_with_log_std_override(core.actor, normalized, value, gen)
                             for _ in range(4)])
        det = deterministic.repeat(4, 1); delta = samples - det
        xyz = torch.linalg.vector_norm(delta[:, :3], dim=-1).numpy()
        rot = torch.linalg.vector_norm(delta[:, 3:6], dim=-1).numpy()
        grip = delta[:, 6].abs().numpy()
        perturbation[f"log_std_{value:g}"] = {
            "samples": len(samples), "translation_normalized_l2": _quantiles(xyz),
            "translation_physical_mm": _quantiles(xyz * 25.0),
            "rotation_normalized_l2": _quantiles(rot),
            "rotation_physical_rad": _quantiles(rot * .1),
            "gripper_normalized_abs": _quantiles(grip),
            "gripper_physical_mm": _quantiles(grip * 40.0),
        }
    return settings, paired, {"phase_failures": phase_failures, "action_perturbation": perturbation}


def _phase_from_formal(stage: int) -> str:
    return {0: "PRE_GRASP", 1: "GRASP", 2: "TRANSPORT", 3: "PLACE_AND_RETREAT"}.get(stage, "UNKNOWN")


def _formal_states(limit: int = 500) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    dataset = ManifestActorDataset(MANIFEST, "train")
    for entry in dataset.entries:
        if all(len(grouped[p]) >= limit for p in PHASES): break
        with h5py.File(entry.path, "r") as episode:
            states = np.asarray(episode["observations/policy_state_42"][:], np.float32)
            stages = np.asarray(episode["labels/stage"][:], int)
        for state, stage in zip(states, stages, strict=True):
            phase = _phase_from_formal(int(stage))
            if phase in PHASES and len(grouped[phase]) < limit: grouped[phase].append(state)
    return {phase: np.asarray(grouped[phase], np.float32) for phase in PHASES}


def _step(env: PickPlaceEnv, adapter: ExpertCommandAdapter, spec: ExpertActionSpec,
          action: np.ndarray, consecutive: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any], int]:
    adapted = adapter.adapt(spec.denormalize(action))
    consecutive = 0 if adapted.accepted else consecutive + 1
    obs, reward, term, trunc, info = env.step(
        adapted.joint_target, true_failure=consecutive >= CollectionConfig().max_consecutive_ik_failures,
        failure_reason="ik_failure_limit",
    )
    return obs, reward, term, trunc, info, consecutive


def _reconstruct_online(payload: dict[str, Any], per_phase: int = 500,
                        snapshots_per_phase: int = 12) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[EnvironmentSnapshot, np.ndarray, np.ndarray | None]]], dict[str, Any]]:
    replay = payload["replay"]; states, actions = replay["observation"], replay["action"]
    terminated, truncated = replay["terminated"][:, 0], replay["truncated"][:, 0]
    # The action spec is immutable and audited; construct explicitly because the
    # training core state intentionally contains weights, not metadata.
    spec = ExpertActionSpec()
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1",
                       control_timestep=.05, max_episode_steps=500)
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    snaps: dict[str, list[tuple[EnvironmentSnapshot, np.ndarray, np.ndarray | None]]] = defaultdict(list)
    errors: list[float] = []; seed = int(payload["protocol"]["training_seed_start"]); consecutive = 0
    try:
        obs, info = env.reset(seed=seed); adapter.reset(obs["ee_pose"], obs["q_obs"])
        for index, (stored_state, action) in enumerate(zip(states, actions, strict=True)):
            error = float(np.max(np.abs(info["policy_obs"] - stored_state))); errors.append(error)
            phase = env.sac_task.phase.name
            if len(grouped[phase]) < per_phase: grouped[phase].append(np.asarray(stored_state))
            if len(snaps[phase]) < snapshots_per_phase:
                snaps[phase].append((snapshot_environment(env, adapter, consecutive), np.asarray(stored_state), None))
            obs, _r, term, trunc, info, consecutive = _step(env, adapter, spec, action, consecutive)
            if bool(terminated[index]) != term or bool(truncated[index]) != trunc:
                raise RuntimeError(f"online terminal reconstruction mismatch at replay row {index}")
            if term or trunc:
                seed += 1; obs, info = env.reset(seed=seed)
                adapter.reset(obs["ee_pose"], obs["q_obs"]); consecutive = 0
        if max(errors) > 5e-5:
            raise RuntimeError(f"online Replay reconstruction max error {max(errors)}")
        return ({p: np.asarray(grouped[p], np.float32) for p in PHASES}, snaps,
                {"rows_replayed": len(states), "policy_state_max_abs_error": max(errors),
                 "policy_state_mean_abs_error": float(np.mean(errors))})
    except Exception:
        env.close(); raise
    # Keep env alive: snapshots refer to values only and can be restored in a new
    # identical model, but model-specific state indices are identical.
    finally:
        env.close()


def _reconstruct_formal(
    per_phase: int = 500, snapshots_per_phase: int = 12,
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[EnvironmentSnapshot, np.ndarray]]], dict[str, Any]]:
    """Replay frozen nominal episodes exactly; never infer physics from state_42."""
    dataset = ManifestActorDataset(MANIFEST, "train")
    spec = ExpertActionSpec(); grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    snaps: dict[str, list[tuple[EnvironmentSnapshot, np.ndarray, np.ndarray | None]]] = defaultdict(list)
    errors: list[float] = []; rows = episodes = 0
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1",
                       control_timestep=.05, max_episode_steps=500)
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    try:
        for entry in dataset.entries:
            if (all(len(grouped[p]) >= per_phase for p in PHASES)
                    and all(len(snaps[p]) >= snapshots_per_phase for p in PHASES)):
                break
            with h5py.File(entry.path, "r") as episode:
                options = json.loads(episode.attrs["reset_parameters_json"])
                # Persisted positions are authoritative.  Disabling random draws
                # makes reconstruction independent of RNG implementation details.
                options.update(randomize_arm=False, randomize_object=False, randomize_goal=False)
                seed = int(episode.attrs["environment_seed"])
                states = np.asarray(episode["observations/policy_state_42"][:], np.float32)
                next_states = np.asarray(episode["next_observations/policy_state_42"][:], np.float32)
                # Preserve the recorded float64 command exactly; converting to
                # float32 is enough to alter a long closed-loop reconstruction.
                actions = np.asarray(episode["actions/normalized"][:], np.float64)
            obs, info = env.reset(seed=seed, options=options)
            adapter.reset(obs["ee_pose"], obs["q_obs"]); consecutive = 0; episodes += 1
            for stored, expected_next, action in zip(states, next_states, actions, strict=True):
                error = float(np.max(np.abs(info["policy_obs"] - stored))); errors.append(error)
                if error > 5e-5:
                    raise RuntimeError(f"formal reconstruction error {error} in {entry.path}")
                phase = env.sac_task.phase.name
                if len(grouped[phase]) < per_phase: grouped[phase].append(stored)
                if len(snaps[phase]) < snapshots_per_phase:
                    # The exact recorded Rule Expert attempted policy action is
                    # the formal a_star.  Fallback deployment never replaces it.
                    snaps[phase].append((snapshot_environment(env, adapter, consecutive), stored, action.copy()))
                obs, _r, term, trunc, info, consecutive = _step(
                    env, adapter, spec, action, consecutive
                ); rows += 1
                next_error = float(np.max(np.abs(info["policy_obs"] - expected_next)))
                errors.append(next_error)
                if next_error > 5e-5:
                    raise RuntimeError(f"formal next-state reconstruction error {next_error}")
                # The legacy collector may retain settling rows after the new
                # frozen SAC-v1 terminal.  They are deliberately excluded.
                if term or trunc: break
        return ({p: np.asarray(grouped[p], np.float32) for p in PHASES}, snaps,
                {"episodes_replayed": episodes, "sac_semantic_rows_replayed": rows,
                 "policy_state_max_abs_error": max(errors),
                 "policy_state_mean_abs_error": float(np.mean(errors))})
    finally:
        env.close()


def _q_guidance_for_states(core: SACCore, states: dict[str, np.ndarray], source: str) -> dict[str, Any]:
    result = {}
    rng = torch.Generator().manual_seed(20260813 + (0 if source == "formal" else 1))
    for phase, values in states.items():
        if not len(values): result[phase] = {"samples": 0}; continue
        x = core.normalize_observation(torch.as_tensor(values))
        with torch.no_grad(): a0 = core.actor.deterministic_action(x); q10, q20 = core.critics(x, a0)
        grad = q_action_gradient(core, x, a0); q0 = torch.minimum(q10, q20)
        scales = {}
        for size in (1e-4, 3e-4, 1e-3):
            candidate = constrained_local_step(a0, grad, size)
            with torch.no_grad(): q1, q2 = core.critics(x, candidate)
            dq = (torch.minimum(q1, q2) - q0).squeeze(-1).numpy()
            scales[f"{size:g}"] = {**_quantiles(dq), "positive_fraction": float(np.mean(dq > 0))}
        candidates = []
        for _ in range(16):
            direction = torch.randn(a0.shape, generator=rng)
            candidates.append(constrained_local_step(a0, direction, 1e-3))
        stack = torch.stack(candidates, 1)
        with torch.no_grad():
            flat_x = x[:, None, :].expand(-1, 16, -1).reshape(-1, 42)
            q1, q2 = core.critics(flat_x, stack.reshape(-1, 7))
            q = torch.minimum(q1, q2).reshape(len(x), 16)
        best = q.max(1).values - q0.squeeze(-1)
        gnorm = torch.linalg.vector_norm(grad, dim=-1).numpy()
        result[phase] = {
            "samples": len(values), "gradient_norm": _quantiles(gnorm),
            "gradient_candidates_delta_q": scales,
            "random_best_of_16_delta_q": {**_quantiles(best.numpy()),
                                           "positive_fraction": float((best > 0).float().mean())},
        }
    return result


def _branch(core: SACCore, snapshot: EnvironmentSnapshot, first_action: np.ndarray,
            horizons: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, Any]:
    spec = ExpertActionSpec(**core.action_spec)
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1",
                       control_timestep=.05, max_episode_steps=500)
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    try:
        # Initialize private adapter arrays before restoring their exact values.
        obs, _ = env.reset(seed=0); adapter.reset(obs["ee_pose"], obs["q_obs"])
        consecutive = restore_environment(env, adapter, snapshot)
        rewards = []; terminal_reason = "none"; event_counts: Counter[str] = Counter()
        action = first_action
        for step in range(max(horizons)):
            _obs, reward, term, trunc, info, consecutive = _step(
                env, adapter, spec, action, consecutive
            )
            rewards.append(reward)
            for name, value in info["reward_components"].items():
                if value and name in ("grasp_event", "place_event", "illegal_drop",
                                      "success_terminal", "failure_terminal"):
                    event_counts[name] += 1
            if term or trunc:
                terminal_reason = info.get("termination_reason", "time_limit" if trunc else "other")
                break
            action = core.select_action(info["policy_obs"], deterministic=True)
        return {"returns": {
                    h: float(sum(core.config.gamma ** i * r for i, r in enumerate(rewards[:h])))
                    for h in horizons},
                "terminal_reason": terminal_reason, "events": dict(event_counts)}
    finally:
        env.close()


def _counterfactual(core: SACCore, snapshots: dict[str, list[tuple[EnvironmentSnapshot, np.ndarray, np.ndarray | None]]],
                    source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []; rng = torch.Generator().manual_seed(20260814)
    for phase, items in snapshots.items():
        for state_index, (snapshot, state, expert_action) in enumerate(items[:10]):
            x = core.normalize_observation(torch.as_tensor(state)).unsqueeze(0)
            with torch.no_grad(): a0 = core.actor.deterministic_action(x)
            if expert_action is not None:
                a0 = torch.as_tensor(expert_action, dtype=torch.float32).unsqueeze(0)
            grad = q_action_gradient(core, x, a0)
            candidates: list[tuple[str, torch.Tensor]] = []
            groups = {"translation": slice(0,3), "rotation": slice(3,6), "gripper": slice(6,7)}
            for group, indices in groups.items():
                masked = torch.zeros_like(grad); masked[:, indices] = grad[:, indices]
                for size in (1e-4, 3e-4, 1e-3):
                    candidates.append((f"gradient_{group}_{size:g}",
                                       constrained_local_step(a0, masked, size)))
            random_candidates=[]
            for group, indices in groups.items():
                for index in range(16):
                    direction=torch.zeros_like(a0)
                    direction[:,indices]=torch.randn(direction[:,indices].shape,generator=rng)
                    random_candidates.append((f"random_{group}_{index}",
                                              constrained_local_step(a0,direction,1e-3)))
            with torch.no_grad():
                q10, q20 = core.critics(x, a0); q0 = float(torch.minimum(q10, q20))
            candidates.extend(random_candidates)
            reference = _branch(core, snapshot, a0.squeeze(0).numpy())
            for name, candidate in candidates:
                with torch.no_grad():
                    q1, q2 = core.critics(x, candidate); q = float(torch.minimum(q1, q2))
                actual = _branch(core, snapshot, candidate.squeeze(0).numpy())
                rows.append({
                    "source": source, "phase": phase,
                    "state_index": state_index, "candidate": name,
                    "policy_state": np.asarray(state, dtype=np.float32).tolist(),
                    "reference_action": a0.squeeze(0).numpy().tolist(),
                    "candidate_action": candidate.squeeze(0).numpy().tolist(),
                    "delta_q": q - q0, "action_l2_delta": float(torch.linalg.vector_norm(candidate-a0)),
                    "reference_terminal_reason": reference["terminal_reason"],
                    "candidate_terminal_reason": actual["terminal_reason"],
                    "reference_events": reference["events"], "candidate_events": actual["events"],
                    **{f"delta_g_h{h}": actual["returns"][h] - reference["returns"][h]
                       for h in reference["returns"]},
                })
    return rows


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    dimensions = [("overall", rows)]
    dimensions += [(source, [r for r in rows if r["source"] == source])
                   for source in sorted({r["source"] for r in rows})]
    dimensions += [(phase, [r for r in rows if r["phase"] == phase]) for phase in PHASES]
    dimensions += [(f"{source}:{phase}", [r for r in rows
                    if r["source"] == source and r["phase"] == phase])
                   for source in sorted({r["source"] for r in rows}) for phase in PHASES]
    for key, selected in dimensions:
        if not selected: summary[key] = {"samples": 0}; continue
        entry: dict[str, Any] = {"samples": len(selected)}
        for h in (1, 5, 10, 20):
            positive = [r for r in selected if r["delta_q"] > 0]
            entry[f"predicted_positive_actual_negative_h{h}_fraction"] = (
                float(np.mean([r[f"delta_g_h{h}"] < 0 for r in positive])) if positive else float("nan")
            )
            correlations = []
            groups = defaultdict(list)
            for row in selected:
                groups[(row["source"], row["phase"], row["state_index"])].append(row)
            for group in groups.values():
                if len(group) >= 3:
                    corr = spearmanr([r["delta_q"] for r in group],
                                     [r[f"delta_g_h{h}"] for r in group]).statistic
                    if np.isfinite(corr): correlations.append(corr)
            entry[f"mean_per_state_spearman_h{h}"] = (
                float(np.mean(correlations)) if correlations else float("nan")
            )
        summary[key] = entry
    return summary


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--reuse-exploration-run", type=Path)
    parser.add_argument("--exploration-only", action="store_true")
    args = parser.parse_args(); CHECKPOINT = args.checkpoint.resolve()
    run = Path("outputs/sac_diagnostics") / f"sac_local_policy_audit_{args.run_id}"
    run.mkdir(parents=True, exist_ok=False)
    before = _sha(CHECKPOINT); core, payload = _load_core(CHECKPOINT)
    if args.reuse_exploration_run:
        previous = args.reuse_exploration_run.resolve()
        exploration = json.loads((previous / "exploration_scale_results.json").read_text())
        paired = json.loads((previous / "paired_seed_exploration.json").read_text())
        phase_stats = json.loads((previous / "phase_failure_stats.json").read_text())
    else:
        exploration, paired, phase_stats = exploration_audit(core)
    (run / "exploration_scale_results.json").write_text(json.dumps(exploration, default=_json, indent=2)+"\n")
    (run / "paired_seed_exploration.json").write_text(json.dumps(paired, indent=2)+"\n")
    (run / "phase_failure_stats.json").write_text(json.dumps(phase_stats, default=_json, indent=2)+"\n")
    if args.exploration_only:
        print(json.dumps({"run_directory": str(run.resolve()),
                          "exploration": {k: v.get("success") for k, v in exploration.items()},
                          "networks_updated": False, "replay_written": False}, indent=2))
        return
    formal, formal_snapshots, formal_reconstruction = _reconstruct_formal()
    online, online_snapshots, reconstruction = _reconstruct_online(payload)
    q_static = {"formal": _q_guidance_for_states(core, formal, "formal"),
                "online": _q_guidance_for_states(core, online, "online"),
                "formal_reconstruction": formal_reconstruction,
                "online_reconstruction": reconstruction}
    rows = (_counterfactual(core, formal_snapshots, "formal_nominal_reconstructed")
            + _counterfactual(core, online_snapshots, "online_replay_reconstructed"))
    counter = _counterfactual_summary(rows)
    (run / "q_guidance_summary.json").write_text(json.dumps(q_static, default=_json, indent=2)+"\n")
    (run / "counterfactual_results.json").write_text(json.dumps(rows, default=_json, indent=2)+"\n")
    (run / "phase_q_reliability.json").write_text(json.dumps(counter, default=_json, indent=2)+"\n")
    gradients = {source: {phase: data.get("gradient_norm") for phase, data in values.items()}
                 for source, values in q_static.items() if source in ("formal", "online")}
    (run / "q_gradient_stats.json").write_text(json.dumps(gradients, default=_json, indent=2)+"\n")
    if _sha(CHECKPOINT) != before: raise RuntimeError("diagnostic mutated source checkpoint")
    summary = {"run_directory": str(run.resolve()), "checkpoint": str(CHECKPOINT),
               "checkpoint_sha256": before, "networks_updated": False,
               "replay_written": False, "exploration": {k: v.get("success") for k,v in exploration.items()},
               "q_counterfactual": counter,
               "formal_reconstruction": formal_reconstruction,
               "online_reconstruction": reconstruction}
    (run / "summary.json").write_text(json.dumps(summary, default=_json, indent=2)+"\n")
    print(json.dumps(summary, default=_json, indent=2))


if __name__ == "__main__": main()
