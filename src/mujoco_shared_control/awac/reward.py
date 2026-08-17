"""Offline AWAC Reward V1 and conservative terminal reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mujoco_shared_control.awac.dataset import _atomic_json, _atomic_npz
from mujoco_shared_control.awac.milestones import MilestoneConfig, MilestoneTracker
from mujoco_shared_control.collection.manifest import load_manifest, sha256_file
from mujoco_shared_control.data.recording import FrameEvent
from mujoco_shared_control.experts.rule_pick_place import RuleExpertStage


AWAC_REWARD_VERSION = "awac_reward_v1"


@dataclass(frozen=True)
class AWACRewardV1Config:
    """Frozen reward constants, expressed in reward units per transition."""

    step_penalty: float = -0.001
    progress_scale: float = 1.0
    progress_clip: float = 0.02
    grasp_bonus: float = 0.5
    lift_bonus: float = 0.5
    transport_bonus: float = 0.5
    release_bonus: float = 0.75
    retreat_bonus: float = 0.5
    success_bonus: float = 5.0
    failure_penalty: float = -5.0
    grasp_offset_z_m: float = 0.012
    hover_height_m: float = 0.12
    above_goal_height_m: float = 0.16
    retreat_height_m: float = 0.16


@dataclass(frozen=True)
class AWACRewardV1OnlineStep:
    reward: float
    terminated: bool
    truncated: bool
    task_success: bool
    termination_reason: str
    milestones: np.ndarray
    stage: int
    components: dict[str, float]


class AWACRewardV1Online:
    """Stateful online/evaluation form of the frozen offline reward.

    The policy does not observe expert stage. The reward protocol derives the same
    ordered task phase from one-shot task milestones, then calls the frozen
    `_distance_for_stage` implementation used by offline conversion.
    """

    def __init__(
        self, initial_observation_43: np.ndarray,
        config: AWACRewardV1Config = AWACRewardV1Config(),
        tracker: MilestoneTracker | None = None,
    ) -> None:
        observation = np.asarray(initial_observation_43, np.float32)
        if observation.shape != (43,) or not np.isfinite(observation).all():
            raise ValueError("online Reward V1 requires a finite 43-D initial observation")
        self.config = config
        self.tracker = tracker or MilestoneTracker(MilestoneConfig(
            retreat_height_m=config.retreat_height_m,
        ))
        self.tracker.reset(observation)
        self.stable_release_steps = 0
        self.terminal = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker.state_dict(),
            "stable_release_steps": self.stable_release_steps,
            "terminal": self.terminal,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tracker.load_state_dict(state["tracker"])
        self.stable_release_steps = int(state["stable_release_steps"])
        self.terminal = bool(state["terminal"])

    def _stage(self) -> RuleExpertStage:
        milestones = self.tracker.current
        if milestones[3]:
            return RuleExpertStage.RETREAT
        if milestones[2]:
            return RuleExpertStage.DESCEND_TO_GOAL
        if milestones[1]:
            return RuleExpertStage.TRANSPORT
        if milestones[0]:
            return RuleExpertStage.LIFT
        return RuleExpertStage.DESCEND

    def step(
        self, observation_43: np.ndarray, next_observation_43: np.ndarray, *,
        ik_failure: bool = False, time_limit: bool = False,
    ) -> AWACRewardV1OnlineStep:
        if self.terminal:
            raise RuntimeError("online Reward V1 cannot step after a terminal boundary")
        observation = np.asarray(observation_43, np.float32)
        next_observation = np.asarray(next_observation_43, np.float32)
        if observation.shape != (43,) or next_observation.shape != (43,):
            raise ValueError("online Reward V1 observations must be 43-D")
        stage = self._stage()
        components = {
            "step_penalty": self.config.step_penalty, "progress": 0.0,
            "grasp": 0.0, "lift": 0.0, "transport": 0.0,
            "release": 0.0, "retreat": 0.0,
            "success_terminal": 0.0, "failure_terminal": 0.0,
        }
        before = _distance_for_stage(observation, int(stage), self.config)
        after = _distance_for_stage(next_observation, int(stage), self.config)
        if before is not None and after is not None:
            components["progress"] = float(np.clip(
                self.config.progress_scale * (before - after),
                -self.config.progress_clip, self.config.progress_clip,
            ))
        was_grasped = bool(observation[42]); grasped = bool(next_observation[42])
        update = self.tracker.update(next_observation)
        bonuses = np.asarray([
            self.config.grasp_bonus, self.config.lift_bonus,
            self.config.transport_bonus, self.config.release_bonus,
            self.config.retreat_bonus,
        ])
        for name, value in zip(("grasp", "lift", "transport", "release", "retreat"), bonuses):
            index = ("grasp", "lift", "transport", "release", "retreat").index(name)
            if update.rising[index]:
                components[name] = float(value)
        illegal_drop = bool(was_grasped and not grasped and not update.current[3])
        self.stable_release_steps = (
            self.stable_release_steps + 1
            if update.current[3] and update.conditions["goal_contained"] and not grasped else 0
        )
        success = bool(
            update.current.all() and self.stable_release_steps >= 4
        )
        terminated = truncated = False; reason = ""
        if success:
            components["success_terminal"] = self.config.success_bonus
            terminated, reason = True, "task_success"
        elif illegal_drop:
            components["failure_terminal"] = self.config.failure_penalty
            terminated, reason = True, "illegal_drop"
        elif ik_failure:
            components["failure_terminal"] = self.config.failure_penalty
            terminated, reason = True, "ik_failure_limit"
        elif time_limit:
            components["failure_terminal"] = self.config.failure_penalty
            truncated, reason = True, "timeout"
        self.terminal = terminated or truncated
        return AWACRewardV1OnlineStep(
            float(sum(components.values())), terminated, truncated, success,
            reason, update.current, int(stage), components,
        )


def _distance_for_stage(state: np.ndarray, stage: int, config: AWACRewardV1Config) -> float | None:
    ee = state[14:17].astype(np.float64)
    obj = state[22:25].astype(np.float64)
    goal = state[29:32].astype(np.float64)
    if stage == int(RuleExpertStage.PRE_GRASP):
        target = obj + np.array([0.0, 0.0, config.hover_height_m])
        return float(np.linalg.norm(ee - target))
    if stage == int(RuleExpertStage.DESCEND):
        target = obj + np.array([0.0, 0.0, config.grasp_offset_z_m])
        return float(np.linalg.norm(ee - target))
    if stage == int(RuleExpertStage.LIFT):
        return float(abs(obj[2] - (goal[2] + config.above_goal_height_m)))
    if stage == int(RuleExpertStage.TRANSPORT):
        target = goal + np.array([0.0, 0.0, config.above_goal_height_m])
        return float(np.linalg.norm(obj - target))
    if stage == int(RuleExpertStage.DESCEND_TO_GOAL):
        return float(np.linalg.norm(obj - goal))
    if stage == int(RuleExpertStage.RETREAT):
        target = goal + np.array([0.0, 0.0, config.retreat_height_m])
        return float(np.linalg.norm(ee - target))
    # CLOSE_GRIPPER and OPEN_GRIPPER deliberately have no dense reward.
    return None


def _episode_reward(
    obs: np.ndarray,
    next_obs: np.ndarray,
    stages: np.ndarray,
    milestones: np.ndarray,
    steps: np.ndarray,
    config: AWACRewardV1Config,
) -> tuple[np.ndarray, dict[str, int]]:
    rewards = np.full(len(obs), config.step_penalty, dtype=np.float64)
    for index, stage in enumerate(stages.astype(int)):
        before = _distance_for_stage(obs[index], stage, config)
        after = _distance_for_stage(next_obs[index], stage, config)
        if before is not None and after is not None:
            progress = np.clip(
                config.progress_scale * (before - after),
                -config.progress_clip,
                config.progress_clip,
            )
            rewards[index] += progress

    bonuses = np.asarray(
        [
            config.grasp_bonus,
            config.lift_bonus,
            config.transport_bonus,
            config.release_bonus,
            config.retreat_bonus,
        ],
        dtype=np.float64,
    )
    awarded = np.zeros(5, dtype=bool)
    suppressed = np.zeros(5, dtype=np.int64)
    previous = np.zeros(5, dtype=bool)
    for index, current in enumerate(milestones.astype(bool)):
        contiguous = index == 0 or int(steps[index]) == int(steps[index - 1]) + 1
        rising = current & ~previous
        eligible = rising & ~awarded & contiguous
        rewards[index] += float(bonuses[eligible].sum())
        awarded |= eligible
        suppressed += (rising & ~contiguous).astype(np.int64)
        previous = current
    return rewards.astype(np.float32), {
        name: int(value)
        for name, value in zip(
            ("grasp", "lift", "transport", "release", "retreat"), suppressed
        )
    }


def _copy_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        return {name: np.asarray(dataset[name]).copy() for name in dataset.files}


def derive_awac_reward_v1(
    manifest_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    config: AWACRewardV1Config = AWACRewardV1Config(),
    excluded_categories: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Replace reward/terminal labels in a copy of the frozen AWAC transition data.

    Original HDF5 and source NPZ files are opened read-only.  A terminal outcome is
    folded onto a retained boundary only when it was already decided at that exact
    source step.  State-changing filtered tails are never represented as if their
    outcome had been caused by the last retained action.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    by_id = {item["episode_id"]: item for item in manifest["episodes"]}
    if len(by_id) != len(manifest["episodes"]):
        raise ValueError("formal manifest contains duplicate episode IDs")

    source_hashes = {
        split: sha256_file(source_dir / f"{split}.npz")
        for split in ("train", "validation")
    }
    output_counts: dict[str, int] = {}
    terminal_counts = {
        "explicit_success_preserved": 0,
        "failure_reconstructed_at_expert_failed_step": 0,
        "delayed_recovery_marked_censored_truncation": 0,
    }
    suppressed_total = {name: 0 for name in ("grasp", "lift", "transport", "release", "retreat")}
    tail_summary = {
        "episode_count_with_filtered_tail": 0,
        "filtered_transition_count": 0,
        "all_tail_rows_settling": 0,
        "all_tail_rows_zero_cartesian_motion": 0,
        "state_changing_tail_episode_count": 0,
        "foldable_no_control_tail_episode_count": 0,
        "tail_episode_count_with_grasp_lost": 0,
        "tail_grasp_lost_transition_count": 0,
    }
    tail_episodes: list[dict[str, Any]] = []

    for split in ("train", "validation"):
        arrays = _copy_npz(source_dir / f"{split}.npz")
        episode_ids = arrays["episode_id"]
        split_seen = set(map(str, np.unique(episode_ids)))
        expected = {
            item["episode_id"] for item in manifest["episodes"]
            if item["split"] == split and item["category"] not in set(excluded_categories)
        }
        if split_seen != expected:
            raise ValueError(f"{split}.npz episode IDs do not match the formal manifest split")

        new_reward = np.empty(len(episode_ids), dtype=np.float32)
        new_terminated = arrays["terminated"].astype(bool, copy=True)
        new_truncated = arrays["truncated"].astype(bool, copy=True)
        reasons = arrays["termination_reason"].astype("U64", copy=True)

        for episode_id in sorted(split_seen):
            indices = np.flatnonzero(episode_ids == episode_id)
            order = np.argsort(arrays["step_index"][indices])
            indices = indices[order]
            steps = arrays["step_index"][indices].astype(np.int64)
            reward, suppressed = _episode_reward(
                arrays["obs"][indices], arrays["next_obs"][indices],
                arrays["expert_stage"][indices], arrays["task_milestones"][indices],
                steps, config,
            )
            for name, count in suppressed.items():
                suppressed_total[name] += count
            new_reward[indices] = reward

            item = by_id[episode_id]
            source_path = (root / item["path"]).resolve()
            if not source_path.is_relative_to(root):
                raise ValueError(f"unsafe manifest path for {episode_id}")
            if sha256_file(source_path) != item["sha256"]:
                raise ValueError(f"formal HDF5 checksum mismatch: {source_path}")
            with h5py.File(source_path, "r") as episode:
                total = len(episode["identity/step_index"])
                last = int(steps[-1])
                expert_failed_step = int(episode.attrs.get("expert_failed_step", -1))
                source_terminated = bool(episode["labels/terminated"][-1])
                source_truncated = bool(episode["labels/truncated"][-1])
                source_success = bool(episode["labels/task_success"][-1])
                source_reason = str(episode.attrs.get("termination_reason", ""))

                tail_steps = np.arange(last + 1, total, dtype=np.int64)
                state_changing = False
                tail_record: dict[str, Any] | None = None
                if len(tail_steps):
                    stages = episode["labels/expert_stage"][tail_steps].astype(int)
                    commands = episode["actions/command_after_clipping"][tail_steps]
                    before = episode["observations/policy_state_42"][tail_steps]
                    after = episode["next_observations/policy_state_42"][tail_steps]
                    cartesian_zero = bool(np.allclose(commands[:, :6], 0.0, atol=1e-12))
                    gripper_changes = bool(np.any(np.abs(commands[:, 6] - before[:, 21]) > 1e-5))
                    object_motion = bool(np.any(np.linalg.norm(after[:, 22:25] - before[:, 22:25], axis=1) > 1e-5))
                    gripper_opening_motion = bool(np.any(np.abs(after[:, 21] - before[:, 21]) > 1e-5))
                    object_grasp_change = bool(np.any(
                        episode["observations/object_grasped"][tail_steps].astype(bool)
                        != episode["next_observations/object_grasped"][tail_steps].astype(bool)
                    ))
                    milestone_change = bool(np.any(np.diff(
                        np.vstack((arrays["task_milestones"][indices[-1]], episode["labels/task_milestones"][tail_steps])),
                        axis=0,
                    ) != 0))
                    tail_grasp_lost_count = int(np.sum(
                        (episode["labels/events"][tail_steps].astype(np.uint32)
                         & int(FrameEvent.GRASP_LOST)) != 0
                    ))
                    state_changing = (
                        gripper_changes or object_motion or gripper_opening_motion
                        or object_grasp_change or milestone_change
                    )
                    tail_summary["episode_count_with_filtered_tail"] += 1
                    tail_summary["filtered_transition_count"] += len(tail_steps)
                    tail_summary["all_tail_rows_settling"] += int(np.all(stages == int(RuleExpertStage.SETTLING)))
                    tail_summary["all_tail_rows_zero_cartesian_motion"] += int(cartesian_zero)
                    tail_summary["state_changing_tail_episode_count"] += int(state_changing)
                    tail_summary["foldable_no_control_tail_episode_count"] += int(not state_changing)
                    tail_summary["tail_episode_count_with_grasp_lost"] += int(tail_grasp_lost_count > 0)
                    tail_summary["tail_grasp_lost_transition_count"] += tail_grasp_lost_count
                    tail_record = {
                        "episode_id": episode_id,
                        "category": str(item["category"]),
                        "split": split,
                        "last_retained_step": last,
                        "source_final_step": total - 1,
                        "filtered_tail_length": len(tail_steps),
                        "expert_failed_step": expert_failed_step,
                        "all_settling": bool(np.all(stages == int(RuleExpertStage.SETTLING))),
                        "zero_cartesian_motion": cartesian_zero,
                        "gripper_command_changes_state": gripper_changes,
                        "object_motion": object_motion,
                        "gripper_opening_motion": gripper_opening_motion,
                        "object_grasp_state_change": object_grasp_change,
                        "milestone_change": milestone_change,
                        "grasp_lost_transition_count": tail_grasp_lost_count,
                        "state_changing": state_changing,
                        "source_final_success": source_success,
                        "source_final_terminated": source_terminated,
                        "source_final_truncated": source_truncated,
                        "source_termination_reason": source_reason,
                        "terminal_folded": False,
                    }

                category = str(item["category"])
                last_index = indices[-1]
                if category in ("nominal_success", "normal_recovered"):
                    if not (new_terminated[last_index] and arrays["task_success"][last_index]):
                        raise ValueError(f"successful episode lacks retained terminal row: {episode_id}")
                    new_reward[last_index] += np.float32(config.success_bonus)
                    terminal_counts["explicit_success_preserved"] += 1
                elif category == "failure":
                    if expert_failed_step != last:
                        raise ValueError(
                            f"failure boundary differs from expert_failed_step for {episode_id}"
                        )
                    new_terminated[last_index] = True
                    new_truncated[last_index] = False
                    reasons[last_index] = "expert_unrecoverable_failure"
                    new_reward[last_index] += np.float32(config.failure_penalty)
                    terminal_counts["failure_reconstructed_at_expert_failed_step"] += 1
                    if tail_record is not None:
                        tail_record["terminal_folded"] = True
                        tail_record["fold_basis"] = "failure was decided at retained expert_failed_step; tail outcome was not folded"
                elif category == "delayed_recovery":
                    if not len(tail_steps) or not state_changing:
                        raise ValueError(
                            f"delayed recovery lacks the expected state-changing filtered tail: {episode_id}"
                        )
                    new_terminated[last_index] = False
                    new_truncated[last_index] = True
                    reasons[last_index] = "awac_state_changing_tail_censored"
                    terminal_counts["delayed_recovery_marked_censored_truncation"] += 1
                else:
                    raise ValueError(f"unsupported formal category: {category}")
                if tail_record is not None:
                    tail_episodes.append(tail_record)

        arrays["reward"] = new_reward
        arrays["terminated"] = new_terminated.astype(np.bool_)
        arrays["truncated"] = new_truncated.astype(np.bool_)
        arrays["termination_reason"] = reasons
        if not np.isfinite(new_reward).all():
            raise RuntimeError("AWAC Reward V1 produced a non-finite reward")
        _atomic_npz(output_dir / f"{split}.npz", arrays)
        output_counts[split] = len(new_reward)

    output_hashes = {
        split: sha256_file(output_dir / f"{split}.npz")
        for split in ("train", "validation")
    }
    if source_hashes != {
        split: sha256_file(source_dir / f"{split}.npz")
        for split in ("train", "validation")
    }:
        raise RuntimeError("source AWAC NPZ changed while deriving rewards")

    report = {
        "version": AWAC_REWARD_VERSION,
        "manifest": str(manifest_path),
        "source_dataset_dir": str(source_dir),
        "output_dataset_dir": str(output_dir),
        "source_npz_sha256": source_hashes,
        "output_npz_sha256": output_hashes,
        "transition_count_by_split": output_counts,
        "reward_config": asdict(config),
        "excluded_categories": list(excluded_categories),
        "progress_definition": {
            "formula": "clip(progress_scale * (distance(obs)-distance(next_obs)), +/-progress_clip) + step_penalty",
            "stages": {
                "PRE_GRASP": "ee to object hover target",
                "DESCEND": "ee to object grasp target",
                "CLOSE_GRIPPER": "none",
                "LIFT": "object z to goal z + above_goal_height",
                "TRANSPORT": "object to above-goal target",
                "DESCEND_TO_GOAL": "object to goal",
                "OPEN_GRIPPER": "none",
                "RETREAT": "ee to retreat target",
            },
        },
        "milestone_bonus_semantics": "one reward per episode, only on a rising edge contiguous with the preceding retained source step",
        "suppressed_milestone_edges_after_filtered_gap": suppressed_total,
        "terminal_reconstruction": terminal_counts,
        "filtered_tail_summary": tail_summary,
        "filtered_tail_policy": {
            "failure": "terminal failure is reconstructed because expert_failed_step equals the retained boundary; no later tail outcome is folded",
            "delayed_recovery": "success is not folded because the filtered tail contains state-changing control; retained boundary is marked truncated/censored",
        },
        "filtered_tail_episodes": tail_episodes,
        "schema_changed": False,
        "source_hdf5_modified": False,
        "training_started": False,
    }
    _atomic_json(output_dir / "reward_v1_conversion_report.json", report)
    return report
