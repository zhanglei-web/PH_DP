"""Episode-safe Monte-Carlo supervision for offline SAC Critic initialization."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

from mujoco_shared_control.sac.critic import TwinSACCritic


GAMMA = 0.995
PHASES = ("P1", "P2", "P3", "P4")
CATEGORIES = ("nominal_success", "normal_recovered", "delayed_recovery", "failure")


@dataclass(frozen=True)
class CriticPretrainConfig:
    gamma: float = GAMMA
    batch_size: int = 512
    learning_rate: float = 3e-4
    max_epochs: int = 100
    early_stopping_patience: int = 12
    gradient_clip: float = 1.0
    seed: int = 20260813


@dataclass
class CriticArrays:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    mc_return: np.ndarray
    category: np.ndarray
    phase: np.ndarray
    episode_id: np.ndarray

    def subset(self, mask: np.ndarray) -> "CriticArrays":
        return CriticArrays(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})

    def __len__(self) -> int: return len(self.reward)


def fixed_split(seed: int) -> str:
    if 100_000 <= seed <= 100_799 or 200_000 <= seed <= 200_239: return "train"
    if 100_800 <= seed <= 100_899 or 200_240 <= seed <= 200_269: return "validation"
    if 100_900 <= seed <= 100_999 or 200_270 <= seed <= 200_299: return "test"
    raise ValueError(f"formal seed outside frozen Critic split: {seed}")


def monte_carlo_returns(reward: np.ndarray, terminated: np.ndarray,
                        truncated: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Finite recorded return; true terminals and truncations both end the record.

    Truncation remains distinguishable in the dataset and Bellman diagnostics.
    A finite-trajectory MC label has no unobserved post-horizon reward to append.
    """
    reward = np.asarray(reward, np.float64); terminated = np.asarray(terminated, bool)
    truncated = np.asarray(truncated, bool)
    if not (reward.shape == terminated.shape == truncated.shape):
        raise ValueError("reward/terminal arrays must share shape")
    result = np.empty_like(reward); running = 0.0
    for index in range(len(reward)-1, -1, -1):
        if terminated[index] or truncated[index]: running = 0.0
        running = reward[index] + gamma * running; result[index] = running
    return result.astype(np.float32)


def build_arrays(manifest_path: Path, reward_run: Path) -> tuple[dict[str, CriticArrays], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((reward_run / "summary.json").read_text())
    regression = summary["official_regression"]
    if regression["reward_max_abs_difference"] > 1e-12 or any(
        regression[key] for key in ("terminal_decision_mismatch_count", "phase_mismatch_count",
                                    "illegal_drop_mismatch_count")
    ):
        raise RuntimeError("reward validation artifact is not production-equivalent")
    reward_rows: dict[str, list[dict[str, str]]] = {}
    with (reward_run / "reward_components.csv").open() as stream:
        for row in csv.DictReader(stream): reward_rows.setdefault(row["episode_id"], []).append(row)
    episode_rows = {}
    with (reward_run / "episode_returns.csv").open() as stream:
        for row in csv.DictReader(stream): episode_rows[row["episode_id"]] = row
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in CriticArrays.__dataclass_fields__}
        for split in ("train", "validation", "test")
    }
    fallback = total = projected = 0
    split_episode_ids: dict[str, list[str]] = {key: [] for key in accum}
    for item in manifest["episodes"]:
        split = fixed_split(int(item["environment_seed"])); episode_id = item["episode_id"]
        rows = reward_rows[episode_id]; length = len(rows); ep = episode_rows[episode_id]
        with h5py.File(root / item["path"], "r") as handle:
            obs = np.asarray(handle["observations/policy_state_42"][:length], np.float32)
            nxt = np.asarray(handle["next_observations/policy_state_42"][:length], np.float32)
            action = np.asarray(handle["actions/normalized"][:length], np.float32)
            status = np.asarray(handle["actions/status"][:length], np.uint8)
        rewards = np.asarray([float(row["reward_total"]) for row in rows], np.float32)
        phases = np.asarray([row["phase"] for row in rows], dtype="U4")
        term = np.zeros(length, bool); trunc = np.zeros(length, bool)
        if ep["sac_reason"] == "time_limit": trunc[-1] = True
        else: term[-1] = True
        returns = monte_carlo_returns(rewards, term, trunc)
        xyz = np.linalg.norm(action[:, :3], axis=1); rot = np.linalg.norm(action[:, 3:6], axis=1)
        if np.any(xyz > 1+1e-6) or np.any(rot > 1+1e-6) or np.any(abs(action[:, 6]) > 1+1e-6):
            raise RuntimeError(f"inadmissible attempted action in {episode_id}")
        fallback += int(status[:, 2].sum()); projected += int(status[:, 1].sum()); total += length
        values = {"observation": obs, "action": action, "reward": rewards[:, None],
                  "next_observation": nxt, "terminated": term[:, None],
                  "truncated": trunc[:, None], "mc_return": returns[:, None],
                  "category": np.full(length, item["category"], dtype="U24"),
                  "phase": phases, "episode_id": np.full(length, episode_id, dtype="U80")}
        for name, value in values.items(): accum[split][name].append(value)
        split_episode_ids[split].append(episode_id)
    arrays = {split: CriticArrays(**{name: np.concatenate(parts) for name, parts in values.items()})
              for split, values in accum.items()}
    audit = {"episodes": {k: len(v) for k, v in split_episode_ids.items()},
             "transitions": {k: len(v) for k, v in arrays.items()},
             "episode_ids": split_episode_ids, "fallback_attempted_action_rows": fallback,
             "adapter_projected_rows": projected, "total_semantic_transitions": total,
             "reward_regression": regression}
    return arrays, audit


def build_arrays_from_semantic_run(
    manifest_path: Path, semantic_run: Path, *, reward_version: str,
) -> tuple[dict[str, CriticArrays], dict[str, Any]]:
    """Load an immutable HDF5 corpus using a derived reward/terminal interpretation."""
    manifest = json.loads(manifest_path.read_text())
    derived = json.loads((semantic_run / "dataset_manifest.json").read_text())
    if derived["source_manifest_content_sha"] != manifest["content_sha256"]:
        raise RuntimeError("derived semantic corpus does not match formal manifest")
    if derived["reward_version"] != reward_version:
        raise RuntimeError("unexpected derived reward version")
    episode_semantics = {row["episode_id"]: row for row in derived["episodes"]}
    reward_rows: dict[str, list[dict[str, str]]] = {}
    with (semantic_run / "reward_components.csv").open() as stream:
        for row in csv.DictReader(stream): reward_rows.setdefault(row["episode_id"], []).append(row)
    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in CriticArrays.__dataclass_fields__}
        for split in ("train", "validation", "test")
    }
    split_episode_ids: dict[str, list[str]] = {key: [] for key in accum}
    fallback = projected = total = 0
    for item in manifest["episodes"]:
        split = fixed_split(int(item["environment_seed"])); episode_id = item["episode_id"]
        semantic = episode_semantics[episode_id]; rows = reward_rows[episode_id]
        length = int(semantic["v2_transitions"])
        if len(rows) != length: raise RuntimeError(f"semantic row count mismatch for {episode_id}")
        with h5py.File(root / item["path"], "r") as handle:
            obs = np.asarray(handle["observations/policy_state_42"][:length], np.float32)
            nxt = np.asarray(handle["next_observations/policy_state_42"][:length], np.float32)
            action = np.asarray(handle["actions/normalized"][:length], np.float32)
            status = np.asarray(handle["actions/status"][:length], np.uint8)
        rewards = np.asarray([float(row["reward_total"]) for row in rows], np.float32)
        phases = np.asarray([row["phase"] for row in rows], dtype="U4")
        term = np.asarray([row["terminated"].lower() == "true" for row in rows], bool)
        trunc = np.asarray([row["truncated"].lower() == "true" for row in rows], bool)
        returns = monte_carlo_returns(rewards, term, trunc)
        if not (term[-1] or trunc[-1]):
            raise RuntimeError(f"derived episode does not end at a terminal boundary: {episode_id}")
        xyz = np.linalg.norm(action[:, :3], axis=1); rot = np.linalg.norm(action[:, 3:6], axis=1)
        if np.any(xyz > 1+1e-6) or np.any(rot > 1+1e-6) or np.any(abs(action[:, 6]) > 1+1e-6):
            raise RuntimeError(f"inadmissible attempted action in {episode_id}")
        fallback += int(status[:, 2].sum()); projected += int(status[:, 1].sum()); total += length
        values = {"observation": obs, "action": action, "reward": rewards[:, None],
                  "next_observation": nxt, "terminated": term[:, None],
                  "truncated": trunc[:, None], "mc_return": returns[:, None],
                  "category": np.full(length, item["category"], dtype="U24"),
                  "phase": phases, "episode_id": np.full(length, episode_id, dtype="U80")}
        for name, value in values.items(): accum[split][name].append(value)
        split_episode_ids[split].append(episode_id)
    arrays = {split: CriticArrays(**{name: np.concatenate(parts) for name, parts in values.items()})
              for split, values in accum.items()}
    audit = {"reward_version": reward_version,
             "source_manifest_content_sha": manifest["content_sha256"],
             "derived_manifest_path": str((semantic_run / "dataset_manifest.json").resolve()),
             "episodes": {key: len(value) for key, value in split_episode_ids.items()},
             "transitions": {key: len(value) for key, value in arrays.items()},
             "episode_ids": split_episode_ids,
             "fallback_attempted_action_rows": fallback,
             "adapter_projected_rows": projected,
             "total_semantic_transitions": total}
    return arrays, audit


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1); target = np.asarray(target).reshape(-1)
    error = prediction-target
    return {"count": len(target), "mae": float(np.mean(abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "pearson": float(pearsonr(prediction, target).statistic),
            "spearman": float(spearmanr(prediction, target).statistic),
            "prediction_mean": float(prediction.mean()), "prediction_std": float(prediction.std()),
            "target_mean": float(target.mean()), "target_std": float(target.std())}


@torch.no_grad()
def predict(critic: TwinSACCritic, arrays: CriticArrays, mean: torch.Tensor,
            std: torch.Tensor, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    outputs1, outputs2 = [], []
    for start in range(0, len(arrays), batch_size):
        obs = torch.from_numpy(arrays.observation[start:start+batch_size])
        obs = (obs-mean)/std; action = torch.from_numpy(arrays.action[start:start+batch_size])
        q1,q2=critic(obs,action); outputs1.append(q1.numpy());outputs2.append(q2.numpy())
    return np.concatenate(outputs1),np.concatenate(outputs2)


def evaluate(critic: TwinSACCritic, arrays: CriticArrays, mean: torch.Tensor,
             std: torch.Tensor) -> dict[str, Any]:
    q1,q2=predict(critic,arrays,mean,std); avg=(q1+q2)/2
    result={"overall":regression_metrics(avg,arrays.mc_return),
            "q1_q2_disagreement_mae":float(np.mean(abs(q1-q2))),
            "q1": {"mean":float(q1.mean()),"std":float(q1.std()),"min":float(q1.min()),"max":float(q1.max())},
            "q2": {"mean":float(q2.mean()),"std":float(q2.std()),"min":float(q2.min()),"max":float(q2.max())}}
    for key,values in (("category",CATEGORIES),("phase",PHASES)):
        result[key]={}
        labels=getattr(arrays,key)
        for value in values:
            mask=labels==value
            if mask.any(): result[key][value]=regression_metrics(avg[mask],arrays.mc_return[mask])
    return result


def train_critic(train: CriticArrays, validation: CriticArrays, mean: torch.Tensor,
                 std: torch.Tensor, config: CriticPretrainConfig,
                 history_path: Path) -> tuple[TwinSACCritic, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    critic=TwinSACCritic(); optimizer=torch.optim.Adam(critic.parameters(),lr=config.learning_rate)
    rng=np.random.default_rng(config.seed); best_state=None;best_optimizer=None;best=float("inf");best_epoch=0;stale=0;history=[]
    for epoch in range(1,config.max_epochs+1):
        indices=rng.permutation(len(train)); losses=[];critic.train()
        for start in range(0,len(indices),config.batch_size):
            idx=indices[start:start+config.batch_size]
            obs=(torch.from_numpy(train.observation[idx])-mean)/std
            action=torch.from_numpy(train.action[idx]); target=torch.from_numpy(train.mc_return[idx])
            q1,q2=critic(obs,action);loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),config.gradient_clip);optimizer.step()
            losses.append(float(loss.detach()))
        critic.eval(); val=evaluate(critic,validation,mean,std); val_loss=val["overall"]["rmse"]**2*2
        row={"epoch":epoch,"train_mc_loss":float(np.mean(losses)),"validation_mc_loss":val_loss,
             "validation_spearman":val["overall"]["spearman"],
             "q1_q2_disagreement_mae":val["q1_q2_disagreement_mae"]}
        history.append(row)
        with history_path.open("a") as stream: stream.write(json.dumps(row)+"\n")
        if val_loss < best-1e-8:
            best=val_loss;best_epoch=epoch;stale=0
            best_state={k:v.detach().clone() for k,v in critic.state_dict().items()}
            best_optimizer=deepcopy(optimizer.state_dict())
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f"epoch={epoch} train={row['train_mc_loss']:.6g} val={val_loss:.6g} rho={row['validation_spearman']:.4f}",flush=True)
        if stale>=config.early_stopping_patience: break
    assert best_state is not None and best_optimizer is not None
    critic.load_state_dict(best_state);critic.eval()
    return critic,{"best_epoch":best_epoch,"best_validation_mc_loss":best,"epochs_completed":len(history),
                   "optimizer_state_dict":best_optimizer},history
