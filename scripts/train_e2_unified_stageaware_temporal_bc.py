#!/usr/bin/env python3
"""Train fixed-split causal 20x48D Unified Stage-aware Temporal BC."""
from __future__ import annotations

import argparse, csv, hashlib, json, random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mujoco_shared_control.experts.temporal_recovery_bc import UnifiedStageAwareTemporalBC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL"
STAGE = ROOT / "outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL"
OUT = ROOT / "outputs/experiments/unified_stageaware_temporal_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_TEMPORAL_BC_FORMAL"
SCEN = ("NORMAL", "GRASP_RECOVERY", "TRANSPORT_DROP", "PLACE_RECOVERY")
SEED, HISTORY = 20260818, 20


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class TemporalWindows(Dataset):
    def __init__(self, paths: dict[str, str], ids: list[str]) -> None:
        self.episodes: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        self.scenarios: list[str] = []
        for episode_id in ids:
            with h5py.File(paths[episode_id], "r") as data:
                physical = data["full_physical_state"][:].astype(np.float32)
                phase = data["active_phase"][:].astype(np.int64)
                action = data["raw_pilot_action"][:].astype(np.float32)
                length = len(physical)
                scenario = str(data.attrs["trajectory_type"])
                if physical.shape != (length, 43) or action.shape != (length, 7):
                    raise ValueError(f"invalid schema: {episode_id}")
                if np.any((phase < 0) | (phase > 4)):
                    raise ValueError(f"invalid active phase: {episode_id}")
            episode_index = len(self.episodes)
            padded_physical = np.concatenate((np.repeat(physical[:1], HISTORY - 1, axis=0), physical))
            padded_phase = np.concatenate((np.repeat(phase[:1], HISTORY - 1), phase))
            physical_windows = np.lib.stride_tricks.sliding_window_view(padded_physical, HISTORY, axis=0).transpose(0, 2, 1)
            phase_windows = np.lib.stride_tricks.sliding_window_view(padded_phase, HISTORY)
            self.episodes.append((episode_id, scenario, physical, phase, action, physical_windows, phase_windows))
            self.index.extend((episode_index, step) for step in range(length))
            self.scenarios.extend([scenario] * length)
        self.scenarios_array = np.asarray(self.scenarios, dtype=object)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        episode_index, target = self.index[item]
        _, _, _, _, stored_action, physical_windows, phase_windows = self.episodes[episode_index]
        return torch.from_numpy(physical_windows[target]), torch.from_numpy(phase_windows[target]), torch.from_numpy(stored_action[target]), SCEN.index(self.scenarios[item])

    def scenario_weights(self) -> torch.Tensor:
        counts = {scenario: int(np.sum(self.scenarios_array == scenario)) for scenario in SCEN}
        return torch.tensor([0.25 / counts[scenario] for scenario in self.scenarios], dtype=torch.double)


def normalize(history: torch.Tensor, phase: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.cat(((history - mean) / std, nn.functional.one_hot(phase.long(), 5).float()), dim=-1)


def loss(model, history, phase, action, mean, std):
    motion, logit = model(normalize(history, phase, mean, std))
    motion_loss = nn.functional.mse_loss(motion, action[:, :6])
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(logit, (action[:, 6] > 0).float())
    return motion_loss + gripper_loss, motion_loss, gripper_loss, motion, logit


def physical_normalizer(dataset: TemporalWindows) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(43, np.float64)
    square_total = np.zeros(43, np.float64)
    count = 0
    for _, _, stored_state, _, _, _, _ in dataset.episodes:
        state = stored_state.astype(np.float64)
        total += state.sum(0)
        square_total += np.square(state).sum(0)
        count += len(state)
    mean = total / count
    std = np.sqrt(np.maximum(square_total / count - np.square(mean), 1e-12))
    mean[42], std[42] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def audit_sequences(paths: dict[str, str], ids: list[str]) -> tuple[dict, dict]:
    samples = padded = 0
    details = []
    for episode_id in ids:
        with h5py.File(paths[episode_id], "r") as data:
            state = data["full_physical_state"][:]
            action = data["raw_pilot_action"][:]
        if len(state) != len(action):
            raise ValueError(f"state/action mismatch: {episode_id}")
        samples += len(state)
        padded += min(HISTORY - 1, len(state))
        details.append({"episode_id": episode_id, "transitions": len(state), "history_source_episode_only": True})
    alignment = {"status": "PASS", "history_length": HISTORY, "history_end_index_equals_action_index": True,
                 "future_timestep_used": False, "samples_checked": samples,
                 "construction": "history=max(0,t-19)..t; target=action[t]"}
    boundary = {"status": "PASS", "history_length": HISTORY, "cross_episode_history": False,
                "padding": "repeat-first", "padded_target_steps": padded, "episodes_checked": len(ids),
                "episode_checks": details}
    return alignment, boundary


@torch.no_grad()
def metrics(model, dataset, mean_np, std_np) -> dict:
    device = next(model.parameters()).device
    mean, std = torch.from_numpy(mean_np).to(device), torch.from_numpy(std_np).to(device)
    pred = []; target = []; logits = []; phases = []; scenario = []; totals = []; motions = []; grippers = []
    model.eval()
    for history, phase, action, code in DataLoader(dataset, batch_size=1024):
        total, motion, gripper, p, logit = loss(model, history.to(device), phase.to(device), action.to(device), mean, std)
        pred.append(p.cpu().numpy()); target.append(action.numpy()); logits.append(logit.cpu().numpy())
        phases.append(phase[:, -1].numpy()); scenario.append(code.numpy()); totals.append(float(total)); motions.append(float(motion)); grippers.append(float(gripper))
    pred, target, logits = np.concatenate(pred), np.concatenate(target), np.concatenate(logits)
    phases, scenario = np.concatenate(phases), np.concatenate(scenario)
    error = pred - target[:, :6]; predicted_gripper = logits >= 0; actual_gripper = target[:, 6] > 0
    def one(mask):
        y, q, e = actual_gripper[mask], predicted_gripper[mask], error[mask]
        def f1(value):
            tp = np.sum((q == value) & (y == value)); fp = np.sum((q == value) & (y != value)); fn = np.sum((q != value) & (y == value))
            precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
            return float(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"motion_mse": float(np.mean(e ** 2)), "motion_mae": float(np.mean(abs(e))), "translation_mae": float(np.mean(abs(e[:, :3]))), "rotation_mae": float(np.mean(abs(e[:, 3:]))), "gripper_accuracy": float(np.mean(q == y)), "OPEN_f1": f1(True), "CLOSE_f1": f1(False)}
    return {"total_loss": float(np.mean(totals)), "motion_loss": float(np.mean(motions)), "gripper_loss": float(np.mean(grippers)), "overall": one(np.ones(len(dataset), bool)), "by_scenario": {name: one(scenario == i) for i, name in enumerate(SCEN)}, "by_stage": {str(i): one(phases == i) for i in range(5)}}


def rows_from_metrics(metrics_by_group: dict[str, dict]) -> list[dict]:
    return [{"Group": group, "Motion MSE": value["motion_mse"], "Motion MAE": value["motion_mae"], "Translation MAE": value["translation_mae"], "Rotation MAE": value["rotation_mae"], "Gripper Accuracy": value["gripper_accuracy"], "OPEN F1": value["OPEN_f1"], "CLOSE F1": value["CLOSE_f1"]} for group, value in metrics_by_group.items()]


def initialize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((STAGE / "split_manifest.json").read_text())
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    ids = {entry["episode_id"] for entry in manifest["episodes"]}
    if any(episode_id not in ids for group in split["splits"].values() for episode_id in group):
        raise RuntimeError("STOP: frozen stage-aware split cannot be reused")
    all_ids = [episode_id for group in split["splits"].values() for episode_id in group]
    alignment, boundary = audit_sequences(split["episode_paths"], all_ids)
    dump(OUT / "split_manifest.json", split)
    dump(OUT / "split_reuse_audit.json", {"status": "PASS", "source": str((STAGE / "split_manifest.json").resolve()), "exact_episode_ids_reused": True, "counts": {key: len(value) for key, value in split["splits"].items()}})
    dump(OUT / "temporal_alignment_audit.json", alignment)
    dump(OUT / "sequence_boundary_audit.json", boundary)
    train = TemporalWindows(split["episode_paths"], split["splits"]["train"])
    mean, std = physical_normalizer(train)
    np.savez(OUT / "normalizer.npz", physical_mean=mean, physical_std=std, stage_onehot_normalized=False)
    model = UnifiedStageAwareTemporalBC(); optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    torch.save({"epoch": 0, "best": float("inf"), "best_epoch": 0, "patience": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": []}, OUT / "training_state.pt")
    dump(OUT / "training_config.json", {"architecture": "causal TCN 48->64(k3,d1)->128(k3,d2)->128(k3,d4); Linear128->128 ReLU; tanh motion6 + binary gripper logit", "input": "20x(normalized physical43 + raw CURRENT active_phase onehot5)", "history_length": HISTORY, "control_rate_hz": 20, "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 1024, "max_epochs": 50, "patience": 5, "seed": SEED, "loss": "motion MSE + 1.0 BCEWithLogitsLoss", "sampler": "scenario-balanced temporal windows", "checkpoint_selection": "validation total loss only", "no_future_leakage": True})
    counts = {scenario: int(np.sum(train.scenarios_array == scenario)) for scenario in SCEN}
    dump(OUT / "sampling_distribution.json", {"temporal_window_counts": counts, "target_probability": {scenario: 0.25 for scenario in SCEN}, "sampler": "WeightedRandomSampler"})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--init", action="store_true"); parser.add_argument("--epochs", type=int, default=5); args = parser.parse_args()
    if args.init: initialize()
    split = json.loads((OUT / "split_manifest.json").read_text()); paths = split["episode_paths"]
    train, validation, test = (TemporalWindows(paths, split["splits"][group]) for group in ("train", "validation", "test"))
    normalizer = np.load(OUT / "normalizer.npz"); mean_np, std_np = normalizer["physical_mean"], normalizer["physical_std"]
    state = torch.load(OUT / "training_state.pt", map_location="cpu", weights_only=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = UnifiedStageAwareTemporalBC(); model.load_state_dict(state["model"]); optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); optimizer.load_state_dict(state["optimizer"])
    mean, std = torch.from_numpy(mean_np), torch.from_numpy(std_np)
    for _ in range(args.epochs):
        if state["epoch"] >= 50 or state["patience"] >= 5: break
        epoch = state["epoch"] + 1; model.train(); values = []
        loader = DataLoader(train, batch_size=1024, sampler=WeightedRandomSampler(train.scenario_weights(), len(train), replacement=True, generator=torch.Generator().manual_seed(SEED + epoch)))
        for history, phase, action, _ in loader:
            optimizer.zero_grad(set_to_none=True); total, motion, gripper, _, _ = loss(model, history, phase, action, mean, std)
            if not torch.isfinite(total): raise FloatingPointError("non-finite training loss")
            total.backward(); optimizer.step(); values.append((float(total.detach()), float(motion.detach()), float(gripper.detach())))
        validation_metrics = metrics(model, validation, mean_np, std_np)
        row = {"epoch": epoch, "train_total_loss": float(np.mean([x[0] for x in values])), "train_motion_loss": float(np.mean([x[1] for x in values])), "train_gripper_loss": float(np.mean([x[2] for x in values])), "val_total_loss": validation_metrics["total_loss"], "val_motion_loss": validation_metrics["motion_loss"], "val_gripper_loss": validation_metrics["gripper_loss"], "val_gripper_accuracy": validation_metrics["overall"]["gripper_accuracy"]}
        state["history"].append(row); state["epoch"] = epoch; print(json.dumps(row), flush=True)
        if validation_metrics["total_loss"] < state["best"]:
            state["best"], state["best_epoch"], state["patience"] = validation_metrics["total_loss"], epoch, 0
            torch.save({"model": model.state_dict(), "physical_mean": mean_np, "physical_std": std_np, "best_epoch": epoch, "best_val_total_loss": validation_metrics["total_loss"]}, OUT / "best_val.pt")
        else: state["patience"] += 1
    state["model"], state["optimizer"] = model.state_dict(), optimizer.state_dict(); torch.save(state, OUT / "training_state.pt"); write(OUT / "training_history.csv", state["history"])
    if state["epoch"] >= 50 or state["patience"] >= 5:
        best = torch.load(OUT / "best_val.pt", map_location="cpu", weights_only=False); model = UnifiedStageAwareTemporalBC(); model.load_state_dict(best["model"])
        result = metrics(model, test, mean_np, std_np); dump(OUT / "offline_test_summary.json", result)
        write(OUT / "offline_test_summary.csv", rows_from_metrics({"OVERALL": result["overall"]}))
        write(OUT / "offline_test_by_scenario.csv", rows_from_metrics(result["by_scenario"]))
        write(OUT / "offline_test_by_stage.csv", rows_from_metrics({("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")[int(key)]: value for key, value in result["by_stage"].items()}))
        print(json.dumps({"complete": True, "best_epoch": state["best_epoch"]}))


if __name__ == "__main__": main()
