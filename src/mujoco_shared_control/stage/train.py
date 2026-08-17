"""Train and evaluate Stage TCN V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from mujoco_shared_control.stage.dataset import (
    HISTORY, StageNormalization, StageWindowDataset, fit_normalization, load_split,
)
from mujoco_shared_control.stage.tcn import StageTCNV1


PHASE_NAMES = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")


@dataclass(frozen=True)
class StageTrainingConfig:
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stop_patience: int = 5
    seed: int = 42


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((5, 5), np.int64)
    for target, prediction in zip(labels, predictions): matrix[int(target), int(prediction)] += 1
    per_stage = {}
    precision_values = []; recall_values = []; f1_values = []
    for index, name in enumerate(PHASE_NAMES):
        tp = matrix[index, index]; fp = matrix[:, index].sum() - tp; fn = matrix[index].sum() - tp
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precision_values.append(precision); recall_values.append(recall); f1_values.append(f1)
        per_stage[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)), "macro_f1": float(np.mean(f1_values)),
        "per_stage": per_stage, "confusion_matrix": matrix.tolist(),
    }


@torch.no_grad()
def predict(model: StageTCNV1, dataset: StageWindowDataset, device: torch.device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = np.empty(len(dataset), np.int64)
    model.eval()
    for windows, _labels, indices in loader:
        logits = model(windows.to(device)); all_predictions[indices.numpy()] = logits.argmax(-1).cpu().numpy()
    return all_predictions


def recovery_metrics(dataset: StageWindowDataset, predictions: np.ndarray) -> dict[str, Any]:
    per_episode = {episode.episode_id: np.full(len(episode.labels), -1, np.int64) for episode in dataset.episodes}
    for index, prediction in enumerate(predictions):
        episode = dataset.episodes[int(dataset.episode_indices[index])]
        per_episode[episode.episode_id][int(dataset.time_indices[index])] = prediction
    results = {}
    all_delays = []
    type_for_old = {1: "GRASP_RECOVERY", 2: "TRANSPORT_DROP", 3: "PLACE_RECOVERY"}
    for old in (1, 2, 3):
        total = immediate = stable = 0; delays = []
        for episode in dataset.episodes:
            if episode.trajectory_type != type_for_old[old]: continue
            regressions = np.flatnonzero((episode.labels[:-1] == old) & (episode.labels[1:] == 0)) + 1
            episode_predictions = per_episode[episode.episode_id]
            for start in regressions:
                total += 1; immediate += int(episode_predictions[start] == 0)
                end = start
                while end < len(episode.labels) and episode.labels[end] == 0: end += 1
                found = None
                for index in range(start, max(start, end - 2)):
                    if np.all(episode_predictions[index:index + 3] == 0): found = index; break
                if found is not None:
                    stable += 1; delays.append(found - start); all_delays.append(found - start)
        results[f"{old}->0"] = {
            "count": total, "immediate_recognition_rate": immediate / total if total else 0.0,
            "stable_recognition_rate": stable / total if total else 0.0,
            "delay_steps": {"mean": float(np.mean(delays)) if delays else None,
                            "median": float(np.median(delays)) if delays else None,
                            "p95": float(np.percentile(delays, 95)) if delays else None},
        }
    delay = np.asarray(all_delays, np.float64)
    aggregate = {
        "mean_steps": float(delay.mean()), "median_steps": float(np.median(delay)),
        "p95_steps": float(np.percentile(delay, 95)),
        "mean_seconds": float(delay.mean() * 0.05), "median_seconds": float(np.median(delay) * 0.05),
        "p95_seconds": float(np.percentile(delay, 95) * 0.05),
    } if len(delay) else {}
    return {"by_regression": results, "aggregate_delay": aggregate, "stable_steps": 3, "control_dt": 0.05}


def train_stage_tcn(dataset_root: str | Path, output_dir: str | Path, config: StageTrainingConfig = StageTrainingConfig()) -> Path:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_episodes = load_split(dataset_root, "train"); val_episodes = load_split(dataset_root, "validation"); test_episodes = load_split(dataset_root, "test")
    normalization = fit_normalization(train_episodes)
    train_data = StageWindowDataset(train_episodes, normalization); val_data = StageWindowDataset(val_episodes, normalization); test_data = StageWindowDataset(test_episodes, normalization)
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "normalization_stats.npz", mean=normalization.mean, std=normalization.std,
             binary_feature_indices=np.asarray([17], np.int64))
    (output / "training_config.json").write_text(json.dumps({**asdict(config), "history_length": HISTORY,
        "architecture": "19->64(d1)->128(d2)->128(d4), residual ReLU dropout0.1, 128->64->5",
        "sampler": "70% ordinary / 30% phase-change-or-failure-nearby"}, indent=2) + "\n")
    sampler = WeightedRandomSampler(train_data.sampler_weights(), num_samples=len(train_data), replacement=True,
                                    generator=torch.Generator().manual_seed(config.seed))
    train_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=sampler)
    model = StageTCNV1().to(device); optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(); best_f1 = -1.0; best_epoch = 0; patience = 0; logs = []
    checkpoint_path = output / "checkpoint_best.pt"
    for epoch in range(1, config.max_epochs + 1):
        model.train(); losses = []
        for windows, labels, _indices in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(windows.to(device)); loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("TCN loss is NaN/Inf")
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        val_predictions = predict(model, val_data, device, config.batch_size)
        val_metrics = metrics(val_data.labels, val_predictions)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_metrics["macro_f1"],
                  "val_accuracy": val_metrics["accuracy"]}; logs.append(record); print(json.dumps(record), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]; best_epoch = epoch; patience = 0
            torch.save({"format_version": "stage-tcn-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "validation_macro_f1": best_f1, "config": asdict(config),
                        "normalization_mean": normalization.mean, "normalization_std": normalization.std,
                        "phase_names": PHASE_NAMES}, checkpoint_path)
        else:
            patience += 1
            if patience >= config.early_stop_patience: break
    (output / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    test_predictions = predict(model, test_data, device, config.batch_size)
    overall = metrics(test_data.labels, test_predictions)
    by_type = {}
    for trajectory_type in ("NORMAL", "GRASP_RECOVERY", "TRANSPORT_DROP", "PLACE_RECOVERY"):
        episode_ids = {index for index, episode in enumerate(test_data.episodes) if episode.trajectory_type == trajectory_type}
        mask = np.asarray([int(index) in episode_ids for index in test_data.episode_indices])
        by_type[trajectory_type] = metrics(test_data.labels[mask], test_predictions[mask])
    recovery = recovery_metrics(test_data, test_predictions)
    report = {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
              "test": overall, "by_trajectory_type": by_type, "recovery": recovery,
              "windows": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
              "nan_inf": 0, "device": str(device)}
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Checkpoint reload + posterior contract.
    reloaded = StageTCNV1(); reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]); reloaded.eval()
    posterior = reloaded.posterior(torch.from_numpy(test_data.windows[:4]))
    if posterior.shape != (4, 5) or not torch.isfinite(posterior).all(): raise RuntimeError("TCN checkpoint sample sanity failed")
    return checkpoint_path
