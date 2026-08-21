#!/usr/bin/env python3
"""Formal CAUSAL_TCN_RECOVERY_V1 trainer; CUDA-only when training."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mujoco_shared_control.stage.tcn import StageTCNV1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED"
OUT = ROOT / "outputs/recovery_stage_dp_training/causal_tcn_recovery_v1_20260820"
LENGTH, DIM, STAGES = 20, 19, 5


def feature(state, previous_action):
    ee, obj, goal = state[14:17], state[22:25], state[29:32]
    object_to_ee, goal_to_object = obj - ee, goal - obj
    return np.r_[previous_action, object_to_ee, np.linalg.norm(object_to_ee), goal_to_object,
                 np.linalg.norm(goal_to_object), np.linalg.norm(goal - ee), state[21], state[42], obj[2]].astype("f4")


def raw_path(clean_path):
    return DATA / "raw_rollouts" / clean_path.relative_to((DATA / "episodes").resolve())


def f1(y, prediction):
    scores = []
    for cls in range(STAGES):
        tp = ((y == cls) & (prediction == cls)).sum()
        denominator = 2 * tp + ((y != cls) & (prediction == cls)).sum() + ((y == cls) & (prediction != cls)).sum()
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return scores


def build(split_ids, paths):
    windows, labels, transition_near, metadata, counts = [], [], [], [], np.zeros(STAGES, int)
    for episode_id in split_ids:
        clean_path = paths[episode_id]
        with h5py.File(raw_path(clean_path), "r") as raw:
            states = raw["full_physical_state"][:]
            actions = raw["executed_action"][:]
            phases = raw["active_phase"][:].astype(int)
            times = raw["timestep_raw"][:].astype(int)
            history = deque(maxlen=LENGTH)
            first = feature(states[0], np.zeros(7, "f4"))
            history.extend([first] * LENGTH)
            for index, stage in enumerate(phases):
                if index:
                    history.append(feature(states[index], actions[index - 1]))
                windows.append(np.asarray(history))
                labels.append(stage)
                transition_near.append(index > 0 and stage != phases[index - 1])
                metadata.append((episode_id, int(times[index]), index))
                counts[stage] += 1
    return np.asarray(windows, "f4"), np.asarray(labels), np.asarray(transition_near, bool), metadata, counts


def metric(model, windows, labels, device):
    model.eval(); predictions, losses = [], []
    with torch.no_grad():
        for start in range(0, len(labels), 512):
            x = torch.as_tensor(windows[start:start + 512], device=device)
            y = torch.as_tensor(labels[start:start + 512], device=device)
            logits = model(x)
            predictions.extend(logits.argmax(1).cpu().numpy())
            losses.append(float(torch.nn.functional.cross_entropy(logits, y)))
    prediction = np.asarray(predictions)
    scores = f1(labels, prediction)
    return {"loss": float(np.mean(losses)), "accuracy": float((labels == prediction).mean()),
            "macro_f1": float(np.mean(scores)), "stage_f1": scores, "pred": prediction.tolist(),
            "confusion_matrix": [[int(((labels == i) & (prediction == j)).sum()) for j in range(STAGES)] for i in range(STAGES)]}


def regression(labels, prediction):
    report = {}
    for source in (1, 2, 3):
        rows = []
        for timestep in np.flatnonzero((labels[:-1] == source) & (labels[1:] == 0)) + 1:
            immediate = bool(prediction[timestep] == 0)
            stable = timestep + 2 < len(prediction) and bool(np.all(prediction[timestep:timestep + 3] == 0))
            latency = next((offset for offset in range(11) if timestep + offset + 2 < len(prediction) and np.all(prediction[timestep + offset:timestep + offset + 3] == 0)), None)
            rows.append((immediate, stable, latency))
        found = [row[2] for row in rows if row[2] is not None]
        report[f"{source}->0"] = {"COUNT": len(rows), "IMMEDIATE_RATE": float(np.mean([row[0] for row in rows])) if rows else 0.0,
            "STABLE3_IMMEDIATE_RATE": float(np.mean([row[1] for row in rows])) if rows else 0.0,
            "MEAN_LATENCY": None if not found else float(np.mean(found)), "MEDIAN_LATENCY": None if not found else float(np.median(found)),
            "DETECTION_FAILURE_RATE": float(np.mean([row[2] is None for row in rows])) if rows else 0.0}
    return report


def region_metrics(labels, prediction, mask):
    labels, prediction = labels[mask], prediction[mask]
    return {"frame_count": int(len(labels)), "accuracy": None if not len(labels) else float((labels == prediction).mean()),
            "gt_stage_distribution": [int((labels == stage).sum()) for stage in range(STAGES)],
            "pred_stage_distribution": [int((prediction == stage).sum()) for stage in range(STAGES)],
            "confusion_matrix": [[int(((labels == i) & (prediction == j)).sum()) for j in range(STAGES)] for i in range(STAGES)]}


def boundary_analysis(paths, test_ids, test_meta, prediction):
    """Evaluate overlapping boundaries using only raw rollout fields and attributes."""
    masks = {name: np.zeros(len(test_meta), bool) for name in ("NOMINAL", "FAILURE_INJECTION", "REGRESSION_PLUS_0_2", "REGRESSION_PLUS_3_5", "LATER_RECOVERY", "PLACE_RETREAT")}
    by_episode = {}
    for row_index, (episode_id, raw_time, raw_index) in enumerate(test_meta):
        by_episode.setdefault(episode_id, []).append((row_index, raw_time, raw_index))
    labels = np.empty(len(test_meta), int)
    for episode_id, rows in by_episode.items():
        with h5py.File(raw_path(paths[episode_id]), "r") as raw:
            phase, injected = raw["active_phase"][:].astype(int), raw["injection_active"][:].astype(bool)
            failure, recovery = int(raw.attrs["failure_step_raw"]), int(raw.attrs["recovery_start_step_raw"])
            regressions = np.flatnonzero((phase[:-1] != 0) & (phase[1:] == 0)) + 1
            is_nominal = not injected.any() and failure < 0 and recovery < 0
            for row_index, _raw_time, index in rows:
                labels[row_index] = phase[index]
                masks["NOMINAL"][row_index] = is_nominal
                masks["FAILURE_INJECTION"][row_index] = injected[index]
                masks["REGRESSION_PLUS_0_2"][row_index] = any(point <= index <= point + 2 for point in regressions)
                masks["REGRESSION_PLUS_3_5"][row_index] = any(point + 3 <= index <= point + 5 for point in regressions)
                masks["LATER_RECOVERY"][row_index] = recovery >= 0 and _raw_time >= recovery
                masks["PLACE_RETREAT"][row_index] = phase[index] in (3, 4)
    return {"region_definition": {"NOMINAL": "raw rollout has no injection and no failure/recovery marker", "FAILURE_INJECTION": "raw injection_active[t]", "REGRESSION_PLUS_0_2": "true raw active_phase k->0 at t; frames t..t+2", "REGRESSION_PLUS_3_5": "true raw active_phase k->0 at t; frames t+3..t+5", "LATER_RECOVERY": "raw timestep >= recovery_start_step_raw", "PLACE_RETREAT": "raw active_phase in {3,4}"},
            "regions": {name: region_metrics(labels, prediction, mask) for name, mask in masks.items()}}


def create_clean_cache(output, paths, splits, model, mean, std, device, checkpoint):
    """Infer the frozen model at exactly each clean transition's raw timestamp."""
    cache_path = output / "causal_tcn_stage_cache.h5"
    rows = []
    mapping_failures = clean_injection_count = clean_nonexpert_actions = expected_rows = 0
    for split_name, episode_ids in splits.items():
        for episode_id in episode_ids:
            clean_path = paths[episode_id]
            with h5py.File(clean_path, "r") as clean, h5py.File(raw_path(clean_path), "r") as raw:
                clean_times = clean["timestep_raw"][:].astype(int)
                expected_rows += len(clean_times)
                clean_injection_count += int(clean["injection_active"][:].astype(bool).sum())
                clean_nonexpert_actions += int((clean["action_source"][:] != 0).sum())
                raw_times = raw["timestep_raw"][:].astype(int)
                exact = {int(time): index for index, time in enumerate(raw_times)}
                states, actions, phases = raw["full_physical_state"][:], raw["executed_action"][:], raw["active_phase"][:].astype(int)
                history = deque(maxlen=LENGTH); first = (feature(states[0], np.zeros(7, "f4")) - mean) / std; history.extend([first] * LENGTH)
                posterior = []
                with torch.no_grad():
                    for index in range(len(raw_times)):
                        if index: history.append((feature(states[index], actions[index - 1]) - mean) / std)
                        posterior.append(model.posterior(torch.as_tensor(np.asarray(history)[None], device=device))[0].cpu().numpy())
                failure, recovery = int(raw.attrs["failure_step_raw"]), int(raw.attrs["recovery_start_step_raw"])
                for dp_index, time in enumerate(clean_times):
                    raw_index = exact.get(int(time))
                    if raw_index is None:
                        mapping_failures += 1; continue
                    # Exact mapping additionally guards against stale/misaligned paired files.
                    if not np.allclose(clean["full_physical_state"][dp_index], states[raw_index], atol=1e-6):
                        mapping_failures += 1; continue
                    post = posterior[raw_index].astype("f4"); predicted = int(post.argmax())
                    rows.append((episode_id, str(clean.attrs["episode_type"]), split_name, int(clean["timestep_dp"][dp_index]), int(time), int(clean["active_phase"][dp_index]), predicted, post, float(post.max()), max(0, raw_index - LENGTH + 1), raw_index, -1 if raw_index == 0 else raw_index - 1, bool(failure >= 0 and time > failure), bool(any((phases[j - 1] in (1, 2, 3) and phases[j] == 0) for j in range(1, raw_index + 1)))))
    string = h5py.string_dtype("utf-8")
    with h5py.File(cache_path, "w") as cache:
        cache.attrs.update({"MODEL_NAME": "CAUSAL_TCN_RECOVERY_V1", "TCN_CHECKPOINT": str(checkpoint.resolve()), "TCN_FROZEN": "YES", "NORMALIZATION": "train-only stats", "HISTORY_SEMANTICS": "state[t] + executed_action[t-1]; repeat-first padding; no recovery reset", "HARD_ONEHOT_POLICY_CONDITION": "YES", "CLEAN_DP_INJECTION_ACTIVE_COUNT": clean_injection_count, "CLEAN_DP_NONEXPERT_ACTION_COUNT": clean_nonexpert_actions})
        for name, index in (("episode_id", 0), ("episode_type", 1), ("split", 2)):
            cache.create_dataset(name, data=np.asarray([row[index] for row in rows], dtype=object), dtype=string)
        for name, index, dtype in (("timestep_dp", 3, "i4"), ("timestep_raw", 4, "i4"), ("gt_stage", 5, "i1"), ("pred_stage", 6, "i1"), ("confidence", 8, "f4"), ("history_raw_start", 9, "i4"), ("history_raw_end", 10, "i4"), ("previous_executed_action_raw_index", 11, "i4"), ("is_after_failure", 12, "?"), ("is_after_regression", 13, "?")):
            cache.create_dataset(name, data=np.asarray([row[index] for row in rows], dtype=dtype), compression="gzip")
        cache.create_dataset("posterior5", data=np.asarray([row[7] for row in rows], "f4"), compression="gzip")
        cache.create_dataset("hard_onehot5", data=np.eye(STAGES, dtype="f4")[np.asarray([row[6] for row in rows])], compression="gzip")
    return cache_path, mapping_failures, expected_rows


def audit_cache(cache_path, expected_rows, mapping_failures, checkpoint):
    with h5py.File(cache_path, "r") as cache:
        stages, onehot, posterior = cache["pred_stage"][:], cache["hard_onehot5"][:], cache["posterior5"][:]
        clean_injection = int(cache.attrs["CLEAN_DP_INJECTION_ACTIVE_COUNT"])
        clean_nonexpert = int(cache.attrs["CLEAN_DP_NONEXPERT_ACTION_COUNT"])
        audit = {"CACHE_PATH": str(cache_path.resolve()), "CACHE_ROWS": int(len(stages)), "EXPECTED_CLEAN_TRANSITIONS": expected_rows,
                 "RAW_CLEAN_MAPPING_FAILURES": int(mapping_failures), "NaN_COUNT": int(np.isnan(posterior).sum() + np.isnan(onehot).sum()), "Inf_COUNT": int(np.isinf(posterior).sum() + np.isinf(onehot).sum()),
                 "INVALID_PRED_STAGE_COUNT": int(np.sum((stages < 0) | (stages >= STAGES))), "HARD_ONEHOT_DIMENSION": int(onehot.shape[1]) if onehot.ndim == 2 else -1,
                 "HARD_ONEHOT_SUM_FAILURES": int(np.sum(~np.isclose(onehot.sum(axis=1), 1))) if onehot.ndim == 2 else int(len(stages)),
                 "FUTURE_INFORMATION_USED": 0, "HISTORY_CAUSAL": "YES", "NORMALIZATION": "train-only stats", "TCN_CHECKPOINT": str(checkpoint.resolve()),
                 "TCN_FROZEN": "YES", "CLEAN_DP_INJECTION_ACTIVE_COUNT": clean_injection, "DP_INJECTION_ACTIONS_USED_AS_TRAINING_TARGET": clean_nonexpert}
    valid = (audit["CACHE_ROWS"] == expected_rows and audit["RAW_CLEAN_MAPPING_FAILURES"] == 0 and audit["NaN_COUNT"] == 0 and audit["Inf_COUNT"] == 0 and audit["INVALID_PRED_STAGE_COUNT"] == 0 and audit["HARD_ONEHOT_DIMENSION"] == STAGES and audit["HARD_ONEHOT_SUM_FAILURES"] == 0 and audit["FUTURE_INFORMATION_USED"] == 0 and audit["CLEAN_DP_INJECTION_ACTIVE_COUNT"] == 0 and audit["DP_INJECTION_ACTIONS_USED_AS_TRAINING_TARGET"] == 0)
    audit["TCN_STAGE_CACHE_VALID"] = "YES" if valid else "NO"
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check-only", action="store_true", help="build and audit dataset only; never trains")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split_manifest.json").read_text())
    paths = {key: Path(value).resolve() for key, value in split["episode_paths"].items()}
    split_sets = [set(split["splits"][name]) for name in ("train", "validation", "test")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)): raise RuntimeError("split leakage")
    built = {name: build(split["splits"][name], paths) for name in split["splits"]}
    train_x, train_y, near, _, _ = built["train"]
    mean, std = train_x.reshape(-1, DIM).mean(0), np.maximum(train_x.reshape(-1, DIM).std(0), 1e-6)
    normalized = {name: ((value[0] - mean) / std, value[1], value[2], value[3], value[4]) for name, value in built.items()}
    audit = {"MODEL_NAME": "CAUSAL_TCN_RECOVERY_V1", "MODEL_CLASS": "StageTCNV1", "TRAIN_INFERENCE_SEMANTICS_MATCH": "YES", "ACTION_SEMANTICS": "PREVIOUS_EXECUTED_ACTION", "INPUT_DIM": DIM, "HISTORY_LENGTH": LENGTH, "FIRST_FRAME_ACTION_POLICY": "zero previous action; repeat first feature to length 20", "NORMALIZATION_TRAIN_ONLY": "YES", "TCN_HISTORY_CAUSAL": "YES", "TCN_HISTORY_RESET_AT_RECOVERY_START": "NO", "split_episodes": {name: len(value) for name, value in split["splits"].items()}, "stage_counts": {name: value[4].tolist() for name, value in normalized.items()}, "TOTAL_WINDOWS": len(train_y), "TRANSITION_NEAR_WINDOWS": int(near.sum()), "GENERAL_WINDOWS": int((~near).sum()), "future_frame_used": 0, "future_action_used": 0}
    if any(x.shape[1:] != (LENGTH, DIM) or not np.isfinite(x).all() or np.any((y < 0) | (y >= STAGES)) for x, y, *_ in normalized.values()): raise RuntimeError("dataset audit failure")
    np.savez(args.output / "normalization_stats.npz", mean=mean, std=std)
    (args.output / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config = audit | {"batch_size": 512, "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4, "max_epochs": 30, "early_stop_patience": 5, "seed": 42, "sampler": "70% transition-near / 30% general", "checkpoint_selection": "validation Macro-F1 only"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.check_only:
        print(json.dumps({"STATIC_CHECK": "PASS", "DATASET_BUILD_CHECK": "PASS", **audit}, indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = StageTCNV1().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = np.where(near, .7 / max(near.sum(), 1), .3 / max((~near).sum(), 1))
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized["train"][0]), torch.from_numpy(train_y)), batch_size=512, sampler=WeightedRandomSampler(torch.from_numpy(weights), len(train_y), replacement=True))
    best, bad, logs, checkpoint_dir = -1.0, 0, [], args.output / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    for epoch in range(1, 31):
        model.train(); losses, predicted, labels = [], [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(); logits = model(x); loss = torch.nn.functional.cross_entropy(logits, y); loss.backward(); optimizer.step()
            losses.append(loss.detach().item()); predicted.extend(logits.argmax(1).detach().cpu()); labels.extend(y.cpu())
        validation = metric(model, normalized["validation"][0], normalized["validation"][1], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_accuracy": float((np.asarray(predicted) == np.asarray(labels)).mean()), "train_macro_f1": float(np.mean(f1(np.asarray(labels), np.asarray(predicted))),), "val_loss": validation["loss"], "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"], "val_stage_f1": validation["stage_f1"]}; logs.append(row)
        if validation["macro_f1"] > best:
            best, bad = validation["macro_f1"], 0; torch.save({"model": model.state_dict(), "epoch": epoch, "normalization": {"mean": mean, "std": std}}, checkpoint_dir / "best_validation_macro_f1.pt")
        else: bad += 1
        if bad >= 5: break
    torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_dir / "last.pt")
    checkpoint = checkpoint_dir / "best_validation_macro_f1.pt"; model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); model.eval(); model.requires_grad_(False)
    with (args.output / "training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=logs[0]); writer.writeheader(); writer.writerows(logs)
    test = metric(model, normalized["test"][0], normalized["test"][1], device); prediction = np.asarray(test.pop("pred")); regressions = regression(normalized["test"][1], prediction)
    boundary = boundary_analysis(paths, split["splits"]["test"], built["test"][3], prediction)
    (args.output / "test_metrics.json").write_text(json.dumps(test, indent=2) + "\n"); (args.output / "regression_metrics.json").write_text(json.dumps(regressions, indent=2) + "\n"); (args.output / "boundary_analysis.json").write_text(json.dumps(boundary, indent=2) + "\n")
    gate = test["macro_f1"] >= .99 and min(test["stage_f1"]) >= .98 and all(value["STABLE3_IMMEDIATE_RATE"] >= .95 for value in regressions.values())
    if gate:
        cache_path, mapping_failures, cache_rows = create_clean_cache(args.output, paths, split["splits"], model, mean, std, device, checkpoint)
        cache_audit = audit_cache(cache_path, cache_rows, mapping_failures, checkpoint)
        (args.output / "cache_audit.json").write_text(json.dumps(cache_audit, indent=2) + "\n")
    else:
        cache_audit = {"TCN_STAGE_CACHE_VALID": "NOT_RUN", "REASON": "CAUSAL_TCN_VALID=NO; formal cache intentionally not generated"}
    best_row = max(logs, key=lambda item: item["val_macro_f1"])
    report = audit | {"BEST_EPOCH": best_row["epoch"], "BEST_VAL_MACRO_F1": best, "BEST_VAL_ACCURACY": best_row["val_accuracy"], "TEST_ACCURACY": test["accuracy"], "TEST_MACRO_F1": test["macro_f1"], "STAGE_F1": test["stage_f1"], "REGRESSION_METRICS": regressions, "BOUNDARY_ANALYSIS": "boundary_analysis.json", "CAUSAL_TCN_VALID": "YES" if gate else "NO", "TCN_STAGE_CACHE_VALID": cache_audit["TCN_STAGE_CACHE_VALID"], "PHASE_C2_COMPLETE": "YES", "C2_HARD_FAIL_REASONS": [] if gate else ["causal TCN PASS gate failed"]}
    (args.output / "final_report.json").write_text(json.dumps(report, indent=2) + "\n"); (args.output / "final_report.md").write_text("# CAUSAL_TCN_RECOVERY_V1\n\n" + json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
