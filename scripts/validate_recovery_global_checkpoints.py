#!/usr/bin/env python3
"""Fixed-case, stage-free validation of Recovery-aware Global DP checkpoints.

The stage-validation manifest is read only.  The environment retains its
tracker to inject/replay failures, but ``GlobalPredictor.sample`` accepts only
the 43D physical state; the stage argument is intentionally absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import validate_recovery_stage_checkpoints as protocol
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "outputs/recovery_stage_dp_training/recovery_global_120k_20260820"
CASES = ROOT / "outputs/recovery_stage_dp_validation_80k_120k"
OUTPUT_DEFAULT = ROOT / "outputs/recovery_global_dp_validation_80k_120k"
STEPS = tuple(range(10_000, 120_001, 10_000))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GlobalPredictor:
    """A policy adapter whose callable input is exclusively physical43."""
    def __init__(self, checkpoint: Path, normalization: Path, device: torch.device):
        self.device = device
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = DiffusionConfig(**payload["diffusion_config"])
        if config.observation_dim != 43 or config.action_dim != 7:
            raise ValueError("Global checkpoint must be 43D physical -> 7D action")
        self.model = RSS2023Diffusion(config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        with np.load(normalization, allow_pickle=False) as stats:
            self.pm = np.asarray(stats["physical_mean"], np.float32)
            self.ps = np.asarray(stats["physical_std"], np.float32)
            self.am = np.asarray(stats["action_mean"], np.float32)
            self.ass = np.asarray(stats["action_std"], np.float32)
        if self.pm.shape != (43,) or self.am.shape != (7,):
            raise ValueError("normalization shape mismatch")
        self.generator = None
        self.spec = ExpertActionSpec()

    def reset(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self, physical: np.ndarray) -> np.ndarray:
        obs = (np.asarray(physical, np.float32) - self.pm) / self.ps
        out = self.model.assist(torch.as_tensor(obs, device=self.device).unsqueeze(0), torch.zeros((1, 7), device=self.device), gamma=1.0, generator=self.generator)
        return (out.squeeze(0).cpu().numpy() * self.ass + self.am).astype(np.float32)


def evaluate_case(case, predictor: GlobalPredictor, trace_dir: Path):
    """Same simulator protocol as V1/V2; tracker output never reaches policy."""
    original_predict = predictor.sample
    # The shared evaluator calls sample(physical, stage); this adapter makes the
    # ignored evaluator stage impossible to enter GlobalPredictor itself.
    predictor.sample = lambda physical, _stage: original_predict(physical)
    try:
        return protocol.evaluate_case(case, predictor, trace_dir)
    finally:
        predictor.sample = original_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cases", type=Path, default=CASES, help="directory containing the frozen validation_case_manifest.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    manifest = args.cases.resolve() / "validation_case_manifest.json"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text())
    if payload.get("independent_from_e2_formal") is not True:
        raise RuntimeError("STOP: validation manifest is not independent from E2 formal")
    cases = payload["cases"]
    counts = {kind: sum(c["kind"] == kind for c in cases) for kind in protocol.KINDS}
    expected_n = int(payload["N_per_kind"])
    if counts != {kind: expected_n for kind in protocol.KINDS}: raise RuntimeError(f"case count mismatch: {counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation_protocol.json").write_text(json.dumps({"status": "PASS", "independent_from_e2_formal": True, "N_per_kind": expected_n, "validation_case_manifest": str(manifest.resolve()), "validation_case_manifest_sha256": sha(manifest), "same_cases_as_v1_v2": True, "GLOBAL_POLICY_STAGE_INPUT": "NO", "checkpoint_steps": list(STEPS), "E2_RUN": "NOT_RUN"}, indent=2) + "\n")
    results = []
    normalization = TRAIN / "normalization_stats.npz"
    for step in STEPS:
        checkpoint = TRAIN / "checkpoints" / f"step_{step:06d}.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        predictor = GlobalPredictor(checkpoint, normalization, device); rows = []
        for i, case in enumerate(cases):
            rows.append(evaluate_case(case, predictor, args.output / "traces" / f"GLOBAL_{step:06d}"))
            if (i + 1) % 25 == 0: print({"model": "GLOBAL", "step": step, "completed": i + 1, "total": len(cases)}, flush=True)
        grouped = {kind: protocol.summarize([r for r in rows if r["kind"] == kind]) for kind in protocol.KINDS}
        recovery = float(np.mean([grouped[k]["success"] for k in protocol.KINDS[1:]])); overall = float(np.mean([grouped[k]["success"] for k in protocol.KINDS]))
        report = {"model": "Global", "step": step, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "groups": grouped, "RecoveryMean": recovery, "OverallMean": overall, "GLOBAL_POLICY_STAGE_INPUT": "NO", "rows": rows}
        reports = args.output / "reports"; reports.mkdir(exist_ok=True); (reports / f"Global_{step:06d}.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append({"Model": "Global", "Step": step, "Normal": grouped["NORMAL"]["success"], "GraspRec": grouped["GRASP_RECOVERY"]["success"], "TransportRec": grouped["TRANSPORT_RECOVERY"]["success"], "PlaceRec": grouped["PLACE_RECOVERY"]["success"], "RecoveryMean": recovery, "OverallMean": overall, "IllegalDrop": float(np.mean([r["illegal_drop"] for r in rows])), "IK": float(np.mean([r["ik"] for r in rows])), "Timeout": float(np.mean([r["timeout"] for r in rows]))})
    with (args.output / "checkpoint_validation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    recovery_best = max(results, key=lambda r: (r["RecoveryMean"], r["OverallMean"]))["Step"]
    overall_best = max(results, key=lambda r: (r["OverallMean"], r["RecoveryMean"]))["Step"]
    (args.output / "selection_summary.json").write_text(json.dumps({"VALIDATION_PROTOCOL_VALID": "YES", "GLOBAL_POLICY_STAGE_INPUT": "NO", "BEST_GLOBAL_RECOVERY_CHECKPOINT": recovery_best, "BEST_GLOBAL_OVERALL_CHECKPOINT": overall_best, "E2_RUN": "NOT_RUN", "results": results}, indent=2) + "\n")
    print(json.dumps({"VALIDATION_PROTOCOL_VALID": "YES", "GLOBAL_POLICY_STAGE_INPUT": "NO", "BEST_GLOBAL_RECOVERY_CHECKPOINT": recovery_best, "BEST_GLOBAL_OVERALL_CHECKPOINT": overall_best, "E2_RUN": "NOT_RUN"}, indent=2))


if __name__ == "__main__":
    main()
