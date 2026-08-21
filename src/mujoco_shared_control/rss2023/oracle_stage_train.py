"""Train the first Oracle current-stage-conditioned Global Diffusion."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.oracle_stage_dataset import (
    ORACLE_ACTION_DIM, ORACLE_OBSERVATION_DIM, PreparedOracleDataset,
    prepare_oracle_dataset,
)
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage, TrainingConfig


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _loader(prepared: PreparedOracleDataset, split_name: str, batch_size: int, shuffle: bool):
    split = getattr(prepared, split_name)
    observation = torch.from_numpy(prepared.observation_normalizer.normalize(split.observation))
    action = torch.from_numpy(prepared.action_normalizer.normalize(split.action))
    return DataLoader(TensorDataset(observation, action), batch_size=min(batch_size, len(split)), shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())


def _infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def _validation_loss(model, loader, device, max_batches: int) -> float:
    model.eval(); losses = []
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(12345)
        for index, (observation, action) in enumerate(loader):
            if index >= max_batches: break
            losses.append(float(model.loss(observation.to(device), action.to(device)).item()))
    model.train(); return float(np.mean(losses))


def _save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def train_oracle(dataset_dir: Path, output: Path, *, device_name: str = "auto", smoke_steps: int = 500, training_config: TrainingConfig = TrainingConfig(steps=80_000, validation_every=1_000, checkpoint_every=10_000), diffusion_config: DiffusionConfig | None = None) -> Path:
    prepared = prepare_oracle_dataset(dataset_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage_label_audit.json").write_text(json.dumps(prepared.stage_audit, indent=2) + "\n")
    with (output / "stage_distribution.csv").open("w") as f:
        f.write("split,stage0,stage1,stage2,stage3,stage4\n")
        for name in ("train", "validation", "test"):
            f.write(name + "," + ",".join(map(str, prepared.stage_audit[name]["stage_counts"])) + "\n")
    diffusion_config = diffusion_config or DiffusionConfig(observation_dim=ORACLE_OBSERVATION_DIM, action_dim=ORACLE_ACTION_DIM)
    training_config = TrainingConfig(**{**asdict(training_config), "steps": 80_000, "validation_every": 1_000, "checkpoint_every": 10_000})
    (output / "dataset_adapter_report.json").write_text(json.dumps(prepared.manifest(), indent=2) + "\n")
    np.savez(output / "normalization_stats.npz", observation_mean=prepared.observation_normalizer.mean, observation_std=prepared.observation_normalizer.std, action_mean=prepared.action_normalizer.mean, action_std=prepared.action_normalizer.std)
    global_config = json.loads((Path(__file__).resolve().parents[3] / "outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/training_config.json").read_text())
    oracle_config = {"implementation": "mujoco_shared_control.rss2023.model.RSS2023Diffusion", "observation_mode": "physical43_active_stage5", "diffusion": diffusion_config.state_dict(), "training": asdict(training_config), "test_split_used_for_selection": False}
    (output / "config.json").write_text(json.dumps(oracle_config, indent=2) + "\n")
    diff = {"global": global_config, "oracle": oracle_config, "only_intended_differences": ["diffusion.observation_dim: 43 -> 48", "observation semantics: physical43 -> physical43_active_stage5"]}
    (output / "global_vs_oracle_stage_config_diff.json").write_text(json.dumps(diff, indent=2) + "\n")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    _seed(7001); smoke = RSS2023Diffusion(diffusion_config).to(device).train(); opt = Adam(smoke.parameters(), lr=training_config.learning_rate); batches = _infinite(_loader(prepared, "train", training_config.batch_size, True)); last=float("nan")
    for _ in range(smoke_steps):
        o,a=next(batches); o,a=o.to(device),a.to(device); opt.zero_grad(set_to_none=True); loss=smoke.loss(o,a)
        if not torch.isfinite(loss): raise FloatingPointError("Oracle smoke loss is NaN/Inf")
        loss.backward(); torch.nn.utils.clip_grad_norm_(smoke.parameters(),1.0); opt.step(); last=float(loss.item())
    _save({"model":smoke.state_dict(),"config":diffusion_config.state_dict()},output/"smoke.pt")
    (output/"smoke_report.json").write_text(json.dumps({"status":"PASS","steps":smoke_steps,"last_loss":last,"nan_inf":0},indent=2)+"\n")
    _seed(training_config.seed); train_loader=_loader(prepared,"train",training_config.batch_size,True); val_loader=_loader(prepared,"validation",training_config.batch_size,False); batches=_infinite(train_loader); model=RSS2023Diffusion(diffusion_config).to(device).train(); optimizer=Adam(model.parameters(),lr=training_config.learning_rate); ema=ExponentialMovingAverage(model,training_config.ema_decay); best=float("inf"); final=float("nan"); window=[]; log_path=output/"training_log.jsonl"; started=time.monotonic()
    def payload(step,value): return {"format_version":"oracle-stage-rss2023-1.0","step":step,"validation_loss":value,"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"diffusion_config":diffusion_config.state_dict(),"training_config":asdict(training_config),"observation_normalizer":prepared.observation_normalizer.state_dict(),"action_normalizer":prepared.action_normalizer.state_dict(),"dataset_manifest":prepared.manifest()}
    with log_path.open("w") as log:
        for step in range(1,training_config.steps+1):
            o,a=next(batches); o,a=o.to(device),a.to(device); optimizer.zero_grad(set_to_none=True); loss=model.loss(o,a)
            if not torch.isfinite(loss): raise FloatingPointError(f"Oracle loss is NaN/Inf at step {step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); ema.update(model); window.append(float(loss.item()))
            validate=step==1 or step%training_config.validation_every==0 or step==training_config.steps
            if validate:
                final=_validation_loss(model,val_loader,device,training_config.validation_batches); record={"step":step,"train_loss":float(np.mean(window)),"validation_loss":final,"elapsed_seconds":time.monotonic()-started}; log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True);window.clear();_save(payload(step,final),output/"latest.pt")
                if final<best: best=final; _save(payload(step,final),output/"best.pt")
            if step%training_config.checkpoint_every==0: _save(payload(step,final),output/"checkpoints"/f"step_{step:08d}.pt")
    (output/"training_report.json").write_text(json.dumps({"status":"PASS","best_validation_loss":best,"final_validation_loss":final,"best_step":next((r["step"] for r in reversed([json.loads(x) for x in log_path.read_text().splitlines()]) if r["validation_loss"]==best),None),"steps":training_config.steps,"nan_inf":0,"device":str(device)},indent=2)+"\n")
    return output/"best.pt"
