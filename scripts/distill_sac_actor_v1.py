#!/usr/bin/env python3
"""Full mean-path action-space distillation from frozen BC to SAC Actor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.actor_bc.model import ActorBC
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.actor import (
    LOG_STD_INIT,
    SACGaussianActor,
    configure_full_mean_path_distillation,
    initialize_from_bc,
)


SEED = 20260812
CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")


def load_states(dataset: ManifestActorDataset) -> np.ndarray:
    values=[]
    for entry in dataset.entries:
        with h5py.File(entry.path,"r") as episode:
            values.append(np.asarray(episode["observations/policy_state_42"][:],np.float32))
    return np.concatenate(values)


def metrics(reference: np.ndarray, action: np.ndarray) -> dict:
    error=action-reference; absolute=np.abs(error)
    return {
        "normalized_mse":float(np.mean(error**2)),
        "normalized_mae":float(absolute.mean()),
        "normalized_max_abs_error":float(absolute.max()),
        "per_dimension_mae":absolute.mean(axis=0).tolist(),
        "xyz_physical_vector_error_m":float(np.linalg.norm(error[:,:3]*.025,axis=1).mean()),
        "rotation_physical_vector_error_rad":float(np.linalg.norm(error[:,3:6]*.10,axis=1).mean()),
        "gripper_physical_mae_m":float((absolute[:,6]*.04).mean()),
    }


def atomic_save(payload:dict,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".inprogress")
    torch.save(payload,temporary); os.replace(temporary,path)


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id"); args=parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    mean=np.asarray(checkpoint["observation_mean"],np.float32)
    std=np.asarray(checkpoint["observation_std"],np.float32)
    train_dataset=ManifestActorDataset(MANIFEST,"train")
    validation_dataset=ManifestActorDataset(MANIFEST,"validation")
    if (len(train_dataset.entries),len(train_dataset))!=(900,115021): raise ValueError("train split changed")
    if (len(validation_dataset.entries),len(validation_dataset))!=(100,12817): raise ValueError("validation split changed")
    train_state=torch.from_numpy((load_states(train_dataset)-mean)/std)
    validation_state=torch.from_numpy((load_states(validation_dataset)-mean)/std)

    teacher=ActorBC(); teacher.load_state_dict(checkpoint["model_state_dict"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    student=SACGaussianActor(); initialize_from_bc(student,CHECKPOINT,option="direct_head_copy",log_std_init=LOG_STD_INIT)
    log_std_before=deepcopy(student.log_std_head.state_dict())
    initial_state=deepcopy(student.state_dict())
    trainable=configure_full_mean_path_distillation(student)
    trainable_names=[name for name,p in student.named_parameters() if p.requires_grad]
    print("trainable_parameters="+json.dumps(trainable_names),flush=True)
    with torch.inference_mode():
        train_target=torch.clamp(teacher(train_state),-1,1)
        val_target=torch.clamp(teacher(validation_state),-1,1)
    config={
        "objective":"MSE(tanh(mu_student), clip(a_bc_raw,-1,1))",
        "optimizer":"Adam","learning_rate":1e-4,"batch_size":256,
        "max_epochs":100,"early_stopping_patience":12,
        "minimum_improvement":1e-9,"gradient_clip_norm":1.0,
        "training_seed":SEED,"log_std_init":LOG_STD_INIT,
        "log_std_bounds":[-5.0,2.0],"trainable_parameters":trainable_names,
    }
    loader=DataLoader(TensorDataset(train_state,train_target),batch_size=256,shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
    optimizer=torch.optim.Adam(trainable,lr=config["learning_rate"])
    best=float("inf"); best_epoch=0; best_state=None; stale=0; history=[]
    for epoch in range(1,config["max_epochs"]+1):
        student.train(); total=0.0
        for state,target in loader:
            prediction=student.deterministic_action(state)
            loss=nn.functional.mse_loss(prediction,target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable,config["gradient_clip_norm"]); optimizer.step()
            total+=float(loss.detach())*len(state)
        student.eval()
        with torch.inference_mode():
            val_action=student.deterministic_action(validation_state)
            val_loss=float(nn.functional.mse_loss(val_action,val_target))
        train_loss=total/len(train_state)
        history.append({"epoch":epoch,"train_action_mse":train_loss,"validation_action_mse":val_loss})
        if val_loss<best-config["minimum_improvement"]:
            best=val_loss;best_epoch=epoch;best_state=deepcopy(student.state_dict());stale=0
        else: stale+=1
        print(f"epoch={epoch:03d} train={train_loss:.9g} val={val_loss:.9g} best={best_epoch}",flush=True)
        if stale>=config["early_stopping_patience"]:break
    if best_state is None:raise RuntimeError("no best student")
    student.load_state_dict(best_state);student.eval()
    for key,value in student.log_std_head.state_dict().items():
        torch.testing.assert_close(value,log_std_before[key],rtol=0,atol=0)
    changed={name:not torch.equal(value,initial_state[name]) for name,value in student.state_dict().items()}
    if not all(changed[name] for name in changed if name.startswith(("trunk.","mean_head."))):
        raise RuntimeError("not every full mean-path tensor changed")
    if any(changed[name] for name in changed if name.startswith("log_std_head.")):
        raise RuntimeError("log_std head changed")
    with torch.inference_mode():
        action=student.deterministic_action(validation_state).numpy()
        mu,log_std,std_output=student.distribution_stats(validation_state)
    reference=val_target.numpy(); boundary=reference[:,6]>.95
    report={
        "samples":len(reference),"reference":"clip(raw BC output,-1,1)",
        "full_distillation":metrics(reference,action),
        "comparisons":{
            "direct_copy":{"normalized_mae":0.01874850131571293,"gripper_physical_mae_m":0.004702914971858263,"success":0},
            "frozen_trunk_atanh":{"normalized_mae":0.005390338134020567,"gripper_physical_mae_m":0.0013789103832095861,"success":4},
        },
        "open_gripper_boundary":{
            "definition":"teacher gripper > 0.95","sample_count":int(boundary.sum()),
            "teacher_mean":float(reference[boundary,6].mean()),
            "student_mean":float(action[boundary,6].mean()),
            "mae":float(np.abs(action[boundary,6]-reference[boundary,6]).mean()),
            "max_error":float(np.abs(action[boundary,6]-reference[boundary,6]).max()),
        },
        "mean_stats":{"min":float(mu.min()),"max":float(mu.max()),
            "absolute_p95":float(torch.quantile(mu.abs(),.95)),"absolute_p99":float(torch.quantile(mu.abs(),.99)),
            "absolute_max":float(mu.abs().max()),"gripper_min":float(mu[:,6].min()),
            "gripper_max":float(mu[:,6].max()),"gripper_absolute_p95":float(torch.quantile(mu[:,6].abs(),.95)),
            "gripper_absolute_p99":float(torch.quantile(mu[:,6].abs(),.99)),"gripper_absolute_max":float(mu[:,6].abs().max())},
        "log_std_stats":{"mean":float(log_std.mean()),"std":float(log_std.std(unbiased=False)),
            "min":float(log_std.min()),"max":float(log_std.max()),"exp_mean":float(std_output.mean())},
        "finite":bool(np.isfinite(action).all() and torch.isfinite(mu).all() and torch.isfinite(log_std).all()),
    }
    run_id=args.run_id or datetime.now(timezone.utc).strftime("sac_actor_v1_full_distill_%Y%m%dT%H%M%SZ")
    run=Path("outputs/sac_actor")/run_id;run.mkdir(parents=True,exist_ok=False)
    payload={"format_version":"sac_actor_v1_full_mean_path_distilled","actor_state_dict":student.state_dict(),
        "observation_mean":mean,"observation_std":std,"action_spec":checkpoint["action_spec"],
        "action_normalization_definition":checkpoint["action_normalization_definition"],
        "bc_checkpoint":str(CHECKPOINT.resolve()),"manifest_path":str(MANIFEST.resolve()),
        "manifest_file_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_content_sha":checkpoint["manifest_content_sha"],"distillation_config":config,
        "best_epoch":best_epoch,"best_validation_action_mse":best,"action_equivalence":report}
    atomic_save(payload,run/"actor_initialized.pt")
    (run/"distillation_config.json").write_text(json.dumps({**config,"run_id":run_id,
        "bc_checkpoint":str(CHECKPOINT.resolve()),"manifest":str(MANIFEST.resolve()),
        "train_episodes":900,"train_transitions":115021,"validation_episodes":100,
        "validation_transitions":12817,"best_epoch":best_epoch,"epochs_completed":epoch},indent=2)+"\n")
    (run/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
    (run/"action_equivalence.json").write_text(json.dumps(report,indent=2)+"\n")
    # Save/reload deterministic identity.
    loaded=SACGaussianActor();loaded.load_state_dict(torch.load(run/"actor_initialized.pt",map_location="cpu",weights_only=False)["actor_state_dict"]);loaded.eval()
    with torch.inference_mode():torch.testing.assert_close(loaded.deterministic_action(validation_state[:128]),student.deterministic_action(validation_state[:128]),rtol=0,atol=0)
    print(f"run_dir={run.resolve()}")


if __name__=="__main__":main()
