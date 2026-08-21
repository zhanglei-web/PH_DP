"""Shared, explicit state builders for distinct 48-D AWAC semantics."""
from __future__ import annotations
import numpy as np

STATE_MODE_PHYSICAL43 = "physical43"
STATE_MODE_MILESTONES5 = "physical43_milestones5"
STATE_MODE_ACTIVE_STAGE5 = "physical43_active_stage5"

def build_stageaware_state48(physical43, current_active_stage, physical_mean43, physical_std43):
    physical=np.asarray(physical43,np.float32);mean=np.asarray(physical_mean43,np.float32);std=np.asarray(physical_std43,np.float32)
    stage=int(current_active_stage)
    if physical.shape!=(43,) or mean.shape!=(43,) or std.shape!=(43,) or stage not in range(5):raise ValueError("requires physical43, train physical normalizer43, and active stage 0..4")
    return np.r_[(physical-mean)/std,np.eye(5,dtype=np.float32)[stage]].astype(np.float32)
