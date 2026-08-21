"""Frozen, label-minimal Stage-Aware Recovery Reward V1.1."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StageAwareRecoveryRewardV11Config:
    step_penalty: float = -0.001
    progress_clip: float = .02
    success_bonus: float = 5.0

class StageAwareRecoveryRewardCalculator:
    def __init__(self, config: StageAwareRecoveryRewardV11Config = StageAwareRecoveryRewardV11Config()): self.config=config
    @staticmethod
    def distances(s):
        return (float(np.linalg.norm(s[14:17]-s[22:25])), float(np.linalg.norm(s[22:25]-s[29:32])), float(s[24]))
    def transition(self, state, next_state, phase, next_phase, event):
        ee,goal,h=self.distances(state); nee,ngoal,nh=self.distances(next_state)
        progress=0.0
        if int(phase)==int(next_phase):
            if phase==0: progress=2*(ee-nee)
            elif phase==1: progress=(ee-nee)-2*(nh-h)
            elif phase in (2,3): progress=2*(goal-ngoal)
            elif phase==4: progress=2*(nee-ee)
            progress=float(np.clip(progress,-self.config.progress_clip,self.config.progress_clip))
        success=int(event)==4
        return {'step':self.config.step_penalty,'progress':progress,'success':self.config.success_bonus if success else 0.,'reward':self.config.step_penalty+progress+(self.config.success_bonus if success else 0.),'done':success,'injected':int(event) in (1,2,3)}
