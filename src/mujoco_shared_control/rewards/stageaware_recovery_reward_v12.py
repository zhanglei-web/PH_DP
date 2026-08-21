"""Sparse active-stage-edge reward that needs no milestone reconstruction."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StageAwareRecoveryRewardV12Config:
    step_penalty: float=-.001
    success_bonus: float=5.
    edge_bonus: tuple[float,float,float,float]=(.5,.5,.5,.75)

class RewardBookkeeping:
    def __init__(self): self.given=[False]*4

class StageAwareRecoveryRewardV12:
    def __init__(self,config=StageAwareRecoveryRewardV12Config()):self.config=config
    def transition(self,phase,next_phase,event,book):
        edge=(int(phase),int(next_phase)); bonus=0.; index=edge[0] if edge in ((0,1),(1,2),(2,3),(3,4)) else None
        if index is not None and not book.given[index]:bonus=self.config.edge_bonus[index];book.given[index]=True
        success=int(event)==4
        return {'step':self.config.step_penalty,'edge_bonus':bonus,'success':self.config.success_bonus if success else 0.,'reward':self.config.step_penalty+bonus+(self.config.success_bonus if success else 0.),'done':success,'injected':int(event) in (1,2,3),'edge':edge}
