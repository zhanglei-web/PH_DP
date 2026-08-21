import numpy as np
from mujoco_shared_control.rewards.stageaware_recovery_reward import StageAwareRecoveryRewardCalculator

def s(ee=(0,0,0),obj=(0,0,0),goal=(1,0,0)):
 x=np.zeros(43,np.float32);x[14:17]=ee;x[22:25]=obj;x[29:32]=goal;return x
def test_v11_stage_switch_injection_and_success():
 r=StageAwareRecoveryRewardCalculator()
 assert r.transition(s((1,0,0)),s((.9,0,0)),0,0,0)['progress']>0
 assert r.transition(s((.9,0,0)),s((1,0,0)),4,4,0)['progress']>0
 assert r.transition(s(),s(),3,0,3)['progress']==0
 assert not r.transition(s(),s(),0,0,1)['done']
 assert r.transition(s(),s(),0,0,4)['done']
