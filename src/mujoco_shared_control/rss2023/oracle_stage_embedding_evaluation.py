"""V2 evaluator; execution semantics are shared with Oracle V1."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.oracle_stage_evaluation import evaluate_episode
from mujoco_shared_control.rss2023.global_evaluation import summarize
import json

class OracleStageEmbeddingPredictor:
    def __init__(self, checkpoint_path, normalization_path, *, device_name='auto'):
        self.device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
        payload=torch.load(Path(checkpoint_path),map_location=self.device,weights_only=False); cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in payload['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
        self.model=StageEmbeddingDiffusion(cfg).to(self.device).eval();self.model.load_state_dict(payload['model'])
        with np.load(normalization_path,allow_pickle=False) as stats:
            self.observation_mean=np.asarray(stats['observation_mean'],np.float32);self.observation_std=np.asarray(stats['observation_std'],np.float32);self.action_mean=np.asarray(stats['action_mean'],np.float32);self.action_std=np.asarray(stats['action_std'],np.float32)
        if self.observation_mean.shape!=(48,) or not np.array_equal(self.observation_mean[43:],np.zeros(5,np.float32)) or not np.array_equal(self.observation_std[43:],np.ones(5,np.float32)):raise ValueError('invalid V2 normalization')
        self.action_spec=ExpertActionSpec();self.generator=None
    def reset_sampling(self,seed): self.generator=torch.Generator(device=self.device).manual_seed(seed)
    @torch.inference_mode()
    def sample(self,observation_43,active_stage):
        if observation_43.shape!=(43,) or active_stage not in range(5):raise ValueError('invalid V2 stage input')
        observation=np.concatenate((observation_43.astype(np.float32),np.eye(5,dtype=np.float32)[active_stage]));norm=torch.from_numpy((observation-self.observation_mean)/self.observation_std).to(self.device).unsqueeze(0)
        action=self.model.assist(norm,torch.zeros((1,7),device=self.device),gamma=1.0,generator=self.generator).squeeze(0).cpu().numpy();return np.asarray(action*self.action_std+self.action_mean,np.float64)

def run_evaluation(checkpoint,normalization,output,*,formal_seeds=range(2_000_000,2_000_100),device='auto'):
    output.mkdir(parents=True,exist_ok=True);predictor=OracleStageEmbeddingPredictor(checkpoint,normalization,device_name=device);rows=[]
    for i,seed in enumerate(formal_seeds):rows.append(evaluate_episode(predictor,seed,8_000_000+seed));print(f'formal {i+1}/{len(formal_seeds)}',flush=True)
    report={'policy':'Oracle Stage Embedding Diffusion V2','checkpoint':str(Path(checkpoint).resolve()),'normalization':str(Path(normalization).resolve()),'policy_observation':'physical43_stage_embedding5','policy_uses_phase':True,'policy_uses_milestones':False,'future_stage_leakage':False,'summary':summarize(rows),'rows':rows};(output/'evaluation_report.json').write_text(json.dumps(report,indent=2)+'\n');return report
