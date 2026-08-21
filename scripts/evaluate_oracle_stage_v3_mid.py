from pathlib import Path
from mujoco_shared_control.rss2023.oracle_stage_v3_mid_evaluation import run_evaluation
root=Path(__file__).resolve().parents[1]/'outputs/oracle_stage_diffusion/oracle_stage_v3_mid_20260819'
for step in range(10000,80001,10000):
 r=run_evaluation(root/'checkpoints'/f'step_{step:08d}.pt',root/'normalization_stats.npz',root/'eval'/f'step_{step:08d}',device='cpu');print(step,r['summary'])
