#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json
from mujoco_shared_control.rss2023.oracle_stage_embedding_evaluation import run_evaluation
ROOT=Path(__file__).resolve().parents[1];TRAINING=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'
def main():
 p=argparse.ArgumentParser();p.add_argument('--training-dir',type=Path,default=TRAINING);p.add_argument('--device',default='auto');p.add_argument('--start-seed',type=int,default=2_000_000);p.add_argument('--count',type=int,default=100);a=p.parse_args();rows=[]
 for step in range(10000,80001,10000):
  ck=a.training_dir/'checkpoints'/f'step_{step:08d}.pt';rep=run_evaluation(ck,a.training_dir/'normalization_stats.npz',a.training_dir/'checkpoint_sweep_n100'/ck.stem,formal_seeds=range(a.start_seed,a.start_seed+a.count),device=a.device);s=rep['summary'];rows.append({'checkpoint':ck.name,'step':step,'success':s['success']['count'],'grasp':s['grasp']['count'],'lift':s['lift']['count'],'transport':s['transport']['count'],'place':s['place']['count'],'release':s['release']['count'],'retreat':s['retreat']['count'],'illegal_drop':s['illegal_drop']['count'],'ik_failure':s['ik_failure']['count'],'timeout':s['timeout']['count'],'mean_episode_length':s['episode_length']['mean'],'average_return':s['average_return']});print(rows[-1],flush=True)
 ck=a.training_dir/'best.pt';rep=run_evaluation(ck,a.training_dir/'normalization_stats.npz',a.training_dir/'checkpoint_sweep_n100'/ck.stem,formal_seeds=range(a.start_seed,a.start_seed+a.count),device=a.device);s=rep['summary'];rows.append({'checkpoint':'best.pt','step':80000,'success':s['success']['count'],'grasp':s['grasp']['count'],'lift':s['lift']['count'],'transport':s['transport']['count'],'place':s['place']['count'],'release':s['release']['count'],'retreat':s['retreat']['count'],'illegal_drop':s['illegal_drop']['count'],'ik_failure':s['ik_failure']['count'],'timeout':s['timeout']['count'],'mean_episode_length':s['episode_length']['mean'],'average_return':s['average_return']});
 with (a.training_dir/'checkpoint_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
