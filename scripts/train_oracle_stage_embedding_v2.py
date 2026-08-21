#!/usr/bin/env python3
from pathlib import Path
import argparse
from mujoco_shared_control.rss2023.oracle_stage_embedding_train import train_v2
ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'
OUTPUT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-dir',type=Path,default=DATASET);p.add_argument('--output-dir',type=Path,default=OUTPUT);p.add_argument('--device',default='auto');p.add_argument('--smoke-steps',type=int,default=500);a=p.parse_args();print(train_v2(a.dataset_dir,a.output_dir,device_name=a.device,smoke_steps=a.smoke_steps))
if __name__=='__main__':main()
