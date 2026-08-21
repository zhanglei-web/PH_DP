#!/usr/bin/env python3
"""Bounded formal E2-1 Global rollout chunk; no training or selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import evaluate_e2_global_failure_recovery as e
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('start',type=int);p.add_argument('stop',type=int);a=p.parse_args();torch.set_num_threads(1)
 man=json.loads((e.V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text());snaps=man['snapshots'][a.start:a.stop]
 if not snaps:raise SystemExit('empty chunk')
 (a.root/'trajectories').mkdir(parents=True,exist_ok=True);(a.root/'chunks').mkdir(exist_ok=True)
 c=e.GlobalSharedController(e.CKPT);post=GlobalActionPostprocessor.from_expert_spec();env,ctx=e.ctx();rows=[]
 for i,m in enumerate(snaps,a.start):
  r,_=e.run_global(m,c,post,a.root/'trajectories'/f"{m['snapshot_id']}.npz",context=ctx);rows.append(r)
 env.close();out=a.root/'chunks'/f'{a.start:03d}_{a.stop:03d}.json';out.write_text(json.dumps(rows,indent=2)+'\n');print(out)
if __name__=='__main__':main()
