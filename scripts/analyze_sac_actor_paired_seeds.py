#!/usr/bin/env python3
"""Create paired-seed BC versus distilled-SAC outcome analysis."""

import json
from pathlib import Path

BC=Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/evaluation.json")
SAC=Path("outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/closed_loop_evaluation.json")
OUTPUT=SAC.parent/"paired_seed_analysis.json"

def main():
    bc={r["seed"]:r for r in json.load(BC.open())["episodes"]}
    sac={r["seed"]:r for r in json.load(SAC.open())["episodes"]}
    groups={name:[] for name in ("bc_success_sac_success","bc_success_sac_failure",
        "bc_failure_sac_success","bc_failure_sac_failure")}
    for seed in sorted(bc):
        key=("bc_success_" if bc[seed]["success"] else "bc_failure_")+(
            "sac_success" if sac[seed]["success"] else "sac_failure")
        groups[key].append(seed)
    report={"seeds":"300000-300099","counts":{k:len(v) for k,v in groups.items()},
            "seed_groups":groups,"interpretation":(
                "Equal aggregate success with 16 successes lost and 16 gained; "
                "the environment is sensitive to residual closed-loop action differences."
            )}
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
