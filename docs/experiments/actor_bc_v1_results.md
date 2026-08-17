# Actor BC v1 Experiment Record

This document is the canonical paper-facing record of the Actor BC v1 experiment. It records measured configuration and results for future tables and comparisons; it is not a general runtime log.

## Experiment Identification

| Item | Value |
|---|---|
| Experiment | Actor BC v1 |
| Run ID | `actor_bc_v1_20260812T170000Z` |
| Rule Expert frozen commit | `d5ce43ff70af25491c545ec513d56e9f988c4f6b` |
| Formal Expert Run | `formal_rule_v1_20260812T050822Z` |
| Manifest | `manifests/rule_expert_v1_formal.json` |
| Manifest content SHA | `5a3f3cfb76f37a95ceed5155c83158529f76cef6f82ea652b9a66ed050aa7c6b` |
| Split seed | `20260812` |
| Training seed | `20260812` |
| Evaluation seeds | `300000–300099` |
| Training device | CPU |
| Code HEAD at training time | `d5ce43ff70af25491c545ec513d56e9f988c4f6b` |
| Actor BC implementation | Local working tree, not committed |
| Workspace status | Dirty by design: local manifest/loader and Actor BC implementation changes are uncommitted; generated outputs are Git-ignored |

## Table 1. Actor BC training dataset

| Split | Episodes | Transitions | Data category | Percentage of Actor BC episodes |
|---|---:|---:|---|---:|
| Train | 900 | 115021 | Nominal Success | 90.0% |
| Validation | 100 | 12817 | Nominal Success | 10.0% |
| Total Actor BC dataset | 1000 | 127838 | Nominal Success | 100.0% |

Excluded formal-data categories:

| Category | Actor BC use |
|---|---|
| Normal recovered | Not used |
| Delayed recovery | Not used |
| Failure | Not used |

> Actor BC intentionally uses only nominal successful expert demonstrations. Perturbed recovered and failed trajectories are reserved for critic/value learning.

## Table 2. Actor BC v1 model and training configuration

| Item | Value |
|---|---|
| Observation | `policy_state_42` |
| Observation dimension | 42 |
| Action | 7D normalized `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` |
| Action dimension | 7 |
| Hidden layers | 256 / 256 / 256 |
| Activation | SiLU |
| Output activation | None |
| Parameters | 144391 |
| Loss | Ordinary MSE |
| Optimizer | AdamW |
| Initial learning rate | `3e-4` |
| Batch size | 256 |
| Weight decay | `1e-4` |
| Maximum epochs | 100 |
| Completed epochs | 100 |
| Best epoch | 98 |
| Early stopping triggered | No |
| Gradient clipping | Global norm 1.0 |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 4 epochs |
| Minimum learning rate | `1e-6` |
| Early stopping patience | 12 epochs |
| Observation normalization | Train-only mean/std; epsilon `1e-6`; constant dimensions retain scale 1 |
| Action normalization | Frozen `ExpertActionSpec` |
| Translation normalization scale | 0.025 m |
| Rotation normalization scale | 0.10 rad |
| Gripper physical range | `[0.0, 0.08]` m |
| Training determinism | Python, NumPy, PyTorch, and DataLoader seeded with `20260812`; PyTorch deterministic algorithms enabled |

## Table 3. Offline imitation performance

Metrics are from `checkpoint_best.pt`, epoch 98. Values below retain the precision stored in the checkpoint; the shorter values in parentheses are paper-friendly renderings.

| Metric | Train | Validation |
|---|---:|---:|
| Total MSE | `9.288616274716333e-05` (`9.28862e-5`) | `8.530670311301947e-05` (`8.53067e-5`) |
| XYZ MSE | `0.00021224821102805436` (`2.12248e-4`) | `0.00019421541946940124` (`1.94215e-4`) |
| Rotation MSE | `2.691139968646894e-07` (`2.69114e-7`) | `2.8667832907558477e-07` (`2.86678e-7`) |
| Gripper MSE | `1.2650704775296617e-05` (`1.26507e-5`) | `1.3640557881444693e-05` (`1.36406e-5`) |
| Total MAE | `0.002151499968022108` (`0.0021515`) | `0.0021197437308728695` (`0.0021197`) |
| Gripper classification accuracy | `1.0` (100%) | `1.0` (100%) |
| XYZ physical error | `0.00023557049280498177` m (`0.2356 mm`) | `0.0002215081622125581` m (`0.2215 mm`) |

### Validation per-dimension MAE

| Dimension | Raw MAE | Paper-friendly value |
|---|---:|---:|
| Δx | `0.0019009318202733994` | `0.00190093` |
| Δy | `0.0016901755006983876` | `0.00169018` |
| Δz | `0.007424009032547474` | `0.00742401` |
| Δrx | `0.00041986172436736524` | `0.00041986` |
| Δry | `0.00041928267455659807` | `0.00041928` |
| Δrz | `0.0003683705872390419` | `0.00036837` |
| gripper | `0.002615575445815921` | `0.00261558` |

Gripper classification uses normalized value `0.375` as the class threshold, the midpoint between the closed target `-0.25` and open target `+1.0`.

## Table 4. Actor output sanity check

The best checkpoint was evaluated on 1000 transitions sampled from the fixed validation split with sample seed `20260812`.

| Metric | Result |
|---|---:|
| Validation samples checked | 1000 |
| NaN / Inf elements | 0 |
| Illegal actions after clipping | 0 |
| Samples exceeding `[-1,1]` before clipping | 397 / 1000 |
| Pre-clip exceedance rate | `0.397` (39.7%) |
| Pre-clip element exceedance rate | `0.05671428571428572` (5.67%) |
| Rotation absolute prediction mean | `0.0004076698678545654` (`0.00040767`) |
| Rotation absolute prediction max | `0.004872010089457035` (`0.00487201`) |
| Gripper normalized min | `-0.266801118850708` (`-0.2668`) |
| Gripper normalized max | `1.0203219652175903` (`1.0203`) |
| Sample total MSE | `7.993209146661684e-05` |

> Pre-clipping exceedance is dominated by small output overshoot around the open-gripper target at normalized value +1.0.

## Table 5. Closed-loop Actor BC performance on unseen seeds

Evaluation protocol:

```text
Seeds:                  300000–300099
Episodes:               100
Rule Expert assistance: None
Checkpoint:             checkpoint_best.pt (epoch 98)
```

| Metric | Count | Rate |
|---|---:|---:|
| Task success | 22 | `0.22` (22%) |
| Grasp milestone | 82 | `0.82` (82%) |
| Lift milestone | 54 | `0.54` (54%) |
| Transport milestone | 54 | `0.54` (54%) |
| Release milestone | 52 | `0.52` (52%) |
| Retreat milestone | 22 | `0.22` (22%) |

### Termination reasons

| Termination reason | Episodes | Rate |
|---|---:|---:|
| `task_success` | 22 | 22% |
| `ik_failure_limit` | 70 | 70% |
| `time_limit` | 8 | 8% |

### Closed-loop control indicators

| Metric | Events/steps | Episodes affected |
|---|---:|---:|
| IK fallback | 455 events | 70 / 100 (70%) |
| Cartesian adapter clipping | 1750 events | 80 / 100 (80%) |
| Network pre-clip exceedance | 8927 steps | 100 / 100 (100%) |

### Episode length

| Subset | Mean | Min | Max |
|---|---:|---:|---:|
| All episodes | `237.02` | 96 | 500 |
| Successful episodes | `122.81818181818181` (`122.82`) | 106 | 202 |
| Failed episodes | `269.2307692307692` (`269.23`) | 96 | 500 |

## Table 6. Milestone progression during closed-loop evaluation

| Milestone | Episodes reached | Reach rate | Mean first-reaching step | Min | Max |
|---|---:|---:|---:|---:|---:|
| Grasped | 82 | 82% | `64.61` | 35 | 339 |
| Lifted | 54 | 54% | `58.22` | 46 | 191 |
| Transported | 54 | 54% | `84.37` | 66 | 217 |
| Released | 52 | 52% | `109.31` | 87 | 287 |
| Retreated | 22 | 22% | `121.82` | 105 | 201 |

The mean first-reaching step is averaged independently over the episodes that reached each milestone. Consequently, aggregate means across different episode subsets are not themselves a temporal ordering claim.

## Milestone retention

| Transition | Retained episodes | Retention |
|---|---:|---:|
| Start → Grasp | 82 / 100 | `0.8200` (82.00%) |
| Grasp → Lift | 54 / 82 | `0.6585365853658537` (65.85%) |
| Lift → Transport | 54 / 54 | `1.0` (100.00%) |
| Transport → Release | 52 / 54 | `0.9629629629629629` (96.30%) |
| Release → Retreat | 22 / 52 | `0.4230769230769231` (42.31%) |

## Table 7. Failure distribution by last achieved milestone

Total failed episodes: 78.

| Last milestone | Episodes | Percentage of failures |
|---|---:|---:|
| None | 18 | `0.23076923076923078` (23.08%) |
| Grasped | 28 | `0.358974358974359` (35.90%) |
| Transported | 2 | `0.02564102564102564` (2.56%) |
| Released | 30 | `0.38461538461538464` (38.46%) |
| Total | 78 | 100.00% |

No failed episode had `lifted` as its last achieved milestone; the count for that category is 0.

## Failure reason × last milestone

| Failure reason | None | Grasped | Transported | Released | Total |
|---|---:|---:|---:|---:|---:|
| IK failure limit | 15 | 24 | 2 | 29 | 70 |
| Time limit | 3 | 4 | 0 | 1 | 8 |
| Total | 18 | 28 | 2 | 30 | 78 |

## Table 8. Closed-loop behavioral diagnostics

| Metric | Result |
|---|---:|
| Drop events | 79 |
| Episodes with at least one drop | 38 / 100 (38%) |
| Wrong gripper switch events | 1312 |
| Episodes affected by wrong gripper switches | 77 / 100 (77%) |
| IK fallback events | 455 |
| IK fallback episodes | 70 / 100 (70%) |
| Cartesian adapter clipping events | 1750 |
| Adapter clipping episodes | 80 / 100 (80%) |

Wrong gripper switch definition:

> Any additional gripper class transition beyond the expected first open→closed transition followed by the subsequent closed→open transition.

Collision count: **N/A**. No additional collision metric was measured in Actor BC v1.

## Key empirical observations

### Observation 1: one-step error versus closed-loop outcome

```text
Best offline validation MSE: 8.530670311301947e-05 (8.53067e-5)
Closed-loop task success:    22 / 100 (22%)
```

Very low one-step imitation error did not translate to high closed-loop task success.

### Observation 2: grasp-to-lift retention

```text
Grasp milestone:        82 / 100 (82%)
Lift milestone:         54 / 100 (54%)
Grasp → Lift retention: 54 / 82 = 65.85%
```

This is the first clear performance-loss interval in the measured milestone sequence.

### Observation 3: release-to-retreat retention

```text
Release milestone:         52 / 100 (52%)
Retreat milestone:         22 / 100 (22%)
Release → Retreat retention: 22 / 52 = 42.31%
```

Release-to-retreat is the largest measured late-stage performance bottleneck.

### Observation 4: accumulated closed-loop errors

```text
Episodes with IK fallback:             70%
Episodes with extra gripper switching: 77%
Episodes with Cartesian clipping:      80%
```

These results are consistent with closed-loop distribution shift and/or accumulated action error. This experiment does not separately establish their causal contributions.

### Observation 5: observation hypothesis

`policy_state_42` does not contain explicit `object_grasped` or contact state. This may contribute to ambiguity around grasp/release boundaries, but the present BC experiment does not establish causality. This remains a hypothesis for a future controlled ablation.

## Future Main Comparison Table

| Method | Success | Grasp | Lift | Transport | Release | Retreat | Drop Episodes | IK Failure Episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC v1 | 22% | 82% | 54% | 54% | 52% | 22% | 38% | 70% |
| BC v2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAC Expert | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Global DP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Phase-DP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared Ctrl | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Future Ablation Table

| Observation / Policy Variant | Success | Grasp | Lift | Release | Retreat |
|---|---:|---:|---:|---:|---:|
| `policy_state_42` BC | 22% | 82% | 54% | 52% | 22% |
| + grasp/contact state | TBD | TBD | TBD | TBD | TBD |
| + phase condition | TBD | TBD | TBD | TBD | TBD |
| + history | TBD | TBD | TBD | TBD | TBD |

These rows reserve comparison structure only; no unmeasured ablation result is implied.

## Training artifact inventory

Run directory:

```text
outputs/actor_bc/actor_bc_v1_20260812T170000Z/
```

| Artifact | Purpose |
|---|---|
| `checkpoint_best.pt` | Best validation-MSE checkpoint, epoch 98 |
| `checkpoint_last.pt` | Final checkpoint, epoch 100 |
| `training_config.json` | Fixed configuration, data counts, determinism, and code-state metadata |
| `normalization.json` | Train-only 42D observation statistics and frozen action normalization |
| `metrics.csv` | Per-epoch train and validation metrics |
| `evaluation.json` | Offline sanity results, aggregate closed-loop results, and 100 per-seed records |

## Reproducibility references

```text
Formal Expert Run:       formal_rule_v1_20260812T050822Z
Actor BC Run:            actor_bc_v1_20260812T170000Z
Rule Expert commit:      d5ce43ff70af25491c545ec513d56e9f988c4f6b
Manifest content SHA:    5a3f3cfb76f37a95ceed5155c83158529f76cef6f82ea652b9a66ed050aa7c6b
Manifest split seed:     20260812
Actor training seed:     20260812
Actor evaluation seeds:  300000–300099
Actor BC implementation: local working tree, not committed
```

Primary measured result sources:

```text
outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt
outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_last.pt
outputs/actor_bc/actor_bc_v1_20260812T170000Z/training_config.json
outputs/actor_bc/actor_bc_v1_20260812T170000Z/normalization.json
outputs/actor_bc/actor_bc_v1_20260812T170000Z/metrics.csv
outputs/actor_bc/actor_bc_v1_20260812T170000Z/evaluation.json
```
