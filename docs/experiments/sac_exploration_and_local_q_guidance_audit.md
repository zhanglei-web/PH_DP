# SAC Exploration Scale and Local Q-Guidance Audit

## Scope and provenance

- Audit checkpoint: `outputs/sac_training/sac_v2_final_safe_trust_20260813T100000Z/checkpoints/latest.pt`
- Checkpoint step: 30,000
- Checkpoint SHA-256: `43ba44675f7f195245828e59526eb9f5e46e21ac82cec3f4b8736e2df52daf81`
- Development seeds: 420000–420099
- Reward: `sac_reward_v1`; discount: 0.995
- Networks, optimizers, target networks, alpha, and Replay were not updated.
- Final test seeds 500000–500099 were not used.

The 24% deterministic result below is specific to development seeds 420000–420099. It does not replace the previously measured 41% result on the separate 100-seed validation pool.

## Exploration scale

All stochastic settings use common random numbers for every paired seed. Only the Gaussian standard deviation changes; the deterministic mean path is fixed.

| Policy | Std | Success | Grasp | Lift | Transport | Release | Retreat | Mean return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 0 | 24/100 | 87% | 70% | 70% | 53% | 24% | 6.168179 |
| log_std=-5.0 | 0.006738 | 8/100 | 79% | 42% | 38% | 16% | 8% | 0.500416 |
| log_std=-4.5 | 0.011109 | 4/100 | 80% | 40% | 37% | 16% | 4% | -0.410102 |
| log_std=-4.0 | 0.018316 | 0/100 | 81% | 38% | 35% | 3% | 0% | -3.601009 |
| log_std=-3.5 | 0.030197 | 0/100 | 67% | 22% | 18% | 1% | 0% | -4.121652 |
| log_std=-3.0 | 0.049787 | 0/100 | 46% | 4% | 1% | 0% | 0% | -4.982395 |

Paired deterministic-success to stochastic-failure counts are 23/24 at -5 and 24/24 at every setting from -4.5 through -3. The stochastic successes at -5 and -4.5 are not simply retained deterministic successes: only 1/24 deterministic success is retained at -5 and none at -4.5; several previously failing seeds change outcome.

At log_std=-3 the mean perturbations measured on 4,096 fixed Replay states (four samples per state) are 1.78675 mm for translation, 0.00787984 rad for rotation, and 0.545682 mm for gripper. Even at log_std=-5 they are 0.242003 mm, 0.00106837 rad, and 0.0738312 mm. The full-success collapse begins between -4.5 and -4.0, and substantial degradation is already present at -5.

Termination at -3 is dominated by 84 IK-failure terminals, 12 time limits, and 4 illegal drops. Phase progression collapses primarily before or immediately after grasp: only 4% lift and 1% transport are reached.

**Exploration diagnosis: `EXPLORATION_SCALE_TOO_LARGE`.** The current -3 scale is well inside the empirically unsafe regime. The evidence also shows that merely reducing it to -5 does not recover the deterministic behavior pipeline.

## State reconstruction validity

Counterfactual branching used complete MuJoCo integration-state snapshots, task-protocol state, adapter targets, and the IK-failure counter.

- Formal nominal-success reconstruction: 50 episodes, 6,295 SAC-semantic transitions, maximum policy-state error `5.90808e-9`.
- Online Replay reconstruction: all 30,000 transitions, maximum policy-state error exactly 0.

Formal trajectories were reset with their recorded arm/object/goal positions and replayed with their recorded normalized actions. Legacy rows after a SAC-v1 terminal were excluded. No counterfactual result was inferred from policy_state_42 alone.

## Static local Critic diagnostics

For each source and phase, 500 states were evaluated. Gradient candidates used normalized L2 action steps 1e-4, 3e-4, and 1e-3. Sixteen admissible random candidates of radius 1e-3 were also ranked by `min(Q1,Q2)`.

Mean / p95 / maximum `||dQ/da||`:

| Source | P1 | P2 | P3 | P4 |
|---|---:|---:|---:|---:|
| Formal nominal | 2.986 / 8.133 / 33.661 | 7.386 / 24.142 / 41.152 | 4.897 / 10.066 / 24.062 | 7.835 / 21.902 / 62.661 |
| Online Replay | 6.124 / 14.039 / 41.253 | 3.757 / 23.587 / 35.004 | 5.336 / 7.925 / 10.809 | 6.345 / 21.192 / 49.548 |

The largest gradient outliers occur in P4, followed by P2/P1 depending on source. `Q` assigns positive improvement to the best of 16 local candidates for every state, but physical counterfactuals do not validate this optimism reliably.

## MuJoCo local counterfactuals

For each source and each phase, 10 exactly restored states were branched. Each state used three Q-gradient candidates and sixteen random local candidates (1,520 candidate branches total). After the candidate first action, both branches used the same stable deterministic policy through horizon 20.

| Scope | ΔQ>0 but ΔG1<0 | ΔQ>0 but ΔG5<0 | ΔQ>0 but ΔG10<0 | ΔQ>0 but ΔG20<0 | Mean per-state Spearman at H20 |
|---|---:|---:|---:|---:|---:|
| Overall | 46.02% | 43.43% | 50.95% | 45.57% | 0.00460 |
| Formal nominal | 40.70% | 39.61% | 42.89% | 43.33% | 0.05399 |
| Online Replay | 51.61% | 47.47% | 59.45% | 47.93% | -0.04478 |
| P1 Pre-Grasp | 45.50% | 44.55% | 46.92% | 34.12% | 0.19444 |
| P2 Grasp | N/A (one-step rewards tied) | 20.44% | 43.11% | 47.11% | -0.07655 |
| P3 Transport | 75.65% | 50.87% | 53.91% | 49.57% | -0.00851 |
| P4 Place & Retreat | 62.22% | 57.78% | 59.56% | 50.67% | -0.09096 |

P4 has the poorest H10 ranking (`Spearman=-0.33237`) and the largest Q-gradient outlier. P3 is worst at immediate consequence: 75.65% of Q-predicted improvements have negative one-step actual progress. P2/P4 are behaviorally sensitive: the small local candidates create 29 additional illegal-drop terminals from P2 snapshots and 17 candidate unstable-release terminals from P4 snapshots over the 20-step branches. This is direct evidence that grasp/release boundaries are sensitive even at 1e-3 normalized action radius.

The H20 overall Spearman value near zero is not evidence of a useful ranking. It means the Critic is essentially uninformative for ordering these small action perturbations, with online states slightly anti-correlated.

**Q-guidance diagnosis: local Q guidance is not reliable enough for unconstrained policy improvement.** The failure is strongest around P4 and P3, with P2 also fragile over longer horizons.

## Combined conclusion

Both audited mechanisms are problematic:

1. Current stochastic exploration (`log_std=-3`) yields 0/100 success, and even -5 retains only 1 of 24 deterministic successes.
2. Critic-preferred local actions do not consistently improve actual short-horizon return; overall H20 rank correlation is approximately zero and the online subset is negative.

Reducing std alone would improve data quality, but it does not solve the demonstrated Critic-ranking problem. Conversely, trusting the current Q direction while merely protecting the mean policy would optimize against a locally unreliable signal. The next change should therefore jointly enforce safe/phase-aware exploration and gate mean-policy updates on validated local improvement; it should not simply release the existing trust region or continue the present actor objective unchanged.

Final diagnostic status:

`SAC_EXPLORATION_AND_Q_GUIDANCE_BOTH_PROBLEMATIC`

