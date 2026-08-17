# SAC Critic Pretraining v1 Experiment

## Scope

This is a Critic-only offline experiment. The constrained Actor reference is frozen, no alpha/Actor optimizer exists, no online transition is collected, and no training Replay is modified.

- Formal run: `formal_rule_v1_20260812T050822Z`
- Reward: production-regressed `sac_reward_v1`
- Reward regression maximum difference: `1.3114509478384662e-15`
- SAC semantic transitions: 157,302
- Gamma: 0.995
- Actor reference: `outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt`
- Main objective: twin Monte-Carlo return regression
- Architecture: two independent `49→256→256→256→1` SiLU critics

## Episode split

The split is fixed by environment seed and never by transition:

| Split | Nominal seeds | Perturbed seeds | Episodes | Transitions |
|---|---|---|---:|---:|
| Train | 100000–100799 | 200000–200239 | 1,040 | 125,942 |
| Validation | 100800–100899 | 200240–200269 | 130 | 15,855 |
| Test | 100900–100999 | 200270–200299 | 130 | 15,505 |

Train categories and transitions:

- Nominal success: 800 episodes / 102,184 transitions
- Normal recovered: 65 / 9,398
- Delayed recovery under SAC semantics: 52 / 5,539
- Failure: 123 / 8,821

Phase counts are P1 40,340, P2 11,955, P3 39,293, P4 34,347, and seven legacy terminal-boundary rows. Uniform transition sampling was used; no reweighting was introduced.

`actions/normalized` is the attempted policy-level normalized Cartesian action. It remains the attempted action on all three IK-fallback rows; the deployed safe joint fallback is not substituted. No retained row required adapter projection.

## Return targets

Episode-start G0 statistics in the training split:

| Category | Mean | Median | Std | Min | Max | P05 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal success | 14.253441 | 14.248269 | 0.252889 | 13.423336 | 14.925665 | 13.835710 | 14.673402 |
| Normal recovered | 13.321902 | 13.340433 | 0.468817 | 12.068110 | 14.335965 | 12.456457 | 13.949974 |
| Delayed recovery | 2.544901 | 2.387659 | 0.672255 | 1.383632 | 3.585441 | 1.484810 | 3.533294 |
| Failure | -0.051142 | -0.095489 | 0.905260 | -1.782573 | 3.460130 | -1.690508 | 1.382009 |

The target scale separates successful behavior from delayed recovery/failure. Delayed recovery is truncated at its SAC-v1 illegal-drop or explicit-failure terminal; no SETTLING recovery or later success bonus is retained.

## Training

Two controlled models were trained:

1. Success-only: 800 nominal training episodes.
2. Mixed: the same 800 nominal episodes plus all 240 perturbed training episodes.

Configuration: Adam, learning rate `3e-4`, batch 512, max 100 epochs, gradient clipping 1.0, validation MC-loss selection, early-stopping patience 12. Success-only selected epoch 93; mixed selected epoch 5 and stopped after 17 epochs.

## Held-out return prediction

| Model | Test MAE | Test RMSE | Pearson | Spearman | Q1/Q2 disagreement MAE |
|---|---:|---:|---:|---:|---:|
| Success-only | 2.576582 | 7.343764 | -0.145699 | 0.557312 | 0.253291 |
| Mixed | 2.122162 | 4.035522 | 0.664198 | 0.601387 | 0.330298 |

The success-only model fits held-out nominal transitions extremely well (MAE `0.042934`, Spearman `0.994774`) but extrapolates catastrophically to failure (MAE `19.066364`, Spearman `-0.168016`). Mixed data materially improves overall scale and failure MAE (`9.961854`) but is still not an accurate value predictor on failure states.

Mixed test Q values remain finite: Q1 mean/std `11.982991/3.258789`, range `[-5.971666, 37.036739]`; Q2 mean/std `12.015467/3.311124`, range `[-6.586067, 28.369345]`. Its deterministic-reference one-step Bellman residual has MAE `0.656007` and RMSE `1.558551`; this is diagnostic only and was never used as a training target.

## Strict local counterfactual ranking

The exact same MuJoCo snapshots and actual H-step returns are used for old and pretrained Critics. Formal branches use the row's recorded Rule Expert attempted action as `a_star`; online branches use the stable deterministic Actor. Candidate groups separately perturb translation, rotation, and gripper at radii `1e-4`, `3e-4`, and `1e-3`, plus 16 random admissible candidates per group. There are 4,560 branches.

### Overall comparison

| Critic | H20 Spearman | H20 false improvement | H20 sign agreement | Expert preference accuracy |
|---|---:|---:|---:|---:|
| Old online Critic | 0.027306 | 50.02% | N/A | N/A |
| Success-only MC | -0.084782 | 54.44% | 43.97% | 44.88% |
| Mixed MC | 0.085774 | 44.36% | 53.42% | 58.06% |

Mixed supervision improves the old Critic modestly, especially at H5 (Spearman `0.350135`, false improvement `27.48%`), but H20 ranking remains weak and far below a reliable policy-improvement signal.

### Mixed Critic H20 by phase

| Phase | Spearman | False improvement | Sign agreement | Expert preference accuracy |
|---|---:|---:|---:|---:|
| P1 Pre-Grasp | 0.470533 | 29.05% | 73.77% | 73.81% |
| P2 Grasp | -0.154535 | 47.12% | 37.63% | 47.53% |
| P3 Transport | -0.091244 | 51.53% | 49.04% | 53.04% |
| P4 Place & Retreat | 0.118342 | 48.26% | 53.25% | 57.56% |

P1 learns meaningful local discrimination. P2 remains the least reliable ranking phase; P3 is also anti-correlated, and P4 remains near random. Therefore the mixed Critic cannot yet be used safely for unconstrained Actor improvement.

## Interpretation

The formal trajectories provide strong labels for `Q(s,a_behavior)` and broad outcome contrast, but they contain almost no same-state local action alternatives with return labels. An unrestricted neural Critic can fit MC return on the behavior manifold while its off-manifold slope `dQ/da` remains unconstrained. Success-only makes this failure worse; perturbed trajectories help but do not supply sufficiently dense same-state local contrasts, particularly at grasp and transport boundaries.

This is not a numerical-training failure and should not be addressed by continuing epochs, changing alpha, or starting Actor optimization. The next justified data step is a bounded, separately specified dataset of exact expert-state local action perturbations with MuJoCo counterfactual return labels. That dataset was not generated in this experiment.

Final status:

`SAC_CRITIC_NEEDS_LOCAL_COUNTERFACTUAL_DATA`

