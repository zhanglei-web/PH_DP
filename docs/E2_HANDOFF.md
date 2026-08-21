# E2 Handoff — Failure-Induced Stage Regression

Status date: 2026-08-18. All E2 execution is paused. Do not start new rollouts from this handoff alone.

## Frozen components

- `RuleBasedRecoveryPilot`: `src/mujoco_shared_control/experts/recovery_pilot.py`
  - SHA-256: `30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58`
  - Must not change thresholds, hysteresis, debounce, gains, gripper semantics, or phase logic.
- Global Diffusion V2: `outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/best.pt`
  - Input is only `state43 = policy_state_42 + object_grasped`.
  - Canonical gripper semantics: close `-0.25`, open `+1.0`.
- E1 transfer gamma: `0.675`, effective diffusion step `33` of 50. It was selected for Offline AWAC-7.5k normal-task evaluation, **not** for RuleBasedRecoveryPilot.

## E2-0 baseline

Output: `outputs/experiments/e2_recovery_pilot_baseline/run_20260818T010000Z/`

| Condition | N | NoAssist recovery | Notes |
|---|---:|---:|---|
| NORMAL | 100 | 100% task success | Baseline pilot is strong on normal tasks. |
| GRASP_FAILURE | 100 | 99% | Recovery pilot works. |
| TRANSPORT_DROP | 100 | 67% | Early 100%, Mid 87.9%, Late 12.1%. |
| PLACE_FAILURE | 100 | 91% end-to-end | 91% physical injection realization. |

Late transport drops were excluded before any Global-assisted E2 evaluation because near-goal drops can be semantically ambiguous.

## E2-0b snapshot bank

Output: `outputs/experiments/e2_valid_failure_snapshot_bank/run_20260818T024000Z/`

- Parent manifest SHA-256: `86cc9ed97144a3b7cb4cd79cf3539778ff5a9774efcc27f40c6ae923289d859e`
- 300 snapshots: 100 Grasp, 50 Transport Early, 50 Transport Mid, 100 Place.
- Full MuJoCo integration state, control/mocap/userdata, pilot state, adapter state, reward state, and observation state are serialized.
- Replay determinism audit: PASS (20 snapshots, restored twice, first 20 steps identical).
- NoAssist qualification showed Transport Mid `success_without_regrasp=50/50`; Mid is not valid for the formal main benchmark.

## E2-0c Failure Snapshot Bank V2

Output: `outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z/`

- V2 manifest: `e2_failure_snapshot_bank_v2_manifest.json`
- SHA-256: `d06c1b95b821ab797ef83506c4bfec952d861313e4cab68d19017878a545496f`
- Composition: 100 Grasp, 100 Transport **Early** (25% progress), 100 Place.
- Transport Mid/Late are excluded before any Global E2 evaluation.
- V2 NoAssist results:

| Failure | Recovery | Regrasp | Drop | IK | Timeout |
|---|---:|---:|---:|---:|---:|
| Grasp | 100/100 | 100/100 | 0 | 0 | 0 |
| Transport Early | 89/100 | 92/100 | 3 | 0 | 0 |
| Place | 94/100 | 100/100 | 6 | 0 | 0 |
| **Pooled** | **283/300 (94.33%)** | **292/300 (97.33%)** | — | — | — |

- V2 remains `E2_FAILURE_BANK_V2_NOT_READY`, solely because Transport Early `success_without_regrasp=8/100` exceeds the pre-registered `<=5%` gate. Do not change this status or remove those eight snapshots.
- Audit: PASS; NaN/Inf=0; no Global, gamma, TCN, or AWAC used during bank construction.

## Existing Global gamma=.675 transfer diagnostic

Partial E2-1 formal trajectories: `outputs/experiments/e2_global_failure_recovery/run_20260818T044000Z/`

The 300 unique V2 snapshot IDs have Global `.675` trajectories distributed over `chunks/` and `trajectories/`. Duplicate chunk ranges exist from interrupted bounded execution; analyses must deduplicate by `snapshot_id` and never count a snapshot twice.

Deduplicated diagnostic result:

| Failure | NoAssist recovery | Global .675 recovery | NoAssist regrasp | Global .675 regrasp |
|---|---:|---:|---:|---:|
| Grasp | 100/100 | 0/100 | 100/100 | 21/100 |
| Transport Early | 89/100 | 20/100 | 92/100 | 51/100 |
| Place | 94/100 | 1/100 | 100/100 | 50/100 |
| **Pooled** | **283/300** | **21/300 (7.00%)** | **292/300** | **122/300 (40.67%)** |

Observed Global `.675` failure modes: high IK termination, timeout, and unexpected drops. Pre-run NoAssist snapshot consistency audit passed. These findings are a **transfer diagnostic only**, not the final E2 comparison, because `.675` was not characterized for RuleBasedRecoveryPilot normal behavior.

## Required next work (not started)

1. Implement `RuleBasedRecoveryPilot + Global` NORMAL gamma evaluator by reusing the E1 Global pipeline:
   - strict outer gamma-zero bypass;
   - 43D observation only;
   - RSS2023 `assist`, trained normalizers, canonical postprocess, `ExpertCommandAdapter`, and frozen reward/termination semantics.
2. Run a fixed NORMAL validation sweep on 50 new shared seeds for gamma:
   `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9, 1.0`.
3. Before the sweep, prove gamma-zero identity on at least 10 seeds: raw action, executed action, state trajectory, and termination must be exact.
4. Select `gamma_RP` using **only** NORMAL validation:
   - success >=95%, IK/drop/timeout <=5%, finite outputs, non-zero correction;
   - among candidates within 2 pp of best success, choose largest gamma.
5. Run 100 new paired normal confirmation seeds for NoAssist vs `gamma_RP`; Global success must be >=90% or stop.
6. Only after confirmation, perform formal failure comparison on the frozen 300 V2 snapshots:
   - if `gamma_RP==.675`, reuse and deduplicate existing `.675` trajectories, then generate missing statistics;
   - otherwise run exactly one new Global branch per frozen snapshot; never rerun NoAssist, replace snapshots, or remove the eight ambiguous Transport entries.

## Required formal statistics

- Failure-specific and pooled recovery/regrasp paired bootstrap CIs (10,000 paired resamples), exact McNemar tests, and Holm correction across failure types.
- Post-snapshot metrics only: recovery, regrasp, transport/place/retreat, unexpected drop, IK, timeout, and latency.
- Mechanism windows: `PRE_REGRASP` and `FULL_RECOVERY`; translation cosine, conflict rates, physical correction magnitudes, gripper disagreement, adapter diagnostics.
- Global never receives milestones, active stage, failure label, TCN output, or failure type.

## Do-not-do list

- Do not modify RecoveryPilot, snapshot V2, gamma grid, reward, termination rules, gripper mapping, or Global checkpoint.
- Do not use existing `.675` failure outcomes to select `gamma_RP`.
- Do not run Stage-conditioned/TCN-conditioned control, E3, AWAC training, or Global training.
