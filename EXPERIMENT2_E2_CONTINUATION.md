# Experiment 2 E2 Continuation Note

> **INVALIDATED 2026-08-21:** `scripts/evaluate_e2_stage_policy_v1.py`
> and its existing `experiment2_e2_shared_autonomy` output are retained only
> for audit and must not be reported.  The invalid output is marked by
> `INVALID_PROTOCOL_DO_NOT_REPORT.json`.
>
> The replacement entry point is:
>
> ```bash
> cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
> ./.venv/bin/python scripts/evaluate_e2_formal_v2.py
> ```
>
> It uses frozen Stage-aware Offline AWAC step 2500 as the sole formal user
> surrogate, independently selects TCN-V2, independently calibrates gamma,
> prints a provenance gate, and only then evaluates a new final holdout.

Status: the evaluator is implemented and is **resumable after interruption**.
Do not report final metrics until `completion.json` exists.  The frozen
protocol below must be preserved.

## Frozen protocol

- Calibration: 25 `NORMAL` + 25 `PLACE_RECOVERY`, disjoint from final cases
  by case id, environment seed, snapshot path/id, and source episode.
- Final: paired 50 `NORMAL` + 50 `PLACE_RECOVERY` for `NoAssist`, `Global`,
  `Oracle-V2`, and `TCN-V2`.
- Gamma grid: 0.0 through 1.0 in 0.1 steps, selected independently per model.
- Global consumes physical43 only.  Oracle-V2 receives GT stage one-hot.
  TCN-V2 receives only causal-TCN-predicted stage; GT stage is audit-only.
- CUDA is required only for model rollouts.  CPU rollout/fallback is forbidden.

## Completed work

1. `scripts/evaluate_e2_stage_policy_v1.py` has checkpoint and frozen-final
   manifest preflight for Global 110k, Oracle-V2 90k, and TCN-V2 120k.
2. The script has a preliminary `prepare_calibration()` implementation that
   calls `validate_recovery_stage_checkpoints.make_cases(..., 25,
   seed_base=8_500_000)` and filters Normal/Place cases.
3. That preliminary path audits environment-seed overlap with the final set.
   It must be strengthened to audit case id, snapshot path/id, and source
   episode before it is treated as valid.
4. `--check-only` remains CPU-safe and verifies checkpoint paths/payloads.

## Resume behavior

- Each completed rollout writes its trace first and then appends a JSONL event.
  On the next invocation, the evaluator validates and reuses the event by its
  exact trace path.  It therefore never reruns completed rollout traces.
- A malformed final JSONL line from an interrupted write is ignored and only
  that rollout is recomputed deterministically.
- To reduce load after a desktop freeze, run bounded batches with
  `--max-new-rollouts 25`.  It exits with `PAUSED_CLEANLY`, writes
  `resume_status.json`, and the exact same command continues from the next
  unfinished rollout.  Omitting this flag runs through to completion.

## Continue from these repository interfaces

- E1 shared-control semantics: `scripts/run_experiment1_gamma_sweep.py`,
  especially `run_episode`, `OfflineAWACSurrogatePilot`,
  `GlobalSharedController`, and `GlobalActionPostprocessor` imported from
  `evaluate_experiment1_global_effectiveness`.
- Recovery snapshot reset and termination: `scripts/validate_recovery_stage_checkpoints.py`,
  especially `make_cases`, `evaluate_case`, `Predictor`, and
  `build_e2_valid_failure_snapshot_bank.restore`.
- Global inference: `validate_recovery_global_checkpoints.GlobalPredictor`.
- Oracle-V2 inference: `validate_recovery_stage_checkpoints.Predictor('V2', ...)`.
- Causal-TCN architecture: `src/mujoco_shared_control/stage/tcn.py` (`StageTCNV1`);
  checkpoint/cache contract: `outputs/recovery_stage_dp_training/causal_tcn_recovery_v1_20260820/`.
  TCN-V2 is a stage-conditioned diffusion policy trained in
  `scripts/train_recovery_tcn_v2.py`.

## Current modified files

- `mujoco_shared_control/scripts/evaluate_e2_stage_policy_v1.py`
  - preflight and incomplete calibration-manifest preparation only;
  - currently contains `E2_ROLLOUT_BACKEND_NOT_YET_IMPLEMENTED` and must not
    be presented as executable.
- `mujoco_shared_control/EXPERIMENT2_E2_CONTINUATION.md` (this note).

## Required final entry point

For a full uninterrupted run, use:

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
./.venv/bin/python scripts/evaluate_e2_stage_policy_v1.py
```

For a guarded continuation after a freeze, use:

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
./.venv/bin/python scripts/evaluate_e2_stage_policy_v1.py --max-new-rollouts 25
```

It executes calibration preparation/audit, smoke, gamma sweep, gamma
selection, final E2, statistics, and figures without CPU rollout.
