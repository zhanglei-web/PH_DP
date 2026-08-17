# SAC Critic-Only Warmup Validation

The bounded experiment is recorded at:

```text
outputs/sac_training/sac_v1_critic_warmup_sanity_20260813T010000Z/
```

The tested schedule was:

```text
COLLECT:         steps 1-10000
CRITIC_WARMUP:  steps 10001-20000
FULL_SAC:       steps 20001 onward
```

The warmup implementation is semantically correct: Actor and alpha were exactly
unchanged, target Critics received Polyak updates, and deterministic success stayed
8/20 at both 10k and 20k. However, the bootstrapped Critic scale continued drifting
during warmup. Once Actor/alpha updates began, success collapsed to 0/20 by step
25k and Q values subsequently exploded, causing the run to stop before 30k.

See `sanity_report.md` in the run directory for the exact metrics and old-schedule
comparison.

```text
SAC_CRITIC_WARMUP_INSUFFICIENT
```
