# Learned Expert Joint Training v1

This formal run combines the aligned Native Constrained BC Actor and Mixed
Critic v2 with the frozen Reward v2 task. Rule Expert demonstrations initialize
both behavior and value. During training, Critic batches contain exactly 128
Reward-v2 Expert-train transitions and 128 deterministic online transitions.
The Actor maximizes Twin-Q only on online states and receives a nominal-success
Expert BC anchor. There is no entropy, alpha, Gaussian exploration, phase
augmentation, or reward/action/state change.

The planned protocol was 200k environment transitions with 256 deterministic
transitions of replay seeding, UTD 1, Actor/Critic learning rates `3e-4`,
`gamma=.995`, and `tau=.005`. Initial gradient contributions were balanced once:
`g_Q=25.8921`, `g_BC=.00283254`, hence fixed `lambda_BC=9140.967`.

The formal run triggered a protective structural stop around 165k. Success fell
from 50/100 at step 0 to 0/100 at every completed checkpoint. Q mean rose from
single digits through thousands and billions; Critic loss reached approximately
`1.15e19`, then a joint metric became non-finite. Accordingly, the 200k
checkpoint was not produced and no secondary/final evaluation was allowed.

This failure is not a reward/action-semantics regression. It demonstrates that
the specified deterministic off-policy bootstrapped Q maximization is unstable
even with a gradient-balanced nominal BC anchor and a persistent 50% Expert
Critic batch. The valid selected policy remains the unmodified step-0 Actor.
