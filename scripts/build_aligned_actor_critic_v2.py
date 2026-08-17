#!/usr/bin/env python3
"""Build aligned Actor-Critic v2 and run V2 Expert-vs-Actor audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from build_aligned_actor_critic_v1 import HORIZONS, MANIFEST, PHASES, _branch, _heldout_audit_states
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic


ACTOR = Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
CRITIC = Path("outputs/sac_critic/sac_critic_pretrain_v2_20260814T010000Z/critic_pretrained_v2_best.pt")
V1_ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v1_20260813T231500Z")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return None
    value = spearmanr(x, y).statistic if method == "spearman" else pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = [("overall", rows)] + [(phase, [r for r in rows if r["phase"] == phase])
                                         for phase in PHASES]
    dimensions += [("P4a_PLACE", [r for r in rows if r["p4_substage"] == "P4a_PLACE"]),
                   ("P4b_RELEASE_STABILIZE",
                    [r for r in rows if r["p4_substage"] == "P4b_RELEASE_STABILIZE"])]
    result = {}
    for name, selected in dimensions:
        dq = np.asarray([row["delta_q"] for row in selected])
        entry: dict[str, Any] = {"samples": len(selected)}
        for horizon in HORIZONS:
            dg = np.asarray([row[f"delta_g_h{horizon}"] for row in selected])
            non_tie = np.abs(dg) > 1e-10; expert = dg > 1e-10; actor = dg < -1e-10
            entry[f"h{horizon}"] = {
                "spearman": correlation(dq, dg, "spearman"),
                "pearson": correlation(dq, dg, "pearson"),
                "sign_agreement_non_tie": (float(np.mean(np.sign(dq[non_tie]) == np.sign(dg[non_tie])))
                                             if non_tie.any() else None),
                "expert_truly_better_count": int(expert.sum()),
                "actor_truly_better_count": int(actor.sum()),
                "tie_count": int((~non_tie).sum()),
                "expert_better_identification_accuracy": (float(np.mean(dq[expert] > 0))
                                                           if expert.any() else None),
            }
        result[name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(); run = Path("outputs/sac_aligned") / f"aligned_actor_critic_v2_{args.run_id}"
    run.mkdir(parents=True, exist_ok=False)
    actor_source = torch.load(ACTOR, map_location="cpu", weights_only=False)
    critic_source = torch.load(CRITIC, map_location="cpu", weights_only=False)
    if critic_source["actor_sha256"] != sha(ACTOR): raise RuntimeError("Actor reference mismatch")
    actor = SACConstrainedGaussianActor(); actor.load_state_dict(actor_source["actor_state_dict"])
    actor.eval(); actor.requires_grad_(False)
    critics = TwinSACCritic(); critics.load_state_dict(critic_source["critic_state_dict"])
    critics.eval(); critics.requires_grad_(False)
    targets = TwinSACCritic(); targets.load_state_dict(critics.state_dict()); targets.requires_grad_(False)
    mean = torch.as_tensor(actor_source["observation_mean"])
    std = torch.as_tensor(actor_source["observation_std"])
    if not torch.equal(mean, torch.as_tensor(critic_source["observation_mean"])):
        raise RuntimeError("observation mean mismatch")
    if not torch.equal(std, torch.as_tensor(critic_source["observation_std"])):
        raise RuntimeError("observation std mismatch")
    states, reconstruction = _heldout_audit_states(
        MANIFEST, per_phase=25, reward_version="sac_reward_v2_candidate"
    )
    rows = []
    for index, state in enumerate(states):
        x = (torch.from_numpy(state.policy_state) - mean) / std
        with torch.no_grad():
            actor_action = actor.deterministic_action(x.unsqueeze(0)).squeeze(0)
            expert_action = torch.from_numpy(state.recorded_action).float()
            q1e, q2e = critics(x.unsqueeze(0), expert_action.unsqueeze(0))
            q1a, q2a = critics(x.unsqueeze(0), actor_action.unsqueeze(0))
            delta_q = float(torch.minimum(q1e, q2e) - torch.minimum(q1a, q2a))
        expert_branch = _branch(state, state.recorded_action, reward_version="sac_reward_v2_candidate")
        actor_branch = _branch(state, actor_action.numpy(), reward_version="sac_reward_v2_candidate")
        p4_substage = "NONE"
        if state.expert_stage == 5: p4_substage = "P4a_PLACE"
        elif state.expert_stage == 6: p4_substage = "P4b_RELEASE_STABILIZE"
        elif state.expert_stage == 7: p4_substage = "EXCLUDED_RETREAT"
        row = {"phase": state.phase, "p4_substage": p4_substage,
               "expert_stage": state.expert_stage, "episode_id": state.episode_id,
               "seed": state.seed, "step": state.step, "delta_q": delta_q,
               "expert_action": state.recorded_action.tolist(), "actor_action": actor_action.tolist(),
               "action_l2_difference": float(np.linalg.norm(state.recorded_action - actor_action.numpy())),
               "expert_terminal_reason": expert_branch["termination_reason"],
               "actor_terminal_reason": actor_branch["termination_reason"]}
        for horizon in HORIZONS:
            row[f"delta_g_h{horizon}"] = (expert_branch["returns"][str(horizon)]
                                            - actor_branch["returns"][str(horizon)])
        rows.append(row)
        if (index + 1) % 20 == 0: print(f"counterfactual={index+1}/100", flush=True)
    audit = {"protocol": {"reward_version": "sac_reward_v2_candidate", "gamma": .995,
                           "heldout_seeds": [100900, 100999], "states_per_phase": 25,
                           "continuation": "frozen RulePickPlaceExpert feedback"},
             "reconstruction": reconstruction, "summary": summarize(rows), "rows": rows}
    manifest = json.loads(MANIFEST.read_text())
    payload = {"format_version": "aligned_actor_critic_v2",
               "actor_state_dict": actor.state_dict(), "critic_state_dict": critics.state_dict(),
               "target_critic_state_dict": targets.state_dict(), "observation_mean": mean,
               "observation_std": std, "observation_spec": "policy_state_42 float32",
               "action_spec": actor_source["action_spec"],
               "action_semantics": "native constrained B3 x B3 x [-1,1] policy action",
               "reward_version": "sac_reward_v2_candidate", "gamma": .995,
               "manifest_content_sha": manifest["content_sha256"],
               "actor_source": str(ACTOR.resolve()), "actor_source_sha256": sha(ACTOR),
               "critic_source": str(CRITIC.resolve()), "critic_source_sha256": sha(CRITIC),
               "optimizer_state": None, "replay": None, "online_state": None}
    destination = run / "aligned_actor_critic_v2.pt"; torch.save(payload, destination)
    v1 = json.loads((V1_ALIGNED / "expert_vs_actor_value_audit.json").read_text())["summary"]
    v2 = audit["summary"]
    comparison = {}
    for key in ("overall", "P1", "P2", "P3", "P4"):
        comparison[key] = {"v1_spearman": v1[key]["h20"]["spearman"],
                           "v2_spearman": v2[key]["h20"]["spearman"],
                           "difference": v2[key]["h20"]["spearman"] - v1[key]["h20"]["spearman"],
                           "v1_sign_agreement": v1[key]["h20"]["sign_agreement_non_tie"],
                           "v2_sign_agreement": v2[key]["h20"]["sign_agreement_non_tie"]}
    (run / "actor_source.json").write_text(json.dumps({
        "path": str(ACTOR.resolve()), "sha256": sha(ACTOR), "closed_loop_success": 35,
        "updated": False}, indent=2) + "\n")
    (run / "critic_source.json").write_text(json.dumps({
        "path": str(CRITIC.resolve()), "sha256": sha(CRITIC),
        "reward_version": "sac_reward_v2_candidate"}, indent=2) + "\n")
    (run / "expert_vs_actor_value_audit_v2.json").write_text(json.dumps(audit, indent=2) + "\n")
    (run / "phase_audit_v2.json").write_text(json.dumps({k: v2[k] for k in ("overall", *PHASES)}, indent=2) + "\n")
    (run / "p4_substage_audit_v2.json").write_text(json.dumps({
        "P4a_PLACE": v2["P4a_PLACE"], "P4b_RELEASE_STABILIZE": v2["P4b_RELEASE_STABILIZE"],
        "retreat_samples": sum(row["p4_substage"] == "EXCLUDED_RETREAT" for row in rows)}, indent=2) + "\n")
    (run / "v1_vs_v2_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    summary = {"run": str(run.resolve()), "actor_sha_matches_v1": sha(ACTOR) == json.loads(
        (V1_ALIGNED / "actor_source.json").read_text())["sha256"],
        "target_exact_copy": all(torch.equal(value, targets.state_dict()[key])
                                 for key, value in critics.state_dict().items()),
        "comparison": comparison, "p4_substages": {
            "P4a_PLACE": v2["P4a_PLACE"], "P4b_RELEASE_STABILIZE": v2["P4b_RELEASE_STABILIZE"]}}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
