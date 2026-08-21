#!/usr/bin/env python3
"""The single CUDA-only, paired Experiment-2 evaluator.

This file deliberately uses the existing recovery validation backend: its
snapshot reset, RuleBasedRecoveryPilot, adapter, reward, and termination
contracts are the experiment contract.  It does not create another simulator
semantics for E2.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, pickle, time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

import validate_recovery_stage_checkpoints as protocol
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.stage.tcn import StageTCNV1
from train_recovery_causal_tcn_v1 import feature as tcn_feature

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/final_stage_ambiguity_experiments_20260820"
FINAL_MANIFEST = BASE / "growth_analysis/growth_normal_place_manifest.json"
OUT = BASE / "experiment2_e2_shared_autonomy"
CALIBRATION_DIR = OUT / "calibration_cases_25x25"
CALIBRATION_MANIFEST = CALIBRATION_DIR / "calibration_manifest.json"
GAMMAS = [round(value / 10, 1) for value in range(11)]
METHODS = ("NoAssist", "Global", "Oracle-V2", "TCN-V2")
MODELS = {
    "Global": (ROOT / "outputs/recovery_stage_dp_training/recovery_global_120k_20260820", 110000),
    "Oracle-V2": (ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820", 90000),
    "TCN-V2": (ROOT / "outputs/recovery_stage_dp_training/recovery_tcn_v2_120k_20260820", 120000),
}
TCN_DIR = ROOT / "outputs/recovery_stage_dp_training/causal_tcn_recovery_v1_20260820"


class RolloutBudgetReached(RuntimeError):
    """A clean, resumable stop used to keep a desktop evaluation bounded."""


class ResumableRunner:
    """Recover completed rollouts by their *exact* trace path.

    The event log is append-only and can contain repeated case/method/gamma
    entries (for example, a calibration case also used by smoke).  A trace
    path is therefore the only safe resume identity.  We only reuse an entry
    when its trace still exists and parses as a completed JSON trace.
    """
    def __init__(self, output: Path, max_new_rollouts: int | None) -> None:
        self.output = output
        self.max_new_rollouts = max_new_rollouts
        self.new_rollouts = 0
        self.cached: dict[str, dict[str, Any]] = {}
        events = output / "rollout_events.jsonl"
        if not events.exists():
            return
        with events.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    trace = Path(row["trace_path"])
                    # A trace is written before its event.  Requiring both
                    # guards against a partially flushed event after a crash.
                    if not trace.is_file() or not isinstance(json.loads(trace.read_text()), list):
                        continue
                    key = str(trace.resolve())
                    previous = self.cached.get(key)
                    if previous is not None and previous != row:
                        raise RuntimeError(f"conflicting completed rollout events for {key}")
                    self.cached[key] = row
                except (json.JSONDecodeError, KeyError, OSError, TypeError):
                    # The final line may have been torn while the machine was
                    # interrupted.  It is not a completed rollout and will be
                    # recomputed under the same deterministic seed.
                    print({"resume_ignored_event_line": line_number}, flush=True)

    def run(self, case: dict[str, Any], method: str, gamma: float,
            controller: "E2Controller | None", tcn: "CausalTCN | None", trace_path: Path) -> dict[str, Any]:
        key = str(trace_path.resolve())
        cached = self.cached.get(key)
        if cached is not None:
            expected = (str(case["case_id"]), method, float(gamma))
            actual = (str(cached.get("case_id")), cached.get("method"), float(cached.get("gamma")))
            if actual != expected:
                raise RuntimeError(f"resume event metadata mismatch for {key}: {actual} != {expected}")
            return cached
        if self.max_new_rollouts is not None and self.new_rollouts >= self.max_new_rollouts:
            raise RolloutBudgetReached
        result = rollout(case, method, gamma, controller, tcn, trace_path)
        self.cached[key] = result
        self.new_rollouts += 1
        # Prevent allocator fragmentation from accumulating over a multi-hour
        # run.  Models remain resident; only releasable temporary blocks go.
        if torch.cuda.is_available() and self.new_rollouts % 10 == 0:
            torch.cuda.empty_cache()
        print({"rollout_complete": self.new_rollouts, "resumed_cached": len(self.cached),
               "method": method, "gamma": gamma, "case": case["case_id"]}, flush=True)
        return result


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def case_identity(case: dict[str, Any]) -> dict[str, str]:
    """All identity dimensions requested by the frozen protocol."""
    return {
        "case_id": str(case["case_id"]),
        "environment_seed": str(case["environment_seed"]),
        # Empty means this Normal case has no snapshot.  It is intentionally
        # excluded from the snapshot-set intersection below; it is not a
        # shared identity such as a file hash.
        "snapshot": str(case.get("snapshot_sha256") or case.get("snapshot_hash") or case.get("snapshot_path") or ""),
        # Normal cases have no source episode/snapshot; their seed is the source.
        "source_episode": str(case.get("source_episode") or case.get("source_episode_id") or f"seed:{case['environment_seed']}"),
    }


def audit_disjoint(calibration: list[dict[str, Any]], final: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = [case_identity(c) for c in calibration], [case_identity(c) for c in final]
    overlaps = {
        key: sorted(
            ({c[key] for c in left} & {c[key] for c in right})
            if key != "snapshot" else
            ({c[key] for c in left if c[key]} & {c[key] for c in right if c[key]})
        )
        for key in left[0]
    }
    report = {"status": "PASS" if not any(overlaps.values()) else "FAIL", "overlaps": overlaps,
              "calibration_count": len(left), "final_count": len(right)}
    if report["status"] != "PASS":
        raise RuntimeError(f"calibration/final overlap audit failed: {overlaps}")
    return report


def preflight() -> dict[str, Any]:
    final = json.loads(FINAL_MANIFEST.read_text())["cases"]
    counts = {kind: sum(c["kind"] == kind for c in final) for kind in ("NORMAL", "PLACE_RECOVERY")}
    if counts != {"NORMAL": 50, "PLACE_RECOVERY": 50}:
        raise RuntimeError(f"final E2 manifest must be 50+50, got {counts}")
    report: dict[str, Any] = {"final_manifest": str(FINAL_MANIFEST.resolve()), "final_manifest_sha256": sha(FINAL_MANIFEST),
                              "final_case_counts": counts, "methods": list(METHODS), "models": {},
                              "protocol": {"calibration": {"NORMAL": 25, "PLACE_RECOVERY": 25}, "final": counts,
                                           "gamma_grid": GAMMAS, "paired_final_cases": True,
                                           "global_input": "physical43_only", "oracle_stage": "GT_onehot",
                                           "tcn_stage": "causal_TCN_hard_onehot; GT audit_only"}}
    for name, (directory, step) in MODELS.items():
        checkpoint, normalizer = directory / "checkpoints" / f"step_{step:06d}.pt", normalization_path(name, directory)
        if not checkpoint.is_file() or not normalizer.is_file(): raise FileNotFoundError(checkpoint if not checkpoint.is_file() else normalizer)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" not in payload or "diffusion_config" not in payload: raise ValueError(f"invalid {name} checkpoint")
        report["models"][name] = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "normalization": str(normalizer.resolve()), "step": step}
    for path in (TCN_DIR / "checkpoints/best_validation_macro_f1.pt", TCN_DIR / "normalization_stats.npz"):
        if not path.is_file(): raise FileNotFoundError(path)
    return report


def normalization_path(method: str, directory: Path) -> Path:
    """TCN-V2 intentionally reuses the frozen Oracle-V2 train normalizer."""
    direct = directory / "normalization_stats.npz"
    if direct.is_file(): return direct
    reference = directory / "normalization_reference.json"
    if method == "TCN-V2" and reference.is_file():
        path = Path(json.loads(reference.read_text())["ORACLE_V2_NORMALIZATION"])
        if path.is_file(): return path
    return direct


def prepare_calibration() -> dict[str, Any]:
    """Create/freeze real snapshot-backed calibration cases, never final cases."""
    if CALIBRATION_MANIFEST.exists():
        payload = json.loads(CALIBRATION_MANIFEST.read_text())
        audit_disjoint(payload["cases"], json.loads(FINAL_MANIFEST.read_text())["cases"])
        return payload
    # make_cases already uses the frozen recovery snapshot construction/reset path.
    generated = protocol.make_cases(CALIBRATION_DIR / "source_cases", 25, seed_base=8_500_000)
    cases = []
    for item in generated:
        if item["kind"] not in ("NORMAL", "PLACE_RECOVERY"): continue
        case = dict(item); case["case_id"] = "E2CAL_" + case["case_id"]
        case["source_episode"] = f"calibration_seed:{case['environment_seed']}"
        cases.append(case)
    counts = Counter(c["kind"] for c in cases)
    if counts != Counter({"NORMAL": 25, "PLACE_RECOVERY": 25}): raise RuntimeError(f"bad calibration composition: {counts}")
    final = json.loads(FINAL_MANIFEST.read_text())["cases"]
    overlap = audit_disjoint(cases, final)
    payload = {"version": "e2-calibration-normal-place-v2", "frozen": True, "counts": dict(counts),
               "case_generation": "existing validate_recovery_stage_checkpoints.make_cases snapshot backend", "overlap_audit": overlap, "cases": cases}
    write_json(CALIBRATION_MANIFEST, payload)
    return payload


class E2Controller:
    """RSS2023 adapter. Its public input prevents Global stage leakage."""
    def __init__(self, method: str, device: torch.device) -> None:
        self.method, self.device = method, device
        directory, step = MODELS[method]; payload = torch.load(directory / "checkpoints" / f"step_{step:06d}.pt", map_location=device, weights_only=False)
        config = payload["diffusion_config"]
        if method == "Global":
            cfg = DiffusionConfig(**config)
            if cfg.observation_dim != 43: raise ValueError("Global must be physical43 only")
            self.model = RSS2023Diffusion(cfg)
        else:
            cfg = StageEmbeddingDiffusionConfig(**{k: config[k] for k in StageEmbeddingDiffusionConfig.__dataclass_fields__ if k in config})
            self.model = StageEmbeddingDiffusion(cfg)
        self.model.load_state_dict(payload["model"]); self.model.to(device).eval(); self.model.requires_grad_(False)
        with np.load(normalization_path(method, directory), allow_pickle=False) as n:
            self.pm, self.ps = n["physical_mean"].astype("f4"), n["physical_std"].astype("f4"); self.am, self.astd = n["action_mean"].astype("f4"), n["action_std"].astype("f4")
        self.generator: torch.Generator | None = None

    def reset(self, seed: int) -> None: self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def assist(self, physical43: np.ndarray, raw7: np.ndarray, gamma: float, *, stage_onehot: np.ndarray | None = None) -> np.ndarray:
        if gamma == 0.0: return np.asarray(raw7, dtype="f4").copy()
        if self.generator is None: raise RuntimeError("controller reset required")
        physical = (np.asarray(physical43, "f4") - self.pm) / self.ps
        human = (np.asarray(raw7, "f4") - self.am) / self.astd
        if self.method == "Global":
            # No stage parameter exists in this path by construction.
            observation = torch.as_tensor(physical, device=self.device)[None]
        else:
            if stage_onehot is None or np.asarray(stage_onehot).shape != (5,): raise ValueError("stage policy requires one-hot stage")
            observation = torch.as_tensor(np.r_[physical, stage_onehot].astype("f4"), device=self.device)[None]
        action = self.model.assist(observation, torch.as_tensor(human, device=self.device)[None], float(gamma), generator=self.generator)[0]
        result = action.cpu().numpy() * self.astd + self.am
        if not np.isfinite(result).all(): raise FloatingPointError("non-finite RSS2023.assist output")
        return result.astype("f4")


class CausalTCN:
    def __init__(self, device: torch.device) -> None:
        payload = torch.load(TCN_DIR / "checkpoints/best_validation_macro_f1.pt", map_location=device, weights_only=False)
        self.model = StageTCNV1().to(device).eval(); self.model.load_state_dict(payload["model"]); self.model.requires_grad_(False); self.device = device
        with np.load(TCN_DIR / "normalization_stats.npz", allow_pickle=False) as n: self.mean, self.std = n["mean"].astype("f4"), n["std"].astype("f4")
        if self.mean.shape != (19,) or self.std.shape != (19,): raise ValueError("invalid TCN normalizer")

    def initial(self, state: np.ndarray) -> deque[np.ndarray]:
        first = (tcn_feature(state, np.zeros(7, "f4")) - self.mean) / self.std
        return deque([first.astype("f4")] * 20, maxlen=20)

    @torch.inference_mode()
    def predict(self, history: deque[np.ndarray]) -> tuple[int, np.ndarray]:
        posterior = self.model.posterior(torch.as_tensor(np.asarray(history, "f4"), device=self.device)[None])[0].cpu().numpy()
        return int(posterior.argmax()), posterior.astype("f4")


def reset_case(case: dict[str, Any], env: PickPlaceEnv, adapter: ExpertCommandAdapter, pilot: RuleBasedRecoveryPilot) -> tuple[dict[str, Any], int, AWACRewardV1Online]:
    options = {"randomize_arm": True, "arm_joint_noise_scale": 1.0, "randomize_object": True, "randomize_goal": True}
    initial, _ = env.reset(seed=int(case["environment_seed"]), options=options)
    adapter.reset(initial["ee_pose"], initial["q_obs"]); pilot.reset(float(initial["object_pose"][2, 3]), int(case["environment_seed"]) + 17)
    reward = AWACRewardV1Online(protocol.state43(env, initial))
    if case["kind"] == "NORMAL": return initial, 0, reward
    snapshot = pickle.loads(Path(case["snapshot_path"]).read_bytes())
    # ``restore`` mutates ``reward`` from the serialized snapshot and returns
    # only its observation plus the IK counter.
    observation, consecutive = protocol.bank.restore(env, adapter, pilot, reward, snapshot)
    return observation, consecutive, reward


def rollout(case: dict[str, Any], method: str, gamma: float, controller: E2Controller | None, tcn: CausalTCN | None, trace_path: Path) -> dict[str, Any]:
    env = PickPlaceEnv(render_mode=None, control_timestep=protocol.DT, max_episode_steps=protocol.MAX, enable_camera=False)
    pilot = RuleBasedRecoveryPilot(); adapter = ExpertCommandAdapter(env.ik_controller, pilot.action_spec); spec = ExpertActionSpec(); post = GlobalActionPostprocessor.from_expert_spec(spec)
    try:
        obs, consecutive, reward = reset_case(case, env, adapter, pilot)
        if controller: controller.reset(int(case["sampling_seed"]) + int(round(gamma * 1000)))
        history = tcn.initial(protocol.state43(env, obs)) if tcn else None
        previous_cmd = previous_executed = None; rows: list[dict[str, Any]] = []; reason = "timeout"; milestones = np.zeros(5, bool)
        for step in range(protocol.MAX):
            state = protocol.state43(env, obs); command, gt_stage = pilot.predict(_expert_observation(case["case_id"], 0, step, obs, state[:42], previous_cmd, previous_executed)); gt_stage = int(gt_stage)
            raw = spec.normalize(command.delta_pose_gripper).astype("f4")
            pred_stage, posterior = (None, None)
            stage_onehot = None
            if method == "Oracle-V2": stage_onehot = np.eye(5, dtype="f4")[gt_stage]
            elif method == "TCN-V2":
                assert tcn is not None and history is not None
                pred_stage, posterior = tcn.predict(history); stage_onehot = np.eye(5, dtype="f4")[pred_stage]
            started = time.perf_counter()
            assisted = raw.copy() if method == "NoAssist" else controller.assist(state, raw, gamma, stage_onehot=stage_onehot)  # type: ignore[union-attr]
            inference_ms = (time.perf_counter() - started) * 1000 if method != "NoAssist" else 0.0
            bounded = np.clip(assisted, -1, 1); canonical = post(bounded); adapted = adapter.adapt(spec.denormalize(canonical))
            next_obs, *_ = env.step(adapted.joint_target); next_state = protocol.state43(env, next_obs)
            consecutive = 0 if adapted.accepted else consecutive + 1
            outcome = reward.step(state, next_state, ik_failure=consecutive >= protocol.IKMAX, time_limit=step + 1 >= protocol.MAX); milestones = reward.tracker.current.copy()
            rows.append({"step": step, "gt_stage_audit_only": gt_stage, "predicted_stage": pred_stage, "posterior": None if posterior is None else posterior.tolist(), "raw": raw.tolist(), "assisted": assisted.tolist(), "executed": np.asarray(adapted.normalized).tolist(), "adapter_accepted": bool(adapted.accepted), "fallback_used": bool(adapted.fallback_used), "action_clipped": bool(adapted.action_clipped or not np.array_equal(assisted, bounded)), "inference_ms": inference_ms, "termination": outcome.termination_reason})
            if history is not None: history.append(((tcn_feature(next_state, np.asarray(adapted.normalized, "f4")) - tcn.mean) / tcn.std).astype("f4"))
            previous_cmd, previous_executed, obs = command.delta_pose_gripper.copy(), np.asarray(adapted.normalized, "f4"), next_obs
            if outcome.terminated or outcome.truncated: reason = outcome.termination_reason; break
        trace_path.parent.mkdir(parents=True, exist_ok=True); trace_path.write_text(json.dumps(rows) + "\n")
        result = {"case_id": case["case_id"], "kind": case["kind"], "method": method, "gamma": gamma, "success": reason == "task_success", "illegal_drop": reason == "illegal_drop", "ik_failure": reason == "ik_failure_limit", "timeout": reason == "timeout", "termination": reason, "steps": step + 1, "grasp": bool(milestones[0]), "lift": bool(milestones[1]), "transport": bool(milestones[2]), "place": bool(milestones[3]), "retreat": bool(milestones[4]), "tcn_gt_agreement": None if method != "TCN-V2" else float(np.mean([r["gt_stage_audit_only"] == r["predicted_stage"] for r in rows])), "trace_path": str(trace_path.resolve())}
        # Append before returning, so a CUDA interruption still leaves every
        # completed rollout auditable without relying on the final CSV.
        with (OUT / "rollout_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n"); stream.flush()
        return result
    finally: env.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"N": len(rows), **{key: float(np.mean([bool(r[key]) for r in rows])) for key in ("success", "illegal_drop", "ik_failure", "timeout", "grasp", "lift", "transport", "place", "retreat")}, "mean_steps": float(np.mean([r["steps"] for r in rows]))}


def select_gamma(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    candidates = [summarize([r for r in rows if r["gamma"] == gamma]) | {"gamma": gamma} for gamma in GAMMAS]
    # Pre-registered deterministic safety then highest success, largest gamma tie break.
    eligible = [r for r in candidates if r["illegal_drop"] <= .05 and r["ik_failure"] <= .05 and r["timeout"] <= .05]
    if not eligible: eligible = candidates
    chosen = max(eligible, key=lambda r: (r["success"], r["gamma"]))
    return {"method": method, "selection_rule": "safety <=5% each; maximize success; largest gamma tie-break", "candidates": candidates, "selected_gamma": chosen["gamma"], "selected": chosen}


def paired_statistics(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {m: {r["case_id"]: r for r in final_rows if r["method"] == m} for m in METHODS}; base = methods["NoAssist"]; report: dict[str, Any] = {}
    for method in METHODS[1:]:
        report[method] = {}
        for group in ("ALL", "NORMAL", "PLACE_RECOVERY"):
            ids = sorted(case_id for case_id in (base.keys() & methods[method].keys()) if group == "ALL" or base[case_id]["kind"] == group)
            group_report: dict[str, Any] = {"N": len(ids), "metrics": {}}
            for number, metric in enumerate(("success", "illegal_drop", "ik_failure", "timeout")):
                a = np.asarray([base[i][metric] for i in ids], bool); b = np.asarray([methods[method][i][metric] for i in ids], bool); delta = b.astype(float) - a.astype(float); rng = np.random.default_rng(20260820 + number)
                samples = np.asarray([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(10000)])
                plus, minus = int((~a & b).sum()), int((a & ~b).sum()); n = plus + minus; p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(plus, minus) + 1)) / 2**n)
                group_report["metrics"][metric] = {"difference_method_minus_NoAssist": float(delta.mean()), "paired_bootstrap_95_ci": [float(np.quantile(samples,.025)), float(np.quantile(samples,.975))], "mcnemar_exact": {"method_only": plus, "noassist_only": minus, "p_value": p}}
            report[method][group] = group_report
    return report


def figures(output: Path, selections: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    """Portable B1--B6 figures; no optional matplotlib dependency."""
    plot = output / "figures"; plot.mkdir(parents=True, exist_ok=True)
    for index in range(1, 7):
        image = Image.new("RGB", (960, 540), "white"); draw = ImageDraw.Draw(image); draw.text((35, 28), f"E2 Figure B{index}", fill="black")
        if index <= 3:
            method = METHODS[index]; points = selections[method]["candidates"]; draw.line((80,450,900,450), fill="black"); draw.line((80,80,80,450), fill="black")
            for a, b in zip(points, points[1:]): draw.line((80+a["gamma"]*820,450-a["success"]*330,80+b["gamma"]*820,450-b["success"]*330), fill="#2166ac", width=3)
            draw.text((80,470), "gamma (success rate)", fill="black")
        else:
            values = [(m, summarize([r for r in final_rows if r["method"] == m])["success"]) for m in METHODS]
            for j,(name,value) in enumerate(values):
                x=120+j*190; draw.rectangle((x,450-value*320,x+110,450), fill="#4d9221"); draw.text((x,465),name,fill="black")
        image.save(plot / f"B{index}.png")


def write_resume_status(runner: ResumableRunner, phase: str, status: str, message: str | None = None) -> None:
    write_json(OUT / "resume_status.json", {"status": status, "phase": phase,
        "new_rollouts_this_invocation": runner.new_rollouts,
        "cached_completed_trace_count": len(runner.cached), "max_new_rollouts": runner.max_new_rollouts,
        "message": message, "resume_command": "./.venv/bin/python scripts/evaluate_e2_stage_policy_v1.py"})


def main() -> None:
    global OUT, CALIBRATION_DIR, CALIBRATION_MANIFEST
    parser = argparse.ArgumentParser(description="Single-command CUDA-only E2 evaluator")
    parser.add_argument("--check-only", action="store_true", help="static artifact/path validation only; never roll out")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-new-rollouts", type=int, default=None,
                        help="stop cleanly after N new rollouts; rerun unchanged to resume")
    args = parser.parse_args()
    OUT = args.output.resolve(); CALIBRATION_DIR = OUT / "calibration_cases_25x25"; CALIBRATION_MANIFEST = CALIBRATION_DIR / "calibration_manifest.json"
    report = preflight()
    if args.check_only:
        report["static_validation"] = "PASS"; report["cuda_rollout_not_run"] = True; print(json.dumps(report, indent=2)); return
    if args.max_new_rollouts is not None and args.max_new_rollouts <= 0: raise ValueError("--max-new-rollouts must be positive")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED_BUT_UNAVAILABLE")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True); runner = ResumableRunner(OUT, args.max_new_rollouts); phase = "setup"
    try:
        calibration = prepare_calibration(); final = json.loads(FINAL_MANIFEST.read_text())["cases"]; audit = audit_disjoint(calibration["cases"], final); write_json(OUT / "overlap_audit.json", audit)
        controllers = {m: E2Controller(m, device) for m in METHODS[1:]}; tcn = CausalTCN(device)
        phase = "smoke"; smoke = []
        for method in METHODS[1:]:
            for case in (next(c for c in calibration["cases"] if c["kind"] == "NORMAL"), next(c for c in calibration["cases"] if c["kind"] == "PLACE_RECOVERY")):
                smoke.append(runner.run(case, method, .1, controllers[method], tcn if method == "TCN-V2" else None, OUT / "smoke" / method / f"{case['case_id']}.json"))
        write_json(OUT / "smoke_report.json", {"status": "PASS", "rows": smoke})
        sweeps: dict[str, list[dict[str, Any]]] = {}; selections: dict[str, Any] = {}
        for method in METHODS[1:]:
            phase = f"calibration:{method}"; rows=[]
            for gamma in GAMMAS:
                for case in calibration["cases"]:
                    rows.append(runner.run(case, method, gamma, controllers[method], tcn if method == "TCN-V2" else None, OUT / "calibration" / method / f"gamma_{gamma:.1f}" / f"{case['case_id']}.json"))
            sweeps[method] = rows; write_csv(OUT / "calibration" / method / "rollouts.csv", rows); selections[method] = select_gamma(rows, method)
        write_json(OUT / "gamma_selection_frozen.json", selections)
        phase = "final"; final_rows=[]
        for case in final:
            final_rows.append(runner.run(case, "NoAssist", 0., None, None, OUT / "final" / "NoAssist" / f"{case['case_id']}.json"))
            for method in METHODS[1:]: final_rows.append(runner.run(case, method, selections[method]["selected_gamma"], controllers[method], tcn if method == "TCN-V2" else None, OUT / "final" / method / f"{case['case_id']}.json"))
        write_csv(OUT / "final_rollouts.csv", final_rows); stats = paired_statistics(final_rows); write_json(OUT / "paired_statistics.json", stats); figures(OUT, selections, final_rows)
        write_json(OUT / "completion.json", {"E2_EVALUATOR_COMPLETE": "YES", "final_summary": {m: summarize([r for r in final_rows if r["method"] == m]) for m in METHODS}, "paired_statistics": stats})
        write_resume_status(runner, "complete", "COMPLETE")
        print(json.dumps({"E2_EVALUATOR_COMPLETE": "YES", "output": str(OUT)}, indent=2))
    except RolloutBudgetReached:
        write_resume_status(runner, phase, "PAUSED_CLEANLY", "rollout budget reached; rerun the same command to continue")
        print(json.dumps({"E2_EVALUATOR_COMPLETE": "NO", "status": "PAUSED_CLEANLY", "output": str(OUT), "new_rollouts": runner.new_rollouts}, indent=2))


if __name__ == "__main__": main()
