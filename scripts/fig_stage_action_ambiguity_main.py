#!/usr/bin/env python3
"""Create the main Stage_Action_Ambiguity_Analysis_V2 paper figure.

All plotted points are sampled from NORMAL_SUCCESS train episodes and retain the
dataset's ground-truth ``active_phase`` labels.  The Transport->Place pairs are
reconstructed with the same deterministic sample/normalization protocol as C1/C2.
No recovery episode, predicted stage, or synthetic point is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED"
OUT = ROOT / "outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2"
EPSILON = 1.0
CAP = 3000
SEED = 20260820
BLUE, ORANGE, INK, MUTED, GRID, PALE = "#2878B5", "#D95F02", "#18212B", "#59636E", "#DCE2E8", "#F4F6F8"


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x.mean(0)
    _, _, vt = np.linalg.svd(x - center, full_matrices=False)
    return (x - center) @ vt[:2].T, center, vt[:2]


def transform(x: np.ndarray, center: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - center) @ components.T


def nearest_cross_episode(q, qe, r, re):
    ds, js = [], []
    for start in range(0, len(q), 32):
        d = cdist(q[start:start + 32], r)
        d[re[None, :] == qe[start:start + 32, None]] = np.inf
        j = d.argmin(1)
        ds.append(d[np.arange(len(j)), j]); js.append(j)
    return np.concatenate(ds), np.concatenate(js)


def font(size, bold=False):
    root = "/usr/share/fonts/truetype/dejavu/"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(root + name, size)


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def project(points, box, lim):
    x0, y0, x1, y1 = box
    lo, hi = lim
    span = np.maximum(hi - lo, 1e-6)
    x = x0 + (points[:, 0] - lo[0]) / span[0] * (x1 - x0)
    y = y1 - (points[:, 1] - lo[1]) / span[1] * (y1 - y0)
    return np.c_[x, y]


def limits(*arrays, pad=.08):
    x = np.vstack(arrays); lo = x.min(0); hi = x.max(0); d = np.maximum(hi - lo, .1)
    return lo - d * pad, hi + d * pad


def axes(draw, box, lim, xlabel="PC 1", ylabel="PC 2"):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=rgb(GRID), width=2)
    for f in (.25, .5, .75):
        x = int(x0 + (x1 - x0) * f); y = int(y0 + (y1 - y0) * f)
        draw.line((x, y0, x, y1), fill=rgb("#EEF1F4"), width=1)
        draw.line((x0, y, x1, y), fill=rgb("#EEF1F4"), width=1)
    draw.text((x0 + (x1 - x0) / 2 - 18, y1 + 8), xlabel, font=font(21), fill=rgb(MUTED))
    draw.text((x0 - 8, y0 - 30), ylabel, font=font(21), fill=rgb(MUTED))


def dots(draw, pts, color, radius=3, alpha=None):
    # RGB canvas: lightly colored small points preserve density without invented smoothing.
    for x, y in pts:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb(color))


def ellipse_from_points(draw, pts, outline, fill=None):
    if len(pts) < 3:
        return
    mean = pts.mean(0); cov = np.cov(pts.T) + np.eye(2) * 1e-6
    vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2 * np.pi, 120)
    circle = np.c_[np.cos(t), np.sin(t)]
    contour = mean + (circle * (2.0 * np.sqrt(vals))) @ vecs.T
    draw.line([tuple(p) for p in contour], fill=rgb(outline), width=4, joint="curve")


def panel_title(draw, x, title):
    draw.text((x, 35), title, font=font(30, True), fill=rgb(INK))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    stats = json.loads((OUT / "normalization_stats_used.json").read_text())
    mean = np.asarray(stats["physical_mean"], dtype=np.float32); std = np.asarray(stats["physical_std"], dtype=np.float32)
    am = np.asarray(stats["action_mean"], dtype=np.float32); ast = np.asarray(stats["action_std"], dtype=np.float32)
    # Keep the original C1/C2 sampling stream identical: it sampled stages 0..4
    # in this order before evaluating any pair.
    parts = {stage: [[], [], []] for stage in range(5)}
    normal_episodes = 0
    for eid in manifest["splits"]["train"]:
        path = Path(manifest["episode_paths"][eid])
        with h5py.File(path, "r") as f:
            if str(f.attrs["episode_type"]) != "NORMAL_SUCCESS":
                continue
            if "active_phase" not in f:
                raise RuntimeError(f"Missing active_phase in {eid}")
            normal_episodes += 1
            phase = f["active_phase"][:].astype(int)
            state = f["full_physical_state"][:].astype(np.float32)
            action = f["executed_action"][:].astype(np.float32)
            for stage in range(5):
                ix = np.flatnonzero(phase == stage)
                parts[stage][0].append(state[ix]); parts[stage][1].append(action[ix]); parts[stage][2].append(np.full(len(ix), eid))
    if normal_episodes != 800:
        raise RuntimeError(f"Expected 800 NORMAL_SUCCESS train episodes, found {normal_episodes}")
    rng = np.random.default_rng(SEED)
    sampled = {}
    for stage in range(5):
        s, a, ep = (np.concatenate(v) for v in parts[stage])
        ix = rng.choice(len(s), min(CAP, len(s)), replace=False)
        sampled[stage] = ((s[ix] - mean) / std, (a[ix] - am) / ast, ep[ix])
    s2, a2, e2 = sampled[2]; s3, a3, e3 = sampled[3]
    distances, match = nearest_cross_episode(s2, e2, s3, e3)
    close = np.flatnonzero(distances < EPSILON)
    if not len(close):
        raise RuntimeError("No real Transport->Place pairs satisfy the fixed epsilon")
    # Data panels use all real sampled stage points; close pairs identify the overlap neighborhood.
    obs2, oc, ov = pca2(np.vstack([s2, s3])); obs3 = transform(s3, oc, ov)
    action_pair = np.vstack([a2[close], a3[match[close]]])
    act_pair_2, ac, av = pca2(action_pair); act_pair_3 = transform(a3[match[close]], ac, av)
    act_pair_2 = act_pair_2[:len(close)]
    state_mean = float(distances[close].mean()); action_distances = np.linalg.norm(a2[close] - a3[match[close]], axis=1)
    action_mean = float(action_distances.mean())
    c3 = json.loads((OUT / "stage_condition_variance.json").read_text())
    reduction = float(c3["relative_reduction"]) * 100

    W, H = 3300, 1100
    im = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(im)
    # (a) Observation space
    px = 75; panel_title(draw, px, "(a) Overlapping observation regions between stages")
    box = (px + 45, 145, 1035, 855); olim = limits(obs2, obs3)
    axes(draw, box, olim, "Observation PC 1", "Observation PC 2")
    # deterministic thinning only affects display, never data selection.
    dots(draw, project(obs2[::2], box, olim), BLUE, 2); dots(draw, project(obs3[::2], box, olim), ORANGE, 2)
    overlap_xy = np.vstack([project(obs2[close], box, olim), project(obs3[match[close]], box, olim)])
    ellipse_from_points(draw, overlap_xy, INK)
    for pt in overlap_xy:
        draw.ellipse((pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5), outline=rgb(INK), width=2)
    draw.rounded_rectangle((px + 85, 885, 1005, 1020), radius=15, fill=rgb(PALE), outline=rgb(GRID), width=2)
    draw.text((px + 115, 912), "Ground-truth active_phase", font=font(23, True), fill=rgb(INK))
    draw.ellipse((px + 122, 958, px + 142, 978), fill=rgb(BLUE)); draw.text((px + 154, 953), "Stage 2: Transport", font=font(21), fill=rgb(INK))
    draw.ellipse((px + 402, 958, px + 422, 978), fill=rgb(ORANGE)); draw.text((px + 434, 953), "Stage 3: Place", font=font(21), fill=rgb(INK))
    draw.text((px + 745, 952), f"close pairs: D_O = {state_mean:.2f}", font=font(21, True), fill=rgb(INK))
    # (b) Divergent actions
    px = 1130; panel_title(draw, px, "(b) Similar observations induce divergent expert actions")
    box = (px + 45, 145, 2090, 855); alim = limits(act_pair_2, act_pair_3, pad=.18)
    axes(draw, box, alim, "Action PC 1", "Action PC 2")
    pa2, pa3 = project(act_pair_2, box, alim), project(act_pair_3, box, alim)
    for q, r in zip(pa2, pa3):
        draw.line((*q, *r), fill=rgb("#C9D1D9"), width=2)
    dots(draw, pa2, BLUE, 7); dots(draw, pa3, ORANGE, 7)
    ellipse_from_points(draw, pa2, BLUE); ellipse_from_points(draw, pa3, ORANGE)
    draw.rounded_rectangle((px + 65, 885, 2070, 1020), radius=15, fill=rgb(PALE), outline=rgb(GRID), width=2)
    draw.text((px + 97, 910), "Same cross-episode observation neighborhood", font=font(22, True), fill=rgb(INK))
    draw.text((px + 97, 957), f"Transport - Place:  D_O = {state_mean:.2f}     D_A = {action_mean:.2f}", font=font(27, True), fill=rgb(INK))
    draw.text((px + 97, 997), f"{len(close)} matched real pairs; lines connect paired demonstrations", font=font(19), fill=rgb(MUTED))
    # (c) Conditioning shows the same real local action data, before/after labels are observed.
    px = 2225; panel_title(draw, px, "(c) Stage conditioning disentangles action modes")
    left, right = (px + 20, 195, 2665, 790), (2780, 195, 3260, 790)
    axes(draw, left, alim, "Action PC 1", "")
    axes(draw, right, alim, "Action PC 1", "")
    cpa2, cpa3 = project(act_pair_2, left, alim), project(act_pair_3, left, alim)
    mixed = np.vstack([cpa2, cpa3])
    dots(draw, mixed, "#6F7A85", 6)
    ellipse_from_points(draw, mixed, "#4F5963")
    dots(draw, project(act_pair_2, right, alim), BLUE, 6); dots(draw, project(act_pair_3, right, alim), ORANGE, 6)
    ellipse_from_points(draw, project(act_pair_2, right, alim), BLUE); ellipse_from_points(draw, project(act_pair_3, right, alim), ORANGE)
    draw.text((2380, 815), "Global: p(A | O)", font=font(26, True), fill=rgb(INK))
    draw.text((2840, 815), "Stage-conditioned: p(A | O, Z)", font=font(26, True), fill=rgb(INK))
    draw.rounded_rectangle((px + 35, 890, 3250, 1015), radius=15, fill=rgb(PALE), outline=rgb(GRID), width=2)
    draw.text((px + 67, 915), "Local action covariance trace", font=font(20, True), fill=rgb(INK))
    draw.text((px + 67, 956), f"variance reduction = {reduction:.2f}%", font=font(29, True), fill=rgb(INK))
    draw.text((px + 67, 992), "conditioning exposes stage-specific action modes", font=font(19), fill=rgb(MUTED))
    out_png = OUT / "FIG_STAGE_ACTION_AMBIGUITY_MAIN.png"
    im.save(out_png, dpi=(300, 300))
    # PDF contains the same 300-DPI figure, preserving a standard single-column-width layout.
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        out_pdf = OUT / "FIG_STAGE_ACTION_AMBIGUITY_MAIN.pdf"
        c = canvas.Canvas(str(out_pdf), pagesize=(W / 300 * 72, H / 300 * 72))
        c.drawImage(ImageReader(str(out_png)), 0, 0, width=W / 300 * 72, height=H / 300 * 72)
        c.showPage(); c.save()
    except ImportError:
        im.save(OUT / "FIG_STAGE_ACTION_AMBIGUITY_MAIN.pdf", "PDF", resolution=300.0)
    audit = {
        "DATA_SOURCE_CHECK": "PASS",
        "dataset": "recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED / NORMAL_SUCCESS train only",
        "ground_truth_stage_field": "active_phase",
        "normal_success_train_episodes": normal_episodes,
        "state_field": "full_physical_state (physical_state_43)",
        "action_field": "executed_action (expert_action_7)",
        "stage_pair": "2 (Transport) -> 3 (Place)",
        "cross_episode_only": True,
        "epsilon": EPSILON,
        "close_pair_count": int(len(close)),
        "mean_state_distance": state_mean,
        "mean_action_distance": action_mean,
        "variance_reduction_percent": reduction,
        "FIG_GENERATED": "YES",
    }
    (OUT / "FIG_STAGE_ACTION_AMBIGUITY_MAIN_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
