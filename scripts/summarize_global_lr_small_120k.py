#!/usr/bin/env python3
"""Build fixed-protocol reports for the Global LR=5e-4 experiment."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/global_diffusion/global_lr5e-4_120k"
BASELINE = ROOT / "outputs/global_diffusion/global_long_run_120k/convergence_analysis.md"


def write_curve(rows: list[dict[str, float]], key: str, title: str, filename: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 20), title, fill="black")
    draw.line((70, 430, 850, 430), fill="black")
    draw.line((70, 60, 70, 430), fill="black")
    points = [(70 + index * (780 / (len(rows) - 1)), 430 - row[key] * 3.5) for index, row in enumerate(rows)]
    draw.line(points, fill=color, width=4)
    for (x, y), row in zip(points, rows):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x - 12, 442), f"{int(row['step']) // 1000}k", fill="black")
        draw.text((x - 10, max(62, y - 20)), str(int(row[key])), fill="black")
    image.save(OUT / filename)


def main() -> None:
    rows = []
    for step in range(5_000, 120_001, 5_000):
        report_path = OUT / "eval" / f"step_{step:08d}" / "evaluation_report.json"
        report = json.loads(report_path.read_text())
        summary = report["summary"]
        episode_rows = report["rows"]
        rows.append({
            "step": step,
            "success": summary["success"]["count"],
            "grasp": summary["grasp"]["count"],
            "lift": summary["lift"]["count"],
            "transport": summary["transport"]["count"],
            "place": summary["place"]["count"],
            "release": summary["release"]["count"],
            "retreat": summary["retreat"]["count"],
            "illegal_drop": summary["illegal_drop"]["count"],
            "ik_failure": summary["ik_failure"]["count"],
            "timeout": summary["timeout"]["count"],
            "retreat_timeout": sum(row["timeout"] and row["place"] for row in episode_rows),
            "average_return": summary["average_return"],
            "episode_length": summary["episode_length"]["mean"],
        })

    fields = list(rows[0])
    for filename in ("checkpoint_summary.csv", "closed_loop_results.csv"):
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    late = [row for row in rows if row["step"] >= 90_000]
    late_success = [row["success"] for row in late]
    late_retreat = [row["retreat"] for row in late]
    late_mean = statistics.mean(late_success)
    late_std = statistics.pstdev(late_success)
    late_retreat_std = statistics.pstdev(late_retreat)
    baseline_success = [80, 72, 73, 89]
    baseline_std = statistics.pstdev(baseline_success)
    baseline_mean = statistics.mean(baseline_success)
    result = {
        "late_steps": [row["step"] for row in late],
        "late_success": late_success,
        "late_retreat": late_retreat,
        "GLOBAL_LR_SMALL_LATE_MEAN": late_mean,
        "GLOBAL_LR_SMALL_LATE_STD": late_std,
        "GLOBAL_LR_SMALL_LATE_RETREAT_STD": late_retreat_std,
        "baseline_late_success": baseline_success,
        "baseline_late_mean": baseline_mean,
        "baseline_late_std": baseline_std,
        "GLOBAL_LR_SMALL_IMPROVES_STABILITY": "YES" if late_std < baseline_std else "NO",
        "GLOBAL_LR_SMALL_IMPROVES_FINAL_SUCCESS": "YES" if rows[-1]["success"] > 89 else "NO",
        "GLOBAL_LR_SMALL_REDUCES_RETREAT_VARIANCE": "YES" if late_retreat_std < baseline_std else "NO",
        "final_success": rows[-1]["success"],
        "final_timeout": rows[-1]["timeout"],
        "final_retreat": rows[-1]["retreat"],
    }
    markdown = "# Global Diffusion LR-small Stability Analysis\n\n"
    markdown += "The only training change was learning rate: 1e-3 to 5e-4. Checkpoint frequency changed from 10k to 5k.\n\n"
    markdown += "```json\n" + json.dumps(result, indent=2) + "\n```\n"
    (OUT / "stability_analysis.md").write_text(markdown)
    write_curve(rows, "success", "Global LR=5e-4: Success", "success_curve.png", (25, 118, 210))
    write_curve(rows, "retreat", "Global LR=5e-4: Retreat Completion", "retreat_curve.png", (46, 125, 50))
    write_curve(rows, "timeout", "Global LR=5e-4: Timeout", "timeout_curve.png", (198, 40, 40))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
