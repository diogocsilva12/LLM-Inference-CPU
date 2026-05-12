#!/usr/bin/env python3
"""Create lightweight report tables and SVG plots from benchmark summaries."""

import argparse
import csv
import math
import re
from pathlib import Path


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def arch_for(path):
    parts = set(path.parts)
    if "arm" in parts:
        return "arm"
    if "x86" in parts:
        return "x86"
    return "unknown"


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_scaling_rows(root):
    rows = []
    for path in root.rglob("scaling_summary.csv"):
        for row in read_csv(path):
            row["arch"] = arch_for(path)
            row["suite"] = path.parent.name
            row["source"] = str(path)
            rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def points_for(rows, suite_contains, x_key, y_key, filters=None):
    filters = filters or {}
    points = []
    for row in rows:
        if suite_contains not in row.get("suite", ""):
            continue
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y, row))
    return points


def suite_model_index(suite):
    match = re.match(r"model-(\d+)", suite or "")
    if not match:
        return None
    return float(match.group(1))


def scatter_svg(path, title, x_label, y_label, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 460
    left, right, top, bottom = 90, 30, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not points:
        path.write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
            f"<text x='20' y='40'>{title}: no data</text></svg>\n",
            encoding="utf-8",
        )
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min *= 0.9
        y_max *= 1.1 if y_max else 1

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    colors = {"arm": "#1f77b4", "x86": "#d62728", "unknown": "#2ca02c"}
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<text x='{left}' y='30' font-size='18'>{title}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='black'/>",
        f"<text x='{left + plot_w / 2 - 60}' y='{height - 20}' font-size='13'>{x_label}</text>",
        f"<text x='15' y='{top + plot_h / 2}' font-size='13' transform='rotate(-90 15,{top + plot_h / 2})'>{y_label}</text>",
        f"<text x='{left}' y='{top + plot_h + 20}' font-size='11'>{x_min:.3g}</text>",
        f"<text x='{left + plot_w - 40}' y='{top + plot_h + 20}' font-size='11'>{x_max:.3g}</text>",
        f"<text x='{left - 70}' y='{top + plot_h}' font-size='11'>{y_min:.3g}</text>",
        f"<text x='{left - 70}' y='{top + 5}' font-size='11'>{y_max:.3g}</text>",
    ]
    for x, y, row in points:
        arch = row.get("arch", "unknown")
        parts.append(
            f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{colors.get(arch, '#2ca02c')}'/>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "report_artifacts"
    rows = collect_scaling_rows(args.root)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        write_csv(out_dir / "all_scaling_rows.csv", rows, fieldnames)

    scatter_svg(
        out_dir / "throughput_vs_threads.svg",
        "Throughput vs thread count",
        "threads",
        "output tokens/s",
        points_for(rows, "thread-scaling", "threads", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "ttft_vs_context.svg",
        "TTFT vs context size",
        "context size",
        "TTFT seconds",
        points_for(rows, "context-length", "ctx_size", "ttft_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "memory_vs_model.svg",
        "Memory vs model experiment index",
        "model run index",
        "VmHWM KB",
        [
            (suite_model_index(row.get("suite", "")), safe_float(row.get("max_vmhwm_kb")), row)
            for row in rows
            if suite_model_index(row.get("suite", "")) is not None
            and safe_float(row.get("max_vmhwm_kb")) is not None
        ],
    )

    bottleneck = args.root / "bottleneck_model.csv"
    if bottleneck.exists():
        model_rows = read_csv(bottleneck)
        points = []
        for row in model_rows:
            x = safe_float(row.get("tpot_pred_s"))
            y = safe_float(row.get("tpot_observed_s"))
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                points.append((x, y, row))
        scatter_svg(
            out_dir / "predicted_vs_observed_tpot.svg",
            "Predicted vs observed TPOT",
            "predicted TPOT seconds",
            "observed TPOT seconds",
            points,
        )

    print(f"wrote report artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
