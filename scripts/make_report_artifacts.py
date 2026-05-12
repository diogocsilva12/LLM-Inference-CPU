#!/usr/bin/env python3
"""Create lightweight report tables and SVG plots from benchmark summaries."""

import argparse
import csv
import math
import re
from pathlib import Path

import summarize_a1_results


DEFAULT_SUMMARY_NAME = "a1_all_results_summary.csv"
DEFAULT_SCALING_NAME = "a1_all_results_scaling_summary.csv"


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
    if path.name.startswith("._"):
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_run_dirs(root):
    run_dirs = []
    if (root / "requests.jsonl").exists():
        run_dirs.append(root)
    elif root.is_dir():
        run_dirs.extend(sorted(p.parent for p in root.rglob("requests.jsonl")))
    return sorted(set(run_dirs))


def regenerate_summaries(root, summary_out, scaling_out, ttft_sla, tpot_sla):
    rows = []
    for run_dir in discover_run_dirs(root):
        rows.extend(summarize_a1_results.summarize_run(run_dir, ttft_sla, tpot_sla))

    summary_fields = [
        "experiment_dir",
        "run_dir",
        "suite",
        "engine",
        "node_rank",
        "node",
        "model",
        "model_label",
        "threads",
        "threads_batch",
        "ctx_size",
        "parallel_requests",
        "cache_type_k",
        "cache_type_v",
        "max_tokens",
        "trials_configured",
        "warmup_trials",
        "bench_concurrency",
        "mandatory_only",
        "limit_per_category",
        "category",
        "prompt_id",
        "runs",
        "successful_runs",
        "ttft_s_mean",
        "ttft_s_stdev",
        "tpot_s_mean",
        "tpot_s_stdev",
        "throughput_output_tok_s_mean",
        "throughput_output_tok_s_stdev",
        "goodput_sla_fraction",
        "max_rss_kb",
        "max_vmhwm_kb",
    ]
    scaling_fields = [
        "experiment_dir",
        "suite",
        "engine",
        "replica_count",
        "model",
        "model_label",
        "threads",
        "threads_batch",
        "ctx_size",
        "parallel_requests",
        "cache_type_k",
        "cache_type_v",
        "max_tokens",
        "category",
        "prompt_id",
        "successful_runs",
        "total_throughput_output_tok_s_mean",
        "ttft_s_mean_across_replicas",
        "tpot_s_mean_across_replicas",
        "goodput_sla_fraction_mean",
        "max_vmhwm_kb",
    ]
    write_csv(summary_out, rows, summary_fields)
    scaling_rows = summarize_a1_results.aggregate_scaling(rows)
    write_csv(scaling_out, scaling_rows, scaling_fields)
    return rows, scaling_rows


def collect_scaling_rows(root):
    rows = []
    for path in root.rglob("*scaling_summary.csv"):
        if path.name.startswith("._"):
            continue
        for row in read_csv(path):
            row["arch"] = arch_for(path)
            if not row.get("suite"):
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


def number_or_index(value, index):
    numeric = safe_float(value)
    return numeric if numeric is not None else float(index)


def mean(values):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def categorical_mean_points(rows, category_key, value_key, filters=None):
    filters = filters or {}
    groups = {}
    labels = []
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        label = row.get(category_key) or "unknown"
        value = safe_float(row.get(value_key))
        if value is None:
            continue
        if label not in groups:
            labels.append(label)
            groups[label] = []
        groups[label].append(value)
    return [
        (float(idx + 1), mean(groups[label]), {"label": label, "arch": "unknown"})
        for idx, label in enumerate(sorted(labels))
        if mean(groups[label]) is not None
    ]


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
        label = row.get("label")
        parts.append(
            f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{colors.get(arch, '#2ca02c')}'/>"
        )
        if label:
            parts.append(
                f"<text x='{sx(x) + 6:.2f}' y='{sy(y) - 6:.2f}' font-size='10'>{label}</text>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_standard_plots(out_dir, rows):
    scatter_svg(
        out_dir / "throughput_vs_threads.svg",
        "Throughput vs thread count",
        "threads",
        "output tokens/s",
        points_for(rows, "thread-scaling", "threads", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "tpot_vs_threads.svg",
        "TPOT vs thread count",
        "threads",
        "TPOT seconds",
        points_for(rows, "thread-scaling", "threads", "tpot_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "ttft_vs_threads.svg",
        "TTFT vs thread count",
        "threads",
        "TTFT seconds",
        points_for(rows, "thread-scaling", "threads", "ttft_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "throughput_vs_concurrency.svg",
        "Throughput vs concurrent requests",
        "parallel requests",
        "output tokens/s",
        points_for(rows, "concurrency", "parallel_requests", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "tpot_vs_concurrency.svg",
        "TPOT vs concurrent requests",
        "parallel requests",
        "TPOT seconds",
        points_for(rows, "concurrency", "parallel_requests", "tpot_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "ttft_vs_context.svg",
        "TTFT vs context size",
        "context size",
        "TTFT seconds",
        points_for(rows, "context-length", "ctx_size", "ttft_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "tpot_vs_context.svg",
        "TPOT vs context size",
        "context size",
        "TPOT seconds",
        points_for(rows, "context-length", "ctx_size", "tpot_s_mean_across_replicas"),
    )
    scatter_svg(
        out_dir / "throughput_vs_decode_length.svg",
        "Throughput vs decode length",
        "max tokens",
        "output tokens/s",
        points_for(rows, "decode-length", "max_tokens", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "goodput_vs_decode_length.svg",
        "Goodput vs decode length",
        "max tokens",
        "SLA goodput fraction",
        points_for(rows, "decode-length", "max_tokens", "goodput_sla_fraction_mean"),
    )
    scatter_svg(
        out_dir / "memory_vs_threads.svg",
        "Memory vs thread count",
        "threads",
        "VmHWM KB",
        points_for(rows, "thread-scaling", "threads", "max_vmhwm_kb"),
    )
    scatter_svg(
        out_dir / "throughput_by_engine.svg",
        "Mean throughput by engine",
        "engine index",
        "output tokens/s",
        categorical_mean_points(rows, "engine", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "goodput_by_engine.svg",
        "Mean goodput by engine",
        "engine index",
        "SLA goodput fraction",
        categorical_mean_points(rows, "engine", "goodput_sla_fraction_mean"),
    )
    scatter_svg(
        out_dir / "throughput_by_cache.svg",
        "Mean throughput by K cache type",
        "cache index",
        "output tokens/s",
        categorical_mean_points(rows, "cache_type_k", "total_throughput_output_tok_s_mean"),
    )
    scatter_svg(
        out_dir / "memory_vs_model.svg",
        "Memory vs model experiment index",
        "model run index",
        "VmHWM KB",
        [
            (suite_model_index(row.get("model_label", "") or row.get("suite", "")), safe_float(row.get("max_vmhwm_kb")), row)
            for row in rows
            if suite_model_index(row.get("model_label", "") or row.get("suite", "")) is not None
            and safe_float(row.get("max_vmhwm_kb")) is not None
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--scaling-out", type=Path)
    parser.add_argument("--skip-regenerate", action="store_true")
    parser.add_argument("--ttft-sla", type=float, default=2.0)
    parser.add_argument("--tpot-sla", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "report_artifacts"
    summary_out = args.summary_out or args.root / DEFAULT_SUMMARY_NAME
    scaling_out = args.scaling_out or args.root / DEFAULT_SCALING_NAME

    if not args.skip_regenerate:
        regenerate_summaries(args.root, summary_out, scaling_out, args.ttft_sla, args.tpot_sla)
        rows = read_csv(scaling_out)
        for row in rows:
            row["arch"] = arch_for(scaling_out)
            row["source"] = str(scaling_out)
    else:
        rows = collect_scaling_rows(args.root)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        write_csv(out_dir / "all_scaling_rows.csv", rows, fieldnames)

    write_standard_plots(out_dir, rows)

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
