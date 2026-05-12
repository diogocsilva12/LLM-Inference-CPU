#!/usr/bin/env python3
"""Build a simple TPOT performance-model table from benchmark outputs."""

import argparse
import csv
import statistics
from pathlib import Path


def safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def arch_for(path):
    parts = set(path.parts)
    if "arm" in parts:
        return "arm"
    if "x86" in parts:
        return "x86"
    return "unknown"


def read_bandwidths(root):
    bandwidths = {}
    for path in root.rglob("memory_bandwidth.csv"):
        values = []
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = safe_float(row.get("triad_gb_s"))
                if value is not None:
                    values.append(value)
        if values:
            bandwidths[arch_for(path)] = {
                "bw_mem_gb_s_max": max(values),
                "bw_mem_gb_s_median": statistics.median(values),
                "source": str(path),
            }
    return bandwidths


def model_size_bytes(model_path):
    if not model_path:
        return None
    path = Path(model_path)
    if path.exists():
        return path.stat().st_size
    return None


def iter_scaling_rows(root):
    for path in root.rglob("scaling_summary.csv"):
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_summary_path"] = str(path)
                row["_arch"] = arch_for(path)
                yield row


def build_rows(root):
    bandwidths = read_bandwidths(root)
    rows = []
    for row in iter_scaling_rows(root):
        arch = row["_arch"]
        bw = bandwidths.get(arch)
        observed_tpot = safe_float(row.get("tpot_s_mean_across_replicas"))
        model_bytes = model_size_bytes(row.get("model"))
        if not bw or observed_tpot is None or model_bytes is None:
            continue
        bw_bytes_s = bw["bw_mem_gb_s_median"] * 1.0e9
        predicted_tpot = model_bytes / bw_bytes_s if bw_bytes_s > 0 else None
        ratio = observed_tpot / predicted_tpot if predicted_tpot and predicted_tpot > 0 else None
        rows.append({
            "arch": arch,
            "engine": row.get("engine"),
            "model": row.get("model"),
            "model_bytes": model_bytes,
            "threads": row.get("threads"),
            "ctx_size": row.get("ctx_size"),
            "parallel_requests": row.get("parallel_requests"),
            "cache_type_k": row.get("cache_type_k"),
            "cache_type_v": row.get("cache_type_v"),
            "max_tokens": row.get("max_tokens"),
            "category": row.get("category"),
            "prompt_id": row.get("prompt_id"),
            "bw_mem_gb_s_median": bw["bw_mem_gb_s_median"],
            "tpot_pred_s": predicted_tpot,
            "tpot_observed_s": observed_tpot,
            "observed_over_predicted": ratio,
            "throughput_output_tok_s": row.get("total_throughput_output_tok_s_mean"),
            "max_vmhwm_kb": row.get("max_vmhwm_kb"),
            "bandwidth_source": bw["source"],
            "summary_source": row["_summary_path"],
        })
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arch",
        "engine",
        "model",
        "model_bytes",
        "threads",
        "ctx_size",
        "parallel_requests",
        "cache_type_k",
        "cache_type_v",
        "max_tokens",
        "category",
        "prompt_id",
        "bw_mem_gb_s_median",
        "tpot_pred_s",
        "tpot_observed_s",
        "observed_over_predicted",
        "throughput_output_tok_s",
        "max_vmhwm_kb",
        "bandwidth_source",
        "summary_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Ultimate suite output root")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    output = args.out or args.root / "bottleneck_model.csv"
    rows = build_rows(args.root)
    write_csv(rows, output)
    print(f"wrote {len(rows)} rows to {output}")
    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
