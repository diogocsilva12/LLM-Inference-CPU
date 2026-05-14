#!/usr/bin/env python3
"""Summarize Track A1 raw benchmark logs into CSV tables."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ENGINE_DIR_NAMES = {"tq", "vanilla", "mtp"}


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_resources(path):
    if not path.exists():
        return None, None
    max_rss = None
    max_hwm = None
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rss = int(float(row.get("rss_kb") or 0))
            hwm = int(float(row.get("vmhwm_kb") or 0))
            max_rss = rss if max_rss is None else max(max_rss, rss)
            max_hwm = hwm if max_hwm is None else max(max_hwm, hwm)
    return max_rss, max_hwm


def read_system(path):
    if not path.exists():
        return {}
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            values[key] = value
    return values


def parse_path_metadata(run_dir):
    values = {
        "node_rank": None,
        "node": None,
        "threads": None,
        "ctx_size": None,
        "parallel_requests": None,
        "cache_type_k": None,
        "cache_type_v": None,
    }
    node_match = re.match(r"node_(\d+)_(.+)", run_dir.parent.name)
    if node_match:
        values["node_rank"] = node_match.group(1)
        values["node"] = node_match.group(2)
    run_match = re.match(r"t_(\d+)(?:_c_(\d+))?(?:_p_(\d+))?_k_(.+)_v_(.+)", run_dir.name)
    if run_match:
        values["threads"] = run_match.group(1)
        if run_match.group(2):
            values["ctx_size"] = run_match.group(2)
        if run_match.group(3):
            values["parallel_requests"] = run_match.group(3)
        values["cache_type_k"] = run_match.group(4)
        values["cache_type_v"] = run_match.group(5)
    return values


def experiment_dir_for(run_dir, system=None):
    if system and system.get("suite"):
        for parent in run_dir.parents:
            if parent.name == system["suite"]:
                return str(parent.parent)
    if re.match(r"node_\d+_.+", run_dir.parent.name):
        return str(run_dir.parent.parent)
    if run_dir.name in ENGINE_DIR_NAMES:
        return str(run_dir.parent)
    is_sweep_dir = re.match(r"t_\d+(?:_c_\d+)?(?:_p_\d+)?_k_.+_v_.+", run_dir.name)
    if is_sweep_dir and run_dir.parent.name in ENGINE_DIR_NAMES:
        return str(run_dir.parent.parent)
    return str(run_dir)


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_run(run_dir, ttft_sla, tpot_sla):
    request_path = run_dir / "requests.jsonl"
    if not request_path.exists():
        return []
    rows = read_jsonl(request_path)
    max_rss_kb, max_hwm_kb = read_resources(run_dir / "resources.csv")
    system = read_system(run_dir / "system.txt")
    path_metadata = parse_path_metadata(run_dir)

    groups = {}
    for row in rows:
        groups.setdefault((row.get("category", "unknown"), row["prompt_id"]), []).append(row)

    summary = []
    for (category, prompt_id), items in sorted(groups.items()):
        ok = [r for r in items if not r.get("error")]
        ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
        tpot = [float(r["tpot_s"]) for r in ok if r.get("tpot_s") is not None]
        throughput = [float(r["throughput_output_tok_s"]) for r in ok if r.get("throughput_output_tok_s") is not None]
        good = [
            r for r in ok
            if r.get("ttft_s") is not None and r.get("tpot_s") is not None
            and float(r["ttft_s"]) <= ttft_sla and float(r["tpot_s"]) <= tpot_sla
        ]
        summary.append({
            "experiment_dir": experiment_dir_for(run_dir, system),
            "run_dir": str(run_dir),
            "suite": system.get("suite"),
            "engine": system.get("engine"),
            "node_count": system.get("node_count"),
            "node_rank": system.get("rank") or path_metadata["node_rank"],
            "node": system.get("node") or path_metadata["node"],
            "model": system.get("model"),
            "model_label": system.get("model_label"),
            "threads": system.get("threads") or path_metadata["threads"],
            "threads_batch": system.get("threads_batch"),
            "ctx_size": system.get("ctx_size") or path_metadata["ctx_size"],
            "parallel_requests": system.get("parallel_requests") or path_metadata["parallel_requests"],
            "cache_type_k": system.get("cache_type_k") or path_metadata["cache_type_k"],
            "cache_type_v": system.get("cache_type_v") or path_metadata["cache_type_v"],
            "max_tokens": system.get("max_tokens"),
            "trials_configured": system.get("trials"),
            "warmup_trials": system.get("warmup_trials"),
            "bench_concurrency": system.get("bench_concurrency"),
            "mandatory_only": system.get("mandatory_only"),
            "limit_per_category": system.get("limit_per_category"),
            "category": category,
            "prompt_id": prompt_id,
            "runs": len(items),
            "successful_runs": len(ok),
            "ttft_s_mean": mean(ttft),
            "ttft_s_stdev": stdev(ttft),
            "tpot_s_mean": mean(tpot),
            "tpot_s_stdev": stdev(tpot),
            "throughput_output_tok_s_mean": mean(throughput),
            "throughput_output_tok_s_stdev": stdev(throughput),
            "goodput_sla_fraction": len(good) / len(items) if items else None,
            "max_rss_kb": max_rss_kb,
            "max_vmhwm_kb": max_hwm_kb,
        })
    return summary


def safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def aggregate_scaling(rows):
    groups = {}
    for row in rows:
        key = (
            row.get("experiment_dir"),
            row.get("suite"),
            row.get("engine"),
            row.get("node_count"),
            row.get("model"),
            row.get("model_label"),
            row.get("threads"),
            row.get("threads_batch"),
            row.get("ctx_size"),
            row.get("parallel_requests"),
            row.get("cache_type_k"),
            row.get("cache_type_v"),
            row.get("max_tokens"),
            row.get("category"),
            row.get("prompt_id"),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, items in sorted(groups.items()):
        (
            experiment_dir,
            suite,
            engine,
            node_count,
            model,
            model_label,
            threads,
            threads_batch,
            ctx_size,
            parallel_requests,
            cache_type_k,
            cache_type_v,
            max_tokens,
            category,
            prompt_id,
        ) = key
        throughput = [safe_float(r.get("throughput_output_tok_s_mean")) for r in items]
        ttft = [safe_float(r.get("ttft_s_mean")) for r in items]
        tpot = [safe_float(r.get("tpot_s_mean")) for r in items]
        goodput = [safe_float(r.get("goodput_sla_fraction")) for r in items]
        vmhwm = [safe_float(r.get("max_vmhwm_kb")) for r in items]
        throughput_clean = [v for v in throughput if v is not None]
        node_ids = {r.get("node_rank") or r.get("run_dir") for r in items}
        replica_count = len(node_ids)
        if node_count and suite and "node-scaling" in suite:
            replica_count = int(node_count)
        aggregate_rows.append({
            "experiment_dir": experiment_dir,
            "suite": suite,
            "engine": engine,
            "node_count": node_count,
            "replica_count": replica_count,
            "model": model,
            "model_label": model_label,
            "threads": threads,
            "threads_batch": threads_batch,
            "ctx_size": ctx_size,
            "parallel_requests": parallel_requests,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "max_tokens": max_tokens,
            "category": category,
            "prompt_id": prompt_id,
            "successful_runs": sum(int(r.get("successful_runs") or 0) for r in items),
            "total_throughput_output_tok_s_mean": sum(throughput_clean) if throughput_clean else None,
            "ttft_s_mean_across_replicas": mean([v for v in ttft if v is not None]),
            "tpot_s_mean_across_replicas": mean([v for v in tpot if v is not None]),
            "goodput_sla_fraction_mean": mean([v for v in goodput if v is not None]),
            "max_vmhwm_kb": max([v for v in vmhwm if v is not None], default=None),
        })
    return aggregate_rows


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Run directories or a parent measurements directory")
    parser.add_argument("--out", type=Path, default=Path("measurements/a1_summary.csv"))
    parser.add_argument("--scaling-out", type=Path, help="Optional aggregate CSV for replica/thread scaling.")
    parser.add_argument("--ttft-sla", type=float, default=2.0)
    parser.add_argument("--tpot-sla", type=float, default=0.2)
    args = parser.parse_args()

    run_dirs = []
    for path in args.paths:
        if (path / "requests.jsonl").exists():
            run_dirs.append(path)
        elif path.is_dir():
            run_dirs.extend(sorted(p.parent for p in path.rglob("requests.jsonl")))
    run_dirs = sorted(set(run_dirs))

    all_rows = []
    for run_dir in run_dirs:
        all_rows.extend(summarize_run(run_dir, args.ttft_sla, args.tpot_sla))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_dir",
        "run_dir",
        "suite",
        "engine",
        "node_count",
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
    write_csv(all_rows, args.out, fieldnames)
    if args.scaling_out:
        scaling_fieldnames = [
            "experiment_dir",
            "suite",
            "engine",
            "node_count",
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
        write_csv(aggregate_scaling(all_rows), args.scaling_out, scaling_fieldnames)
    print(f"wrote {len(all_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
