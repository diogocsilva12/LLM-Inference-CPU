#!/usr/bin/env python3
"""Extract readable Track A1 benchmark rows and prompt answers into CSV files."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import summarize_a1_results as summarize


METRIC_FIELDS = [
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
    "bench_concurrency",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "trial",
    "prompt_id",
    "category",
    "mandatory",
    "status",
    "error",
    "ttft_s",
    "tpot_s",
    "total_s",
    "output_token_events",
    "throughput_output_tok_s",
    "max_rss_kb",
    "max_vmhwm_kb",
]

ANSWER_FIELDS = [
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
    "ctx_size",
    "parallel_requests",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "trial",
    "prompt_id",
    "category",
    "mandatory",
    "prompt_text",
    "status",
    "error",
    "ttft_s",
    "tpot_s",
    "total_s",
    "output_token_events",
    "throughput_output_tok_s",
    "generated_text",
]

SUMMARY_FIELDS = [
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
    "ctx_size",
    "parallel_requests",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "prompt_id",
    "category",
    "mandatory",
    "prompt_text",
    "runs",
    "successful_runs",
    "error_runs",
    "first_success_answer",
]

PROMPT_FIELDS = ["prompt_id", "category", "mandatory", "length_target", "prompt_text"]

PROMPT_SUMMARY_FIELDS = [
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
    "bench_concurrency",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "category",
    "prompt_id",
    "mandatory",
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

SCALING_SUMMARY_FIELDS = [
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

GLOBAL_METRIC_FIELDS = [
    "scope",
    "configuration_count",
    "model_count",
    "engine_count",
    "prompt_count",
    "request_count",
    "successful_runs",
    "ttft_s_mean",
    "tpot_s_mean",
    "throughput_output_tok_s_mean",
    "total_throughput_output_tok_s_mean",
    "goodput_sla_fraction_mean",
    "max_vmhwm_kb",
]

PER_MODEL_FIELDS = [
    "model_label",
    "model",
    "configuration_count",
    "engine_count",
    "prompt_count",
    "request_count",
    "successful_runs",
    "ttft_s_mean",
    "tpot_s_mean",
    "throughput_output_tok_s_mean",
    "total_throughput_output_tok_s_mean",
    "goodput_sla_fraction_mean",
    "max_vmhwm_kb",
]

NODE_SCALING_FIELDS = [
    "suite",
    "node_count",
    "replica_count",
    "engine",
    "model_label",
    "model",
    "threads",
    "threads_batch",
    "ctx_size",
    "parallel_requests",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "prompt_count",
    "successful_runs",
    "total_throughput_output_tok_s_mean",
    "ttft_s_mean_across_replicas",
    "tpot_s_mean_across_replicas",
    "goodput_sla_fraction_mean",
    "max_vmhwm_kb",
]

NODE_RESOURCE_FIELDS = [
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
    "ctx_size",
    "parallel_requests",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
    "source_file",
    "source_format",
    "sample_index",
    "timestamp",
    "role",
    "cpu_user_percent",
    "cpu_system_percent",
    "cpu_idle_percent",
    "cpu_wait_percent",
    "cpu_user_jiffies",
    "cpu_nice_jiffies",
    "cpu_system_jiffies",
    "cpu_idle_jiffies",
    "cpu_iowait_jiffies",
    "cpu_irq_jiffies",
    "cpu_softirq_jiffies",
    "cpu_steal_jiffies",
    "mem_total_kb",
    "mem_available_kb",
    "mem_used_kb",
    "mem_free_kb",
    "swap_total_kb",
    "swap_free_kb",
    "dstat_mem_used",
    "dstat_mem_free",
    "dstat_mem_buff",
    "dstat_mem_cache",
]


def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def load_prompts(path):
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = {}
    for prompt in data.get("prompts", []):
        prompt_id = prompt.get("id")
        if not prompt_id:
            continue
        prompts[prompt_id] = {
            "prompt_id": prompt_id,
            "category": prompt.get("category", ""),
            "mandatory": prompt.get("mandatory", ""),
            "length_target": prompt.get("length_target", ""),
            "prompt_text": clean_text(prompt.get("text", "")),
        }
    return prompts


def iter_request_files(paths):
    for path in paths:
        if path.is_file() and path.name == "requests.jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("requests.jsonl"))


def iter_node_resource_files(paths):
    for path in paths:
        if path.is_file() and path.name == "node_resources.csv":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("node_resources.csv"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_metadata(run_dir):
    system = summarize.read_system(run_dir / "system.txt")
    path_metadata = summarize.parse_path_metadata(run_dir)
    max_rss_kb, max_vmhwm_kb = summarize.read_resources(run_dir / "resources.csv")
    return {
        "experiment_dir": summarize.experiment_dir_for(run_dir, system),
        "run_dir": str(run_dir),
        "suite": system.get("suite", ""),
        "engine": system.get("engine", ""),
        "node_count": system.get("node_count", ""),
        "node_rank": system.get("rank") or path_metadata.get("node_rank") or "",
        "node": system.get("node") or path_metadata.get("node") or "",
        "model": system.get("model", ""),
        "model_label": system.get("model_label", ""),
        "threads": system.get("threads") or path_metadata.get("threads") or "",
        "threads_batch": system.get("threads_batch", ""),
        "ctx_size": system.get("ctx_size") or path_metadata.get("ctx_size") or "",
        "parallel_requests": system.get("parallel_requests") or path_metadata.get("parallel_requests") or "",
        "bench_concurrency": system.get("bench_concurrency", ""),
        "cache_type_k": system.get("cache_type_k") or path_metadata.get("cache_type_k") or "",
        "cache_type_v": system.get("cache_type_v") or path_metadata.get("cache_type_v") or "",
        "max_tokens": system.get("max_tokens", ""),
        "max_rss_kb": max_rss_kb,
        "max_vmhwm_kb": max_vmhwm_kb,
    }


def collect_rows(paths, prompts_path):
    prompts = load_prompts(prompts_path)
    metric_rows = []
    answer_rows = []

    for request_file in iter_request_files(paths):
        metadata = run_metadata(request_file.parent)
        for request in read_jsonl(request_file):
            prompt_id = request.get("prompt_id", "")
            prompt = prompts.get(prompt_id, {})
            category = request.get("category") or prompt.get("category", "")
            mandatory = request.get("mandatory")
            if mandatory is None:
                mandatory = prompt.get("mandatory", "")

            common = {
                **metadata,
                "trial": request.get("trial", ""),
                "prompt_id": prompt_id,
                "category": category,
                "mandatory": mandatory,
                "status": request.get("status", ""),
                "error": request.get("error", ""),
                "ttft_s": request.get("ttft_s", ""),
                "tpot_s": request.get("tpot_s", ""),
                "total_s": request.get("total_s", ""),
                "output_token_events": request.get("output_token_events", ""),
                "throughput_output_tok_s": request.get("throughput_output_tok_s", ""),
            }
            metric_rows.append({field: common.get(field, "") for field in METRIC_FIELDS})
            answer_rows.append({
                **{field: common.get(field, "") for field in ANSWER_FIELDS if field not in {"prompt_text", "generated_text"}},
                "prompt_text": prompt.get("prompt_text", ""),
                "generated_text": clean_text(request.get("generated_text", "")),
            })

    return metric_rows, answer_rows


def dstat_header_index(lines):
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "usr" in lowered and "sys" in lowered and ("idl" in lowered or "idle" in lowered):
            return index
    return None


def normalize_dstat_field(row, *names):
    lowered = {key.strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        if name in lowered:
            return lowered[name]
    return ""


def collect_node_resource_rows(paths):
    rows = []
    for resource_file in iter_node_resource_files(paths):
        metadata = run_metadata(resource_file.parent)
        lines = resource_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            continue

        if lines[0].startswith("timestamp,role,"):
            reader = csv.DictReader(lines)
            for index, row in enumerate(reader, start=1):
                common = {
                    **metadata,
                    "source_file": str(resource_file),
                    "source_format": "procfs",
                    "sample_index": index,
                    "timestamp": row.get("timestamp", ""),
                    "role": row.get("role", ""),
                    "cpu_user_jiffies": row.get("cpu_user_jiffies", ""),
                    "cpu_nice_jiffies": row.get("cpu_nice_jiffies", ""),
                    "cpu_system_jiffies": row.get("cpu_system_jiffies", ""),
                    "cpu_idle_jiffies": row.get("cpu_idle_jiffies", ""),
                    "cpu_iowait_jiffies": row.get("cpu_iowait_jiffies", ""),
                    "cpu_irq_jiffies": row.get("cpu_irq_jiffies", ""),
                    "cpu_softirq_jiffies": row.get("cpu_softirq_jiffies", ""),
                    "cpu_steal_jiffies": row.get("cpu_steal_jiffies", ""),
                    "mem_total_kb": row.get("mem_total_kb", ""),
                    "mem_available_kb": row.get("mem_available_kb", ""),
                    "mem_used_kb": row.get("mem_used_kb", ""),
                    "mem_free_kb": row.get("mem_free_kb", ""),
                    "swap_total_kb": row.get("swap_total_kb", ""),
                    "swap_free_kb": row.get("swap_free_kb", ""),
                }
                rows.append({field: common.get(field, "") for field in NODE_RESOURCE_FIELDS})
            continue

        header_index = dstat_header_index(lines)
        if header_index is None:
            continue
        reader = csv.DictReader(lines[header_index:])
        for index, row in enumerate(reader, start=1):
            common = {
                **metadata,
                "source_file": str(resource_file),
                "source_format": "dstat",
                "sample_index": index,
                "timestamp": normalize_dstat_field(row, "time", "date/time", "epoch"),
                "role": metadata.get("engine", ""),
                "cpu_user_percent": normalize_dstat_field(row, "usr", "user"),
                "cpu_system_percent": normalize_dstat_field(row, "sys", "system"),
                "cpu_idle_percent": normalize_dstat_field(row, "idl", "idle"),
                "cpu_wait_percent": normalize_dstat_field(row, "wai", "wait"),
                "dstat_mem_used": normalize_dstat_field(row, "used"),
                "dstat_mem_free": normalize_dstat_field(row, "free"),
                "dstat_mem_buff": normalize_dstat_field(row, "buff"),
                "dstat_mem_cache": normalize_dstat_field(row, "cach", "cache"),
            }
            rows.append({field: common.get(field, "") for field in NODE_RESOURCE_FIELDS})
    return rows


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_output_tokens(row):
    return (safe_float(row.get("output_token_events")) or 0.0) > 0.0


def mean(values):
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else ""


def stdev(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return ""
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def group_rows(rows, fields):
    groups = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        groups.setdefault(key, []).append(row)
    return groups


def summarize_metric_rows(metric_rows, ttft_sla=2.0, tpot_sla=0.2):
    group_fields = [
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
        "bench_concurrency",
        "cache_type_k",
        "cache_type_v",
        "max_tokens",
        "category",
        "prompt_id",
        "mandatory",
    ]
    summary_rows = []
    for _, items in sorted(group_rows(metric_rows, group_fields).items()):
        first = items[0]
        ok = [row for row in items if not row.get("error")]
        ttft = [safe_float(row.get("ttft_s")) for row in ok]
        tpot = [safe_float(row.get("tpot_s")) for row in ok]
        throughput = [safe_float(row.get("throughput_output_tok_s")) for row in ok]
        good = [
            row for row in ok
            if safe_float(row.get("ttft_s")) is not None
            and safe_float(row.get("tpot_s")) is not None
            and safe_float(row.get("ttft_s")) <= ttft_sla
            and safe_float(row.get("tpot_s")) <= tpot_sla
        ]
        row = {field: first.get(field, "") for field in PROMPT_SUMMARY_FIELDS}
        row.update({
            "runs": len(items),
            "successful_runs": len(ok),
            "ttft_s_mean": mean(ttft),
            "ttft_s_stdev": stdev(ttft),
            "tpot_s_mean": mean(tpot),
            "tpot_s_stdev": stdev(tpot),
            "throughput_output_tok_s_mean": mean(throughput),
            "throughput_output_tok_s_stdev": stdev(throughput),
            "goodput_sla_fraction": len(good) / len(items) if items else "",
            "max_rss_kb": max((safe_float(row.get("max_rss_kb")) or 0.0 for row in items), default=""),
            "max_vmhwm_kb": max((safe_float(row.get("max_vmhwm_kb")) or 0.0 for row in items), default=""),
        })
        summary_rows.append(row)
    return summary_rows


def aggregate_scaling_rows(summary_rows):
    group_fields = [
        "experiment_dir",
        "suite",
        "engine",
        "node_count",
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
    ]
    scaling_rows = []
    for _, items in sorted(group_rows(summary_rows, group_fields).items()):
        first = items[0]
        throughput = [safe_float(row.get("throughput_output_tok_s_mean")) for row in items]
        ttft = [safe_float(row.get("ttft_s_mean")) for row in items]
        tpot = [safe_float(row.get("tpot_s_mean")) for row in items]
        goodput = [safe_float(row.get("goodput_sla_fraction")) for row in items]
        vmhwm = [safe_float(row.get("max_vmhwm_kb")) for row in items]
        node_ids = {row.get("node_rank") or row.get("run_dir") for row in items}
        replica_count = len(node_ids)
        if first.get("node_count") and "node-scaling" in first.get("suite", ""):
            replica_count = int(safe_float(first.get("node_count")) or replica_count)
        row = {field: first.get(field, "") for field in SCALING_SUMMARY_FIELDS}
        row.update({
            "replica_count": replica_count,
            "successful_runs": sum(int(safe_float(row.get("successful_runs")) or 0) for row in items),
            "total_throughput_output_tok_s_mean": sum(value for value in throughput if value is not None),
            "ttft_s_mean_across_replicas": mean(ttft),
            "tpot_s_mean_across_replicas": mean(tpot),
            "goodput_sla_fraction_mean": mean(goodput),
            "max_vmhwm_kb": max((value for value in vmhwm if value is not None), default=""),
        })
        scaling_rows.append(row)
    return scaling_rows


def config_key(row):
    return (
        row.get("experiment_dir", ""),
        row.get("suite", ""),
        row.get("engine", ""),
        row.get("node_count", ""),
        row.get("model_label", ""),
        row.get("threads", ""),
        row.get("ctx_size", ""),
        row.get("parallel_requests", ""),
        row.get("cache_type_k", ""),
        row.get("cache_type_v", ""),
        row.get("max_tokens", ""),
    )


def aggregate_metric_set(summary_rows, scope, extra_fields=None):
    extra_fields = extra_fields or {}
    request_count = sum(int(safe_float(row.get("runs")) or 0) for row in summary_rows)
    successful_runs = sum(int(safe_float(row.get("successful_runs")) or 0) for row in summary_rows)
    throughput = [safe_float(row.get("throughput_output_tok_s_mean")) for row in summary_rows]
    result = {
        **extra_fields,
        "scope": scope,
        "configuration_count": len({config_key(row) for row in summary_rows}),
        "model_count": len({row.get("model_label", "") for row in summary_rows if row.get("model_label", "")}),
        "engine_count": len({row.get("engine", "") for row in summary_rows if row.get("engine", "")}),
        "prompt_count": len({row.get("prompt_id", "") for row in summary_rows if row.get("prompt_id", "")}),
        "request_count": request_count,
        "successful_runs": successful_runs,
        "ttft_s_mean": mean([safe_float(row.get("ttft_s_mean")) for row in summary_rows]),
        "tpot_s_mean": mean([safe_float(row.get("tpot_s_mean")) for row in summary_rows]),
        "throughput_output_tok_s_mean": mean(throughput),
        "total_throughput_output_tok_s_mean": sum(value for value in throughput if value is not None),
        "goodput_sla_fraction_mean": mean([safe_float(row.get("goodput_sla_fraction")) for row in summary_rows]),
        "max_vmhwm_kb": max((safe_float(row.get("max_vmhwm_kb")) for row in summary_rows if safe_float(row.get("max_vmhwm_kb")) is not None), default=""),
    }
    return result


def global_metric_rows(summary_rows):
    return [aggregate_metric_set(summary_rows, "all_measurements")] if summary_rows else []


def per_model_metric_rows(summary_rows):
    rows = []
    for _, items in sorted(group_rows(summary_rows, ["model_label", "model"]).items()):
        first = items[0]
        row = aggregate_metric_set(items, "model", {
            "model_label": first.get("model_label", ""),
            "model": first.get("model", ""),
        })
        rows.append({field: row.get(field, "") for field in PER_MODEL_FIELDS})
    return rows


def node_scaling_metric_rows(scaling_rows):
    node_rows = [row for row in scaling_rows if row.get("node_count") or "node-scaling" in row.get("suite", "")]
    aggregate_fields = [
        "suite",
        "node_count",
        "replica_count",
        "engine",
        "model_label",
        "model",
        "threads",
        "threads_batch",
        "ctx_size",
        "parallel_requests",
        "cache_type_k",
        "cache_type_v",
        "max_tokens",
    ]
    rows = []
    for _, items in sorted(group_rows(node_rows, aggregate_fields).items()):
        first = items[0]
        row = {field: first.get(field, "") for field in NODE_SCALING_FIELDS}
        row.update({
            "prompt_count": len({item.get("prompt_id", "") for item in items if item.get("prompt_id", "")}),
            "successful_runs": sum(int(safe_float(item.get("successful_runs")) or 0) for item in items),
            "total_throughput_output_tok_s_mean": sum(safe_float(item.get("total_throughput_output_tok_s_mean")) or 0.0 for item in items),
            "ttft_s_mean_across_replicas": mean([safe_float(item.get("ttft_s_mean_across_replicas")) for item in items]),
            "tpot_s_mean_across_replicas": mean([safe_float(item.get("tpot_s_mean_across_replicas")) for item in items]),
            "goodput_sla_fraction_mean": mean([safe_float(item.get("goodput_sla_fraction_mean")) for item in items]),
            "max_vmhwm_kb": max((safe_float(item.get("max_vmhwm_kb")) for item in items if safe_float(item.get("max_vmhwm_kb")) is not None), default=""),
        })
        rows.append(row)
    return rows


def summarize_answers(answer_rows):
    groups = {}
    for row in answer_rows:
        key = (
            row["experiment_dir"],
            row["run_dir"],
            row["suite"],
            row["engine"],
            row["node_count"],
            row["node_rank"],
            row["node"],
            row["model"],
            row["model_label"],
            row["threads"],
            row["ctx_size"],
            row["parallel_requests"],
            row["cache_type_k"],
            row["cache_type_v"],
            row["max_tokens"],
            row["prompt_id"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, items in sorted(groups.items()):
        first = items[0]
        successful = [row for row in items if not row.get("error")]
        summary_rows.append({
            "experiment_dir": first["experiment_dir"],
            "run_dir": first["run_dir"],
            "suite": first["suite"],
            "engine": first["engine"],
            "node_count": first["node_count"],
            "node_rank": first["node_rank"],
            "node": first["node"],
            "model": first["model"],
            "model_label": first["model_label"],
            "threads": first["threads"],
            "ctx_size": first["ctx_size"],
            "parallel_requests": first["parallel_requests"],
            "cache_type_k": first["cache_type_k"],
            "cache_type_v": first["cache_type_v"],
            "max_tokens": first["max_tokens"],
            "prompt_id": first["prompt_id"],
            "category": first["category"],
            "mandatory": first["mandatory"],
            "prompt_text": first["prompt_text"],
            "runs": len(items),
            "successful_runs": len(successful),
            "error_runs": len(items) - len(successful),
            "first_success_answer": successful[0]["generated_text"] if successful else "",
        })
    return summary_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(metric_rows, answer_rows, out_dir, prompts=None, node_resource_rows=None):
    prompts = prompts or {}
    node_resource_rows = node_resource_rows or []
    mandatory_rows = [row for row in answer_rows if str(row.get("mandatory")).lower() == "true"]
    error_rows = [row for row in answer_rows if row.get("error")]
    prompt_rows = [prompts[key] for key in sorted(prompts)]
    summary_rows = summarize_metric_rows(metric_rows)
    scaling_rows = aggregate_scaling_rows(summary_rows)
    nonzero_metric_rows = [row for row in metric_rows if has_output_tokens(row)]
    nonzero_answer_rows = [row for row in answer_rows if has_output_tokens(row)]
    nonzero_summary_rows = summarize_metric_rows(nonzero_metric_rows)
    nonzero_scaling_rows = aggregate_scaling_rows(nonzero_summary_rows)

    write_csv(out_dir / "all_request_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(out_dir / "all_request_metrics_nonzero.csv", nonzero_metric_rows, METRIC_FIELDS)
    write_csv(out_dir / "prompt_answers.csv", answer_rows, ANSWER_FIELDS)
    write_csv(out_dir / "prompt_answers_nonzero.csv", nonzero_answer_rows, ANSWER_FIELDS)
    write_csv(out_dir / "mandatory_prompt_answers.csv", mandatory_rows, ANSWER_FIELDS)
    write_csv(out_dir / "request_errors.csv", error_rows, ANSWER_FIELDS)
    write_csv(out_dir / "answer_summary.csv", summarize_answers(answer_rows), SUMMARY_FIELDS)
    write_csv(out_dir / "prompt_catalog.csv", prompt_rows, PROMPT_FIELDS)
    write_csv(out_dir / "a1_summary.csv", summary_rows, PROMPT_SUMMARY_FIELDS)
    write_csv(out_dir / "a1_summary_nonzero.csv", nonzero_summary_rows, PROMPT_SUMMARY_FIELDS)
    write_csv(out_dir / "a1_scaling_summary.csv", scaling_rows, SCALING_SUMMARY_FIELDS)
    write_csv(out_dir / "a1_scaling_summary_nonzero.csv", nonzero_scaling_rows, SCALING_SUMMARY_FIELDS)
    write_csv(out_dir / "global_metrics.csv", global_metric_rows(summary_rows), GLOBAL_METRIC_FIELDS)
    write_csv(out_dir / "global_metrics_nonzero.csv", global_metric_rows(nonzero_summary_rows), GLOBAL_METRIC_FIELDS)
    write_csv(out_dir / "per_model_metrics.csv", per_model_metric_rows(summary_rows), PER_MODEL_FIELDS)
    write_csv(out_dir / "per_model_metrics_nonzero.csv", per_model_metric_rows(nonzero_summary_rows), PER_MODEL_FIELDS)
    write_csv(out_dir / "node_scaling_metrics.csv", node_scaling_metric_rows(scaling_rows), NODE_SCALING_FIELDS)
    write_csv(out_dir / "node_scaling_metrics_nonzero.csv", node_scaling_metric_rows(nonzero_scaling_rows), NODE_SCALING_FIELDS)
    write_csv(out_dir / "node_resource_metrics.csv", node_resource_rows, NODE_RESOURCE_FIELDS)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Measurement directories or requests.jsonl files")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/track_a_prompts.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("measurements/readable"))
    return parser


def main():
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    metric_rows, answer_rows = collect_rows(args.paths, args.prompts)
    node_resource_rows = collect_node_resource_rows(args.paths)
    write_outputs(metric_rows, answer_rows, args.out_dir, prompts, node_resource_rows)
    print(f"wrote {len(metric_rows)} request rows to {args.out_dir / 'all_request_metrics.csv'}")
    print(f"wrote {len(answer_rows)} prompt answer rows to {args.out_dir / 'prompt_answers.csv'}")
    print(f"wrote {len(node_resource_rows)} node resource rows to {args.out_dir / 'node_resource_metrics.csv'}")
    print(f"wrote readable CSV files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
