#!/usr/bin/env python3
"""Behavioral tests for bottleneck model analysis."""

import csv
import tempfile
from pathlib import Path

import analyze_bottleneck_model as analysis


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_predicted_tpot_uses_model_size_and_median_bandwidth():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "model.gguf"
        model.write_bytes(b"0" * 1000)

        write_csv(
            root / "arm" / "memory-bandwidth" / "memory_bandwidth.csv",
            ["trial", "seconds", "triad_gb_s"],
            [
                {"trial": "1", "seconds": "1.0", "triad_gb_s": "10"},
                {"trial": "2", "seconds": "1.0", "triad_gb_s": "20"},
                {"trial": "3", "seconds": "1.0", "triad_gb_s": "30"},
            ],
        )
        write_csv(
            root / "arm" / "benchmark" / "scaling_summary.csv",
            [
                "experiment_dir",
                "engine",
                "replica_count",
                "model",
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
            ],
            [
                {
                    "experiment_dir": str(root),
                    "engine": "vanilla",
                    "replica_count": "1",
                    "model": str(model),
                    "threads": "48",
                    "threads_batch": "48",
                    "ctx_size": "2048",
                    "parallel_requests": "1",
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                    "max_tokens": "128",
                    "category": "short",
                    "prompt_id": "p",
                    "successful_runs": "3",
                    "total_throughput_output_tok_s_mean": "5",
                    "ttft_s_mean_across_replicas": "1",
                    "tpot_s_mean_across_replicas": "0.0000001",
                    "goodput_sla_fraction_mean": "1",
                    "max_vmhwm_kb": "1000",
                }
            ],
        )

        rows = analysis.build_rows(root)
        assert len(rows) == 1
        assert rows[0]["bw_mem_gb_s_median"] == 20.0
        assert rows[0]["tpot_pred_s"] == 1000 / (20.0 * 1.0e9)
        assert rows[0]["observed_over_predicted"] == 2.0


def main():
    test_predicted_tpot_uses_model_size_and_median_bandwidth()
    print("ok")


if __name__ == "__main__":
    main()
