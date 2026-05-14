#!/usr/bin/env python3
"""Regression tests for readable Track A1 CSV extraction."""

import csv
import json
import tempfile
from pathlib import Path

import parse_readable_a1_outputs as readable


def test_extracts_prompt_answers_with_run_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prompts_path = root / "prompts.json"
        prompts_path.write_text(
            json.dumps({
                "prompts": [
                    {
                        "id": "mandatory_short_capital_france",
                        "category": "short",
                        "mandatory": True,
                        "text": "What is the capital of France?",
                    }
                ]
            }),
            encoding="utf-8",
        )

        run_dir = root / "measurements" / "123-track" / "node-scaling-n2" / "node_count_2" / "node_0_alpha" / "t_24_c_2048_p_1_n_128_k_f16_v_f16"
        run_dir.mkdir(parents=True)
        (run_dir / "system.txt").write_text(
            "\n".join([
                "suite=node-scaling-n2",
                "engine=vanilla",
                "rank=0",
                "node=alpha",
                "node_count=2",
                "model_label=model-1",
                "model=/models/model-1.gguf",
                "threads=24",
                "ctx_size=2048",
                "parallel_requests=1",
                "bench_concurrency=1",
                "cache_type_k=f16",
                "cache_type_v=f16",
                "max_tokens=128",
                "trials=1",
                "warmup_trials=0",
                "mandatory_only=0",
                "limit_per_category=5",
            ]) + "\n",
            encoding="utf-8",
        )
        (run_dir / "requests.jsonl").write_text(
            "\n".join([
                json.dumps({
                    "prompt_id": "mandatory_short_capital_france",
                    "category": "short",
                    "mandatory": True,
                    "trial": 1,
                    "status": 200,
                    "error": None,
                    "ttft_s": 0.1,
                    "tpot_s": 0.2,
                    "total_s": 0.5,
                    "output_token_events": 4,
                    "throughput_output_tok_s": 8.0,
                    "generated_text": "Paris\nis the capital.",
                }),
                json.dumps({
                    "prompt_id": "mandatory_short_capital_france",
                    "category": "short",
                    "mandatory": True,
                    "trial": 2,
                    "status": 200,
                    "error": None,
                    "ttft_s": 0.2,
                    "tpot_s": None,
                    "total_s": 1.0,
                    "output_token_events": 0,
                    "throughput_output_tok_s": 0.0,
                    "generated_text": "",
                }),
            ]) + "\n",
            encoding="utf-8",
        )
        (run_dir / "node_resources.csv").write_text(
            "\n".join([
                "timestamp,role,cpu_user_jiffies,cpu_nice_jiffies,cpu_system_jiffies,cpu_idle_jiffies,cpu_iowait_jiffies,cpu_irq_jiffies,cpu_softirq_jiffies,cpu_steal_jiffies,mem_total_kb,mem_available_kb,mem_used_kb,mem_free_kb,swap_total_kb,swap_free_kb",
                "2026-05-13T12:00:00+00:00,server,100,0,25,1000,1,0,2,0,32768000,20000000,12768000,19000000,0,0",
            ]) + "\n",
            encoding="utf-8",
        )

        rows, prompt_rows = readable.collect_rows([root / "measurements"], prompts_path)
        node_resource_rows = readable.collect_node_resource_rows([root / "measurements"])
        out_dir = root / "readable"
        readable.write_outputs(rows, prompt_rows, out_dir, node_resource_rows=node_resource_rows)

        with (out_dir / "prompt_answers.csv").open(encoding="utf-8") as handle:
            answers = list(csv.DictReader(handle))
        with (out_dir / "mandatory_prompt_answers.csv").open(encoding="utf-8") as handle:
            mandatory = list(csv.DictReader(handle))
        with (out_dir / "answer_summary.csv").open(encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        with (out_dir / "all_request_metrics.csv").open(encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        with (out_dir / "a1_summary.csv").open(encoding="utf-8") as handle:
            prompt_summary = list(csv.DictReader(handle))
        with (out_dir / "a1_scaling_summary.csv").open(encoding="utf-8") as handle:
            scaling_summary = list(csv.DictReader(handle))
        with (out_dir / "global_metrics.csv").open(encoding="utf-8") as handle:
            global_metrics = list(csv.DictReader(handle))
        with (out_dir / "per_model_metrics.csv").open(encoding="utf-8") as handle:
            per_model_metrics = list(csv.DictReader(handle))
        with (out_dir / "node_scaling_metrics.csv").open(encoding="utf-8") as handle:
            node_scaling_metrics = list(csv.DictReader(handle))
        with (out_dir / "node_resource_metrics.csv").open(encoding="utf-8") as handle:
            node_resource_metrics = list(csv.DictReader(handle))
        with (out_dir / "all_request_metrics_nonzero.csv").open(encoding="utf-8") as handle:
            nonzero_metrics = list(csv.DictReader(handle))
        with (out_dir / "a1_summary_nonzero.csv").open(encoding="utf-8") as handle:
            nonzero_prompt_summary = list(csv.DictReader(handle))
        with (out_dir / "global_metrics_nonzero.csv").open(encoding="utf-8") as handle:
            nonzero_global_metrics = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert answers[0]["engine"] == "vanilla"
    assert answers[0]["node_count"] == "2"
    assert answers[0]["node_rank"] == "0"
    assert answers[0]["prompt_text"] == "What is the capital of France?"
    assert answers[0]["generated_text"] == "Paris\\nis the capital."
    assert metrics[0]["node_count"] == "2"
    assert len(mandatory) == 2
    assert summary[0]["successful_runs"] == "2"
    assert summary[0]["first_success_answer"] == "Paris\\nis the capital."
    assert prompt_summary[0]["node_count"] == "2"
    assert scaling_summary[0]["node_count"] == "2"
    assert scaling_summary[0]["replica_count"] == "2"
    assert global_metrics[0]["configuration_count"] == "1"
    assert global_metrics[0]["prompt_count"] == "1"
    assert per_model_metrics[0]["model_label"] == "model-1"
    assert node_scaling_metrics[0]["node_count"] == "2"
    assert node_resource_metrics[0]["role"] == "server"
    assert node_resource_metrics[0]["mem_total_kb"] == "32768000"
    assert node_resource_metrics[0]["mem_used_kb"] == "12768000"
    assert len(nonzero_metrics) == 1
    assert nonzero_metrics[0]["output_token_events"] == "4"
    assert nonzero_prompt_summary[0]["successful_runs"] == "1"
    assert nonzero_global_metrics[0]["request_count"] == "1"


def main():
    test_extracts_prompt_answers_with_run_metadata()
    print("ok")


if __name__ == "__main__":
    main()
