#!/usr/bin/env python3
"""Small regression tests for the benchmark summarizer."""

from pathlib import Path
import json

import summarize_a1_results as summarize


def test_experiment_dir_for_mtp_sweep_dir():
    run_dir = Path("/tmp/run/mtp/t_48_c_2048_p_1_k_f16_v_f16")
    assert summarize.experiment_dir_for(run_dir) == "/tmp/run"


def test_experiment_dir_for_server_sweep_suite_layout():
    run_dir = Path("/tmp/run/cache-sweep/tq/model-1/t_24_c_2048_p_1_n_128_k_f16_v_f16")
    assert summarize.experiment_dir_for(run_dir, {"suite": "cache-sweep"}) == "/tmp/run"


def test_node_scaling_summary_keeps_node_count(tmp_path):
    run_dir = tmp_path / "node-scaling" / "node_count_002" / "node_0_alpha" / "t_46_c_8192_p_2_n_128_k_f16_v_f16"
    run_dir.mkdir(parents=True)
    (run_dir / "system.txt").write_text(
        "\n".join([
            "suite=node-scaling-n2",
            "engine=llamacpp-optimal",
            "rank=0",
            "node=alpha",
            "node_count=2",
            "model=/models/model.gguf",
            "model_label=model-1-mandatory",
            "threads=46",
            "threads_batch=46",
            "ctx_size=8192",
            "parallel_requests=2",
            "bench_concurrency=2",
            "cache_type_k=f16",
            "cache_type_v=f16",
            "max_tokens=128",
            "trials=1",
            "warmup_trials=1",
            "mandatory_only=0",
            "limit_per_category=3",
        ]) + "\n",
        encoding="utf-8",
    )
    row = {
        "prompt_id": "mandatory_short_capital_france",
        "category": "short",
        "trial": 1,
        "error": None,
        "ttft_s": 0.5,
        "tpot_s": 0.05,
        "throughput_output_tok_s": 20.0,
    }
    (run_dir / "requests.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = summarize.summarize_run(run_dir, 2.0, 0.2)
    assert rows[0]["node_count"] == "2"
    aggregate = summarize.aggregate_scaling(rows)
    assert aggregate[0]["node_count"] == "2"
    assert aggregate[0]["replica_count"] == 2


def main():
    test_experiment_dir_for_mtp_sweep_dir()
    test_experiment_dir_for_server_sweep_suite_layout()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_node_scaling_summary_keeps_node_count(Path(directory))
    print("ok")


if __name__ == "__main__":
    main()
