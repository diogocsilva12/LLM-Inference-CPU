#!/usr/bin/env python3
"""Small regression tests for the benchmark summarizer."""

from pathlib import Path

import summarize_a1_results as summarize


def test_experiment_dir_for_mtp_sweep_dir():
    run_dir = Path("/tmp/run/mtp/t_48_c_2048_p_1_k_f16_v_f16")
    assert summarize.experiment_dir_for(run_dir) == "/tmp/run"


def test_experiment_dir_for_server_sweep_suite_layout():
    run_dir = Path("/tmp/run/cache-sweep/tq/model-1/t_24_c_2048_p_1_n_128_k_f16_v_f16")
    assert summarize.experiment_dir_for(run_dir, {"suite": "cache-sweep"}) == "/tmp/run"


def main():
    test_experiment_dir_for_mtp_sweep_dir()
    test_experiment_dir_for_server_sweep_suite_layout()
    print("ok")


if __name__ == "__main__":
    main()
