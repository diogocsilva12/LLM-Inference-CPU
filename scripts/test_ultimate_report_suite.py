#!/usr/bin/env python3
"""Static checks for the complete report benchmark suite."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def assert_contains(text, needle, label):
    assert needle in text, f"missing {label}: {needle}"


def load_script(path):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_sweep_covers_assignment_dimensions():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert_contains(text, "#SBATCH --job-name=a1-server-sweep", "SLURM header")
    assert_contains(text, "normal-arm", "ARM partition")
    assert_contains(text, "model-1-mandatory", "mandatory model run")
    assert_contains(text, "model-2", "second model run")
    assert_contains(text, "model-3", "third model run")
    assert_contains(text, "SmolLM2-360M-Instruct.Q4_K_M.gguf", "SmolLM2 required model path")
    assert_contains(text, "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf", "Gemma required model path")
    assert_contains(text, "thread-scaling", "thread sweep")
    assert_contains(text, "concurrency", "concurrency sweep")
    assert_contains(text, "context-length", "context length sweep")
    assert_contains(text, "decode-length", "decode length sweep")
    assert_contains(text, "benchmark_openai_stream.py", "streaming benchmark client")
    assert_contains(text, "evaluate_mandatory_outputs.py", "mandatory answer quality")
    assert_contains(text, "a1_server_summary.csv", "timing summary")
    assert_contains(text, "mandatory_answer_quality.csv", "quality summary")
    assert "quantisation-levels" not in text
    assert "QUANT_MODEL" not in text


def test_report_artifact_script_exists():
    text = read("scripts/make_report_artifacts.py")
    assert_contains(text, "a1_all_results_summary.csv", "combined timing summary")
    assert_contains(text, "a1_all_results_scaling_summary.csv", "combined scaling summary")
    assert_contains(text, "throughput_vs_threads.svg", "thread plot")
    assert_contains(text, "throughput_vs_threads_nonzero.svg", "nonzero thread plot")
    assert_contains(text, "decode_throughput_vs_threads.svg", "decode throughput thread plot")
    assert_contains(text, "tpot_vs_threads.svg", "thread TPOT plot")
    assert_contains(text, "ttft_vs_threads.svg", "thread TTFT plot")
    assert_contains(text, "throughput_vs_concurrency.svg", "concurrency plot")
    assert_contains(text, "throughput_vs_concurrency_nonzero.svg", "nonzero concurrency plot")
    assert_contains(text, "decode_throughput_vs_concurrency.svg", "decode throughput concurrency plot")
    assert_contains(text, "tpot_vs_context.svg", "context TPOT plot")
    assert_contains(text, "throughput_vs_decode_length.svg", "decode length plot")
    assert_contains(text, "decode_throughput_vs_decode_length.svg", "decode throughput decode length plot")
    assert_contains(text, "goodput_by_engine.svg", "engine goodput plot")
    assert_contains(text, "throughput_by_cache.svg", "cache throughput plot")
    assert_contains(text, "ttft_vs_context.svg", "context plot")
    assert_contains(text, "memory_vs_model.svg", "memory plot")
    assert_contains(text, "predicted_vs_observed_tpot.svg", "model validation plot")
    assert_contains(text, "node_resource_summary.csv", "node resource summary")
    assert_contains(text, "node_cpu_busy_by_suite.svg", "node CPU resource plot")
    assert_contains(text, "node_memory_used_by_suite.svg", "node memory resource plot")
    assert_contains(text, "node_cpu_busy_vs_concurrency.svg", "node CPU concurrency plot")
    assert_contains(text, "node_memory_used_vs_node_count.svg", "node memory node-scaling plot")


def test_report_artifact_model_suite_parser():
    module = load_script("scripts/make_report_artifacts.py")
    assert module.suite_model_index("model-1-mandatory-thread-scaling") == 1.0
    assert module.suite_model_index("model-2-thread-scaling") == 2.0
    assert module.suite_model_index("model-3-thread-scaling") == 3.0
    assert module.suite_model_index("context-length") is None


def test_compare_script_supports_custom_build_dir():
    text = read("slurm/config/compare-mandatory-tq-vanilla.sh")
    assert_contains(text, "ENGINE_BUILD_DIR_NAME", "custom engine build dir variable")
    assert_contains(text, 'local build_dir="$repo_dir/$ENGINE_BUILD_DIR_NAME"', "custom build dir use")


def test_readme_documents_server_sweep():
    text = read("README.md")
    assert_contains(text, "Run The A1 Sweep", "run section")
    assert_contains(text, "sbatch --export=ALL", "sbatch export example")
    assert_contains(text, "slurm/sweep_benchmark/run-track-a1-server-sweep.sh", "server sweep command")
    assert_contains(text, "slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh", "x86 server sweep command")
    assert_contains(text, "slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh", "BLAS sweep command")
    assert_contains(text, "build-vanilla-openblas-fujitsu.sh", "OpenBLAS build command")
    assert_contains(text, "build-vanilla-blis.sh", "BLIS build command")
    assert_contains(text, "measurements/report_artifacts", "report artifact directory")
    assert_contains(text, "Required Model Set", "required model documentation")
    assert_contains(text, "mandatory_answer_quality.csv", "quality output")


def main():
    test_server_sweep_covers_assignment_dimensions()
    test_report_artifact_script_exists()
    test_report_artifact_model_suite_parser()
    test_compare_script_supports_custom_build_dir()
    test_readme_documents_server_sweep()
    print("ok")


if __name__ == "__main__":
    main()
