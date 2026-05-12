#!/usr/bin/env python3
"""Regression tests for the llama.cpp CLI benchmark runner."""

import json
import tempfile
from pathlib import Path

import benchmark_llamacpp_cli as cli


def test_parse_llamacpp_timings_from_stderr():
    stderr = """
llama_perf_context_print: prompt eval time =    124.50 ms /    17 tokens (    7.32 ms per token,   136.55 tokens per second)
llama_perf_context_print:        eval time =    512.00 ms /    64 runs   (    8.00 ms per token,   125.00 tokens per second)
"""
    timings = cli.parse_llamacpp_timings(stderr)
    assert timings["prompt_tokens"] == 17
    assert timings["prompt_eval_s"] == 0.1245
    assert timings["eval_tokens"] == 64
    assert timings["eval_s"] == 0.512
    assert timings["prompt_tokens_per_s"] == 136.55
    assert timings["eval_tokens_per_s"] == 125.0


def test_write_summary_preserves_generated_output_for_mandatory_prompt():
    rows = [
        {
            "prompt_id": "mandatory_short_capital_france",
            "category": "short",
            "mandatory": True,
            "trial": 1,
            "error": None,
            "ttft_s": 0.25,
            "tpot_s": 0.05,
            "throughput_output_tok_s": 20.0,
            "generated_text": "Paris is the capital of France.",
        },
        {
            "prompt_id": "mandatory_short_capital_france",
            "category": "short",
            "mandatory": True,
            "trial": 2,
            "error": None,
            "ttft_s": 0.35,
            "tpot_s": 0.07,
            "throughput_output_tok_s": 15.0,
            "generated_text": "The capital of France is Paris.",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        cli.write_summary(rows, summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary[0]["prompt_id"] == "mandatory_short_capital_france"
    assert summary[0]["ttft_s_mean"] == 0.3
    assert summary[0]["representative_generated_text"] == "Paris is the capital of France."


def test_cli_parser_accepts_assignment_controls():
    parser = cli.build_parser()
    args = parser.parse_args([
        "--llama-cli",
        "/tmp/llama-cli",
        "--model",
        "/tmp/model.gguf",
        "--out-dir",
        "/tmp/out",
        "--threads",
        "24",
        "--ctx-size",
        "2048",
        "--max-tokens",
        "128",
        "--warmup-trials",
        "1",
        "--trials",
        "3",
        "--mandatory-only",
    ])
    assert args.threads == 24
    assert args.ctx_size == 2048
    assert args.max_tokens == 128
    assert args.warmup_trials == 1
    assert args.trials == 3
    assert args.mandatory_only is True


def main():
    test_parse_llamacpp_timings_from_stderr()
    test_write_summary_preserves_generated_output_for_mandatory_prompt()
    test_cli_parser_accepts_assignment_controls()
    print("ok")


if __name__ == "__main__":
    main()
