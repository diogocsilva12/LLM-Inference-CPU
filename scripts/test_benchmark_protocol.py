#!/usr/bin/env python3
"""Checks for the assignment benchmark protocol."""

import json
from pathlib import Path

import benchmark_openai_stream as bench


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_file_has_ten_per_category_and_mandatory_prompts():
    data = json.loads((ROOT / "prompts/track_a_prompts.json").read_text(encoding="utf-8"))
    prompts = data["prompts"]
    counts = {}
    mandatory_ids = set()
    for prompt in prompts:
        counts[prompt["category"]] = counts.get(prompt["category"], 0) + 1
        if prompt.get("mandatory"):
            mandatory_ids.add(prompt["id"])

    assert counts.get("short", 0) >= 10
    assert counts.get("medium", 0) >= 10
    assert counts.get("long", 0) >= 10
    assert {
        "mandatory_short_capital_france",
        "mandatory_medium_ml_vs_dl",
        "mandatory_long_transformer_cpu",
    }.issubset(mandatory_ids)


def test_benchmark_client_supports_warmup_and_seed():
    parser = bench.build_parser()
    args = parser.parse_args(["--out", "/tmp/out.jsonl", "--warmup-trials", "1", "--seed", "42"])
    assert args.warmup_trials == 1
    assert args.seed == 42


def test_server_track_a1_sweep_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    suite = (ROOT / "slurm/run-track-a1-server-sweep.sh").read_text(encoding="utf-8")
    assert "Track A1 llama.cpp Server Sweep" in readme
    assert "scripts/benchmark_openai_stream.py" in readme
    assert "scripts/evaluate_mandatory_outputs.py" in readme
    assert "requests.jsonl" in readme
    assert "generated_text" in readme
    assert "MODEL_SPECS" in suite
    assert "thread-scaling" in suite
    assert "concurrency" in suite
    assert "context-length" in suite
    assert "decode-length" in suite
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-0}"' in suite
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-3}"' in suite
    assert "three mandatory prompts plus six additional prompts" in readme
    assert "TurboQuant and vanilla" in readme
    assert "ENGINE_CACHE_SWEEPS" in suite
    assert "vanilla only `f16:f16`" in readme
    assert "TurboQuant with `turbo3:turbo3` and `turbo4:turbo4`" in readme


def test_server_sweep_runs_nine_prompt_protocol_with_warmup():
    text = (ROOT / "slurm/run-track-a1-server-sweep.sh").read_text(encoding="utf-8")
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-0}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-3}"' in text
    assert 'WARMUP_TRIALS="${WARMUP_TRIALS:-1}"' in text
    assert "mandatory_answer_quality.csv" in text
    assert "evaluate_mandatory_outputs.py" in text


def main():
    test_prompt_file_has_ten_per_category_and_mandatory_prompts()
    test_benchmark_client_supports_warmup_and_seed()
    test_server_track_a1_sweep_is_documented()
    test_server_sweep_runs_nine_prompt_protocol_with_warmup()
    print("ok")


if __name__ == "__main__":
    main()
