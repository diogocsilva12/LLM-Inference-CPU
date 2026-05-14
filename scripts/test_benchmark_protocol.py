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


def test_limit_per_category_always_keeps_mandatory_prompts(tmp_path):
    prompts = []
    for category in ("short", "medium", "long"):
        for idx in range(5):
            prompts.append({
                "id": f"{category}_nonmandatory_{idx}",
                "category": category,
                "mandatory": False,
                "text": "non mandatory",
            })
        prompts.append({
            "id": f"mandatory_{category}",
            "category": category,
            "mandatory": True,
            "text": "mandatory",
        })
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")

    _, selected = bench.load_prompts(
        prompt_file,
        {"short", "medium", "long"},
        limit_per_category=5,
        mandatory_only=False,
    )
    selected_ids = {prompt["id"] for prompt in selected}

    assert len(selected) == 15
    assert {"mandatory_short", "mandatory_medium", "mandatory_long"}.issubset(selected_ids)


def test_server_track_a1_sweep_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    suite = (ROOT / "slurm/sweep_benchmark/run-track-a1-server-sweep.sh").read_text(encoding="utf-8")
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
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"' in suite
    assert "five prompts per category" in readme
    assert "TurboQuant and vanilla" in readme
    assert "ENGINE_CACHE_SWEEPS" in suite
    assert "vanilla only `f16:f16`" in readme
    assert "TurboQuant with `turbo3:turbo3` and `turbo4:turbo4`" in readme


def test_server_sweep_runs_fifteen_prompt_protocol_with_warmup():
    text = (ROOT / "slurm/sweep_benchmark/run-track-a1-server-sweep.sh").read_text(encoding="utf-8")
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-0}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"' in text
    assert 'WARMUP_TRIALS="${WARMUP_TRIALS:-1}"' in text
    assert "mandatory_answer_quality.csv" in text
    assert "evaluate_mandatory_outputs.py" in text


def test_server_sweep_can_submit_derived_array_and_skip_known_bad_configs():
    text = (ROOT / "slurm/sweep_benchmark/run-track-a1-server-sweep.sh").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'SUBMIT_ARRAY="${SUBMIT_ARRAY:-0}"' in text
    assert 'ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"' in text
    assert "sbatch --array=\"0-$last_index%$ARRAY_CONCURRENCY\"" in text
    assert 'EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-tq:model-2:*}"' in text
    assert "config_is_excluded" in text
    assert "SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8" in readme


def test_node_scaling_sweep_runs_fifteen_prompt_protocol():
    text = (ROOT / "slurm/sweep_benchmark/run-track-a1-node-scaling.sh").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'NODE_COUNTS="${NODE_COUNTS:-1 2 4 6 8 16 24 32 64 128 256}"' in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"' in text
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-0}"' in text
    assert "rpc_endpoints" in text
    assert "rpc-server" in text
    assert "--rpc" in text
    assert "node-scaling-n" in text
    assert "a1_node_scaling_summary.csv" in text
    assert "a1_node_scaling_scaling_summary.csv" in text
    assert "1, 2, 4, 6, 8, 16, 24, 32, 64, 128, and 256 nodes" in readme


def main():
    test_prompt_file_has_ten_per_category_and_mandatory_prompts()
    test_benchmark_client_supports_warmup_and_seed()
    test_limit_per_category_always_keeps_mandatory_prompts(Path("/tmp"))
    test_server_track_a1_sweep_is_documented()
    test_server_sweep_runs_fifteen_prompt_protocol_with_warmup()
    test_server_sweep_can_submit_derived_array_and_skip_known_bad_configs()
    test_node_scaling_sweep_runs_fifteen_prompt_protocol()
    print("ok")


if __name__ == "__main__":
    main()
