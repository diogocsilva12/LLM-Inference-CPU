#!/usr/bin/env python3
"""Regression tests for mandatory prompt quality checks."""

import csv
import json
import tempfile
from pathlib import Path

import evaluate_mandatory_outputs as quality


def test_short_prompt_requires_paris():
    result = quality.evaluate_answer("mandatory_short_capital_france", "The capital of France is Paris.")
    assert result["quality_label"] == "good"
    assert result["quality_score"] == 1


def test_medium_prompt_flags_incomplete_answer():
    result = quality.evaluate_answer(
        "mandatory_medium_ml_vs_dl",
        "Machine learning uses data to make predictions.",
    )
    assert result["quality_label"] == "weak"
    assert result["quality_score"] < result["quality_max_score"] - 1


def test_directory_scan_writes_mandatory_rows():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        request_path = run_dir / "requests.jsonl"
        request_path.write_text(
            json.dumps({
                "prompt_id": "mandatory_short_capital_france",
                "category": "short",
                "trial": 1,
                "model": "/models/llama.gguf",
                "threads": 24,
                "ctx_size": 2048,
                "max_tokens": 128,
                "status": 200,
                "error": None,
                "generated_text": "Paris is the capital of France.",
            }) + "\n",
            encoding="utf-8",
        )
        rows = quality.evaluate_paths([Path(tmp)])
        out = Path(tmp) / "quality.csv"
        quality.write_csv(rows, out)
        with out.open(encoding="utf-8") as handle:
            written = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["quality_label"] == "good"
    assert written[0]["prompt_id"] == "mandatory_short_capital_france"


def main():
    test_short_prompt_requires_paris()
    test_medium_prompt_flags_incomplete_answer()
    test_directory_scan_writes_mandatory_rows()
    print("ok")


if __name__ == "__main__":
    main()
