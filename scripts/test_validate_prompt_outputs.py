#!/usr/bin/env python3
"""Regression tests for prompt-output validation with correctness + precision."""

import json
import tempfile
from pathlib import Path

import validate_prompt_outputs as validate


def write_request(path, prompt_id, generated_text):
    path.write_text(
        json.dumps({
            "prompt_id": prompt_id,
            "category": "short",
            "trial": 1,
            "model": "/models/llama.gguf",
            "threads": 24,
            "ctx_size": 2048,
            "max_tokens": 128,
            "status": 200,
            "error": None,
            "generated_text": generated_text,
        }) + "\n",
        encoding="utf-8",
    )


def test_reference_metrics_are_computed():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        request_path = run_dir / "requests.jsonl"
        write_request(request_path, "p1", "Paris is the capital of France.")

        rubric_path = Path(tmp) / "rubric.json"
        rubric_path.write_text(json.dumps({
            "p1": {
                "required_any": [["paris"], ["france"]],
                "reference": "Paris is the capital of France.",
            },
        }), encoding="utf-8")

        rubric = validate.load_rubric(rubric_path)
        rows = validate.evaluate_paths([Path(tmp)], rubric, min_reference_f1=0.6)

    row = rows[0]
    assert row["correctness_label"] == "good"
    assert row["reference_precision"] == 1.0
    assert row["reference_recall"] == 1.0
    assert row["reference_f1"] == 1.0
    assert row["validation_pass"] is True


def test_missing_required_concepts_is_weak():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        request_path = run_dir / "requests.jsonl"
        write_request(request_path, "p2", "France has many beautiful cities.")

        rubric_path = Path(tmp) / "rubric.json"
        rubric_path.write_text(json.dumps({
            "p2": {
                "required_any": [["paris"], ["capital"], ["france"]],
            },
        }), encoding="utf-8")

        rubric = validate.load_rubric(rubric_path)
        rows = validate.evaluate_paths([Path(tmp)], rubric, min_reference_f1=0.6)

    row = rows[0]
    assert row["correctness_label"] == "weak"
    assert row["validation_pass"] is False


def test_builtin_mandatory_rubric_works_without_external_rubric():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        request_path = run_dir / "requests.jsonl"
        write_request(request_path, "mandatory_short_capital_france", "Paris is the capital of France.")

        rows = validate.evaluate_paths([Path(tmp)], {}, min_reference_f1=0.6)

    row = rows[0]
    assert row["correctness_label"] == "good"
    assert row["validation_pass"] is True


def test_reference_threshold_is_not_enforced_unless_requested():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        request_path = run_dir / "requests.jsonl"
        write_request(request_path, "mandatory_short_capital_france", "Paris")

        rows_default = validate.evaluate_paths([Path(tmp)], {}, min_reference_f1=0.9)
        rows_strict = validate.evaluate_paths([Path(tmp)], {}, min_reference_f1=0.9, require_reference_f1=True)

    assert rows_default[0]["validation_pass"] is True
    assert rows_strict[0]["validation_pass"] is False


def main():
    test_reference_metrics_are_computed()
    test_missing_required_concepts_is_weak()
    test_builtin_mandatory_rubric_works_without_external_rubric()
    test_reference_threshold_is_not_enforced_unless_requested()
    print("ok")


if __name__ == "__main__":
    main()
