#!/usr/bin/env python3
"""Validate prompt outputs with rubric correctness checks and reference precision."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


DEFAULT_RUBRIC = {
    "mandatory_short_capital_france": {
        "required_any": [["paris"]],
        "reference": "Paris is the capital of France.",
    },
    "mandatory_medium_ml_vs_dl": {
        "required_any": [
            ["machine learning", " ml "],
            ["deep learning", " dl "],
            ["subset", "neural", "layers", "representation", "features", "data"],
            ["example", "e.g.", "such as", "classification", "regression", "image", "speech"],
        ],
    },
    "mandatory_long_transformer_cpu": {
        "required_any": [
            ["cpu", "memory", "bandwidth", "latency", "compute"],
            ["weights", "footprint", "parameters", "model size", "large model"],
            ["quant", "int4", "int8", "q4", "bits", "lower precision"],
            ["reduce", "smaller", "less memory", "bandwidth", "cache", "faster"],
        ],
    },
}


def contains_any(text, terms):
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def compute_reference_metrics(generated_text, reference_text):
    if not reference_text:
        return {
            "reference_exact_match": None,
            "reference_precision": None,
            "reference_recall": None,
            "reference_f1": None,
        }

    generated_tokens = tokenize(generated_text)
    reference_tokens = tokenize(reference_text)
    generated_count = Counter(generated_tokens)
    reference_count = Counter(reference_tokens)
    overlap = sum((generated_count & reference_count).values())

    precision = overlap / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "reference_exact_match": generated_tokens == reference_tokens,
        "reference_precision": precision,
        "reference_recall": recall,
        "reference_f1": f1,
    }


def evaluate_correctness(generated_text, rubric_entry):
    text = generated_text or ""
    required = rubric_entry.get("required_any", [])
    max_score = len(required) if required else 1
    if not text.strip():
        return {
            "correctness_label": "missing",
            "correctness_score": 0,
            "correctness_max_score": max_score,
            "correctness_fraction": 0.0,
            "correctness_notes": "empty generated output",
        }
    if not required:
        return {
            "correctness_label": "unscored",
            "correctness_score": None,
            "correctness_max_score": None,
            "correctness_fraction": None,
            "correctness_notes": "rubric has no required_any checks",
        }

    passes = [contains_any(text, terms) for terms in required]
    score = sum(1 for passed in passes if passed)
    if score == max_score:
        label = "good"
    elif score >= max_score - 1:
        label = "acceptable"
    else:
        label = "weak"
    missing_groups = ["/".join(required[i]) for i, passed in enumerate(passes) if not passed]
    notes = "all required checks passed" if not missing_groups else "missing: " + "; ".join(missing_groups)
    return {
        "correctness_label": label,
        "correctness_score": score,
        "correctness_max_score": max_score,
        "correctness_fraction": score / max_score if max_score else None,
        "correctness_notes": notes,
    }


def load_rubric(path):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("prompts"), dict):
        return data["prompts"]
    if isinstance(data, dict):
        return data
    raise ValueError("rubric file must be a JSON object")


def iter_request_files(paths):
    for path in paths:
        if path.is_file():
            if path.name == "requests.jsonl":
                yield path
            continue
        if path.is_dir():
            yield from sorted(path.rglob("requests.jsonl"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def evaluate_paths(paths, rubric, min_reference_f1, require_reference_f1=False):
    merged_rubric = dict(DEFAULT_RUBRIC)
    merged_rubric.update(rubric)

    rows = []
    for request_file in iter_request_files(paths):
        run_dir = request_file.parent
        for row in read_jsonl(request_file):
            prompt_id = row.get("prompt_id")
            rubric_entry = merged_rubric.get(prompt_id)
            if rubric_entry:
                correctness = evaluate_correctness(row.get("generated_text", ""), rubric_entry)
                reference = compute_reference_metrics(
                    row.get("generated_text", ""),
                    rubric_entry.get("reference"),
                )
            else:
                correctness = {
                    "correctness_label": "unscored",
                    "correctness_score": None,
                    "correctness_max_score": None,
                    "correctness_fraction": None,
                    "correctness_notes": "no rubric entry for prompt_id",
                }
                reference = {
                    "reference_exact_match": None,
                    "reference_precision": None,
                    "reference_recall": None,
                    "reference_f1": None,
                }

            label = correctness["correctness_label"]
            correctness_ok = label in {"good", "acceptable", "unscored"}
            reference_f1 = reference["reference_f1"]
            if require_reference_f1 and reference_f1 is not None:
                reference_ok = reference_f1 >= min_reference_f1
            else:
                reference_ok = True
            scored = label != "unscored" or reference_f1 is not None
            validation_pass = (correctness_ok and reference_ok) if scored else None
            validation_label = "unscored" if validation_pass is None else ("pass" if validation_pass else "fail")

            rows.append({
                "run_dir": str(run_dir),
                "prompt_id": prompt_id,
                "category": row.get("category"),
                "trial": row.get("trial"),
                "model": row.get("model"),
                "threads": row.get("threads"),
                "ctx_size": row.get("ctx_size"),
                "max_tokens": row.get("max_tokens"),
                "status": row.get("status"),
                "error": row.get("error"),
                "validation_label": validation_label,
                "validation_pass": validation_pass,
                "min_reference_f1": min_reference_f1,
                **correctness,
                **reference,
                "generated_text": row.get("generated_text", ""),
            })
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["run_dir"], row["prompt_id"]), []).append(row)

    summary_rows = []
    for (run_dir, prompt_id), items in sorted(groups.items()):
        correctness_scores = [
            row["correctness_fraction"]
            for row in items
            if isinstance(row.get("correctness_fraction"), (int, float))
        ]
        reference_f1_scores = [
            row["reference_f1"]
            for row in items
            if isinstance(row.get("reference_f1"), (int, float))
        ]
        pass_runs = sum(1 for row in items if row.get("validation_pass") is True)
        fail_runs = sum(1 for row in items if row.get("validation_pass") is False)
        summary_rows.append({
            "run_dir": run_dir,
            "prompt_id": prompt_id,
            "runs": len(items),
            "pass_runs": pass_runs,
            "fail_runs": fail_runs,
            "validation_rate": pass_runs / (pass_runs + fail_runs) if (pass_runs + fail_runs) else None,
            "correctness_fraction_mean": statistics.mean(correctness_scores) if correctness_scores else None,
            "reference_f1_mean": statistics.mean(reference_f1_scores) if reference_f1_scores else None,
            "labels": ",".join(sorted(set(row["validation_label"] for row in items))),
            "representative_notes": next((row["correctness_notes"] for row in items if row.get("correctness_notes")), ""),
        })
    return summary_rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "prompt_id",
        "category",
        "trial",
        "model",
        "threads",
        "ctx_size",
        "max_tokens",
        "status",
        "error",
        "validation_label",
        "validation_pass",
        "min_reference_f1",
        "correctness_label",
        "correctness_score",
        "correctness_max_score",
        "correctness_fraction",
        "correctness_notes",
        "reference_exact_match",
        "reference_precision",
        "reference_recall",
        "reference_f1",
        "generated_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="requests.jsonl files or directories to scan")
    parser.add_argument("--out", type=Path, required=True, help="CSV file for per-request validation rows")
    parser.add_argument("--summary-out", type=Path, help="JSON summary output")
    parser.add_argument("--rubric", type=Path, help="JSON rubric with prompt_id keys and required_any/reference rules")
    parser.add_argument("--min-reference-f1", type=float, default=0.35, help="Minimum reference F1 to count as pass")
    parser.add_argument(
        "--require-reference-f1",
        action="store_true",
        help="Require reference F1 >= --min-reference-f1 for pass/fail (off by default).",
    )
    parser.add_argument("--scored-only", action="store_true", help="Drop rows that have no rubric or reference checks")
    return parser


def main():
    args = build_parser().parse_args()
    rubric = load_rubric(args.rubric)
    rows = evaluate_paths(args.paths, rubric, args.min_reference_f1, args.require_reference_f1)
    if args.scored_only:
        rows = [row for row in rows if row.get("validation_label") != "unscored"]
    write_csv(rows, args.out)
    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} validation rows to {args.out}")
    print(f"wrote validation summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
