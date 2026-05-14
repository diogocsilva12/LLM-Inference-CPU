#!/usr/bin/env python3
"""Heuristic quality checks for the three mandatory Track A prompts.

The assignment requires recording the generated output for the mandatory
prompts. This helper adds a small, reproducible qualitative check without
calling an external judge model: it verifies that each answer contains the
minimum concepts expected for the prompt.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


MANDATORY_IDS = {
    "mandatory_short_capital_france",
    "mandatory_medium_ml_vs_dl",
    "mandatory_long_transformer_cpu",
}


def contains_any(text, terms):
    lower = text.lower()
    return any(term in lower for term in terms)


def evaluate_answer(prompt_id, generated_text):
    text = generated_text or ""
    if not text.strip():
        return {
            "quality_label": "missing",
            "quality_score": 0,
            "quality_max_score": 1,
            "quality_notes": "empty generated output",
        }

    checks = []
    if prompt_id == "mandatory_short_capital_france":
        checks = [
            ("mentions Paris", contains_any(text, ["paris"])),
        ]
    elif prompt_id == "mandatory_medium_ml_vs_dl":
        checks = [
            ("mentions machine learning", contains_any(text, ["machine learning", " ml "])),
            ("mentions deep learning", contains_any(text, ["deep learning", " dl "])),
            ("explains the relationship or distinction", contains_any(
                text,
                ["subset", "neural", "layers", "representation", "features", "data"],
            )),
            ("gives or signals examples", contains_any(
                text,
                ["example", "e.g.", "such as", "classification", "regression", "image", "speech"],
            )),
        ]
    elif prompt_id == "mandatory_long_transformer_cpu":
        checks = [
            ("mentions CPU inference challenge", contains_any(
                text,
                ["cpu", "memory", "bandwidth", "latency", "compute"],
            )),
            ("mentions model footprint or weights", contains_any(
                text,
                ["weights", "footprint", "parameters", "model size", "large model"],
            )),
            ("mentions quantisation", contains_any(
                text,
                ["quant", "int4", "int8", "q4", "bits", "lower precision"],
            )),
            ("explains quantisation benefit", contains_any(
                text,
                ["reduce", "smaller", "less memory", "bandwidth", "cache", "faster"],
            )),
        ]
    else:
        return {
            "quality_label": "unknown",
            "quality_score": None,
            "quality_max_score": None,
            "quality_notes": "not a mandatory prompt with a configured rubric",
        }

    score = sum(1 for _, passed in checks if passed)
    max_score = len(checks)
    if score == max_score:
        label = "good"
    elif score >= max_score - 1:
        label = "acceptable"
    else:
        label = "weak"
    missing = [name for name, passed in checks if not passed]
    notes = "all rubric checks passed" if not missing else "missing: " + "; ".join(missing)
    return {
        "quality_label": label,
        "quality_score": score,
        "quality_max_score": max_score,
        "quality_notes": notes,
    }


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


def evaluate_paths(paths):
    rows = []
    for request_file in iter_request_files(paths):
        run_dir = request_file.parent
        for row in read_jsonl(request_file):
            prompt_id = row.get("prompt_id")
            if prompt_id not in MANDATORY_IDS:
                continue
            quality = evaluate_answer(prompt_id, row.get("generated_text", ""))
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
                **quality,
                "generated_text": row.get("generated_text", ""),
            })
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["run_dir"], row["prompt_id"]), []).append(row)

    summary = []
    for (run_dir, prompt_id), items in sorted(groups.items()):
        scored = [r for r in items if isinstance(r.get("quality_score"), int)]
        scores = [r["quality_score"] / r["quality_max_score"] for r in scored if r.get("quality_max_score")]
        labels = [r["quality_label"] for r in items]
        summary.append({
            "run_dir": run_dir,
            "prompt_id": prompt_id,
            "runs": len(items),
            "good_or_acceptable_runs": sum(1 for label in labels if label in {"good", "acceptable"}),
            "quality_fraction_mean": statistics.mean(scores) if scores else None,
            "labels": ",".join(sorted(set(labels))),
            "representative_notes": next((r["quality_notes"] for r in items if r.get("quality_notes")), ""),
        })
    return summary


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
        "quality_label",
        "quality_score",
        "quality_max_score",
        "quality_notes",
        "generated_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="requests.jsonl files or directories to scan")
    parser.add_argument("--out", type=Path, required=True, help="CSV file for per-request quality rows")
    parser.add_argument("--summary-out", type=Path, help="JSON summary output")
    return parser


def main():
    args = build_parser().parse_args()
    rows = evaluate_paths(args.paths)
    write_csv(rows, args.out)
    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} mandatory quality rows to {args.out}")
    print(f"wrote quality summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
