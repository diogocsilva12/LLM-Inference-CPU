#!/usr/bin/env python3
"""Checks that the analysis notebook loads the readable CSV exports."""

import json
from pathlib import Path


def notebook_source():
    notebook = json.loads(Path("analysis_a1_benchmark.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_notebook_uses_readable_exports():
    source = notebook_source()
    assert "READABLE = MEASUREMENTS / 'readable'" in source
    assert "'global_metrics': READABLE / 'global_metrics.csv'" in source
    assert "'per_model_metrics': READABLE / 'per_model_metrics.csv'" in source
    assert "'node_scaling_metrics': READABLE / 'node_scaling_metrics.csv'" in source
    assert "'node_resource_metrics': READABLE / 'node_resource_metrics.csv'" in source
    assert "'prompt_answers': READABLE / 'prompt_answers.csv'" in source
    assert "'mandatory_prompt_answers': READABLE / 'mandatory_prompt_answers.csv'" in source
    assert "'answer_summary': READABLE / 'answer_summary.csv'" in source
    assert "'request_errors': READABLE / 'request_errors.csv'" in source
    assert "Global metrics overview" in source
    assert "Per-model metric breakdown" in source
    assert "Model-to-model percentage deltas" in source
    assert "Throughput delta vs baseline (%)" in source
    assert "TTFT delta vs baseline (%)" in source
    assert "TPOT delta vs baseline (%)" in source
    assert "Peak memory delta vs baseline (%)" in source
    assert "Multi-node scalability" in source
    assert "RPC coordinator" in source
    assert "one independent `llama-server` replica per allocated node" not in source
    assert "Scaling speedup and efficiency" in source
    assert "Throughput speedup vs 1 node" in source
    assert "Scaling efficiency vs 1 node" in source
    assert "Fixed node-count model comparison" in source
    assert "Resource monitoring overview" in source
    assert "node_resource_summary.csv" in source
    assert "Whole-node CPU and memory samples" in source
    assert "Node count" in source
    assert "Total throughput mean (output tokens/second)" in source
    assert "source_run" in source
    assert "organized_a1_results" not in source

    assert source.index("def normalize_node_scaling_table") < source.index("datasets['node_scaling_readable']")


def main():
    test_notebook_uses_readable_exports()
    print("ok")


if __name__ == "__main__":
    main()
