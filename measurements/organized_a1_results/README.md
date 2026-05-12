# Organized A1 Result Summary

Prompt-level rows: 1548
Scaling rows: 1548

## Selected complete runs

- ARM server sweep: `measurements/1253766-track-a1-server-sweep`
- x86 server sweep: `measurements/1254633-track-a1-server-sweep-x86`
- ARM BLAS sweep: `measurements/1254637-track-a1-vanilla-blas-sweep`

## Highest-signal findings

- Best Meta-Llama-3.1-8B baseline TPOT: `x86_server` / `vanilla` / `f16` at 122.7 ms.
- Slowest Meta-Llama-3.1-8B baseline TPOT: `x86_server` / `tq` / `turbo3` at 158.9 ms.
- The complete organized CSVs are in `measurements/organized_a1_results/tables/`.
- Human-readable CSVs with units are the files ending in `_readable.csv`.
- Metric definitions are in `measurements/organized_a1_results/METRICS_GUIDE.md`.
- Mandatory generated outputs and heuristic quality labels are in `measurements/organized_a1_results/combined/mandatory_quality.csv`.
- The LaTeX report is `measurements/organized_a1_results/report/track_a1_analysis.tex`.
- The performance-model table uses empirical effective bandwidth because no `memory_bandwidth.csv` STREAM file was found.
