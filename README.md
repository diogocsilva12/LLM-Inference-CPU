# LLM Inference on CPU

![Project](https://img.shields.io/badge/project-LLM%20CPU%20Inference-blue)
![Focus](https://img.shields.io/badge/focus-llama.cpp%20%7C%20TurboQuant%20%7C%20CPU%20Benchmarking-informational)
![HPC](https://img.shields.io/badge/platform-Deucalion%20HPC-lightgrey)
![Grade](https://img.shields.io/badge/final%20grade-20%2F20-success)

Research and benchmarking project focused on studying, testing, and optimizing **Large Language Model inference on CPU-based systems**.

The project evaluates how practical CPU inference can be for modern LLM workloads, with particular attention to `llama.cpp`, TurboQuant-style cache/quantization experiments, GGUF models, CPU thread scaling, context-size sensitivity, request concurrency, BLAS variants, and distributed/RPC execution on HPC nodes.

**Final grade: 20/20**

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Research Objectives](#research-objectives)
- [Repository Structure](#repository-structure)
- [Benchmarking Methodology](#benchmarking-methodology)
- [What Is Measured](#what-is-measured)
- [Inference Engines and Variants](#inference-engines-and-variants)
- [Quantization and Cache Experiments](#quantization-and-cache-experiments)
- [One-Time Setup](#one-time-setup)
- [Model Configuration](#model-configuration)
- [Running the Main ARM Sweep](#running-the-main-arm-sweep)
- [Running Follow-Up Experiments](#running-follow-up-experiments)
- [RPC Node Scaling](#rpc-node-scaling)
- [Outputs](#outputs)
- [Post-Processing](#post-processing)
- [Validation and Tests](#validation-and-tests)
- [Technologies](#technologies)
- [Learning Outcomes](#learning-outcomes)
- [Future Work](#future-work)
- [Codebase Hygiene](#codebase-hygiene)

---

## Overview

This repository contains an experimental harness for evaluating **LLM inference on CPU systems**, with a strong focus on reproducible benchmarking in an HPC environment.

The project investigates the behaviour of local LLM inference beyond GPU-centric deployment. It compares multiple engine paths, runtime configurations, cache formats, and scaling strategies using a common OpenAI-compatible streaming benchmark protocol.

The central benchmark flow is the **Track A1 llama.cpp Server Sweep**, which evaluates:

- vanilla `llama.cpp`;
- TurboQuant-enabled `llama.cpp`;
- selected BLAS builds;
- ARM versus x86 execution wrappers;
- CPU thread scaling;
- request concurrency;
- context-size scaling;
- decode-length scaling;
- optional RPC-based node scaling.

The main SLURM entrypoints are located under:

```text
slurm/sweep_benchmark/
```

The Python utilities for benchmarking, validation, result summarisation, parsing, and report artifact generation are located under:

```text
scripts/
```

---

## Motivation

Large Language Model inference is usually associated with GPU acceleration. However, CPU-based inference remains relevant in many realistic scenarios:

- private local AI execution;
- low-cost inference environments;
- educational and research contexts;
- edge and resource-constrained systems;
- HPC systems where CPU resources are widely available;
- model serving where GPUs are unavailable, expensive, or saturated;
- experiments with quantization, cache compression, and lightweight deployment;
- analysis of CPU memory bandwidth, NUMA effects, and thread scheduling behaviour.

The goal of this project was not simply to run an LLM on a CPU, but to understand the **performance envelope**, **bottlenecks**, and **trade-offs** that determine whether CPU inference can become practical for small and medium-sized models.

---

## Research Objectives

The project focused on the following questions:

1. How viable is LLM inference on CPU for small and medium-sized models?
2. How does `llama.cpp` behave under different CPU thread counts?
3. How do context size, decode length, and request concurrency affect latency and throughput?
4. What is the impact of cache and quantization strategies such as `f16:f16`, `turbo3:turbo3`, and `turbo4:turbo4`?
5. How do vanilla `llama.cpp` and TurboQuant-style variants compare?
6. How do selected BLAS builds affect CPU inference performance?
7. What metrics best describe CPU inference quality: TTFT, TPOT, throughput, goodput, memory usage, or answer quality?
8. Can RPC/node-scaling approaches improve feasibility for larger models?
9. What are the practical bottlenecks of CPU-based inference in HPC environments?

---

## Repository Structure

```text
.
├── TinyLlama-MLC/                 # Additional engine/model experimentation path
├── configs/
│   └── a1_arm_scaling.json        # Reusable benchmark configuration
├── llamacpp-tq/                   # TurboQuant-enabled llama.cpp tree
├── llamacpp-vanilla/              # Vanilla llama.cpp tree
├── mclllm/                        # Alternative CPU inference/runtime experiments
├── measurements/                  # Raw benchmark outputs and generated summaries
├── prompts/
│   └── track_a_prompts.json       # Fixed benchmark prompt set
├── scripts/
│   ├── analyze_bottleneck_model.py
│   ├── benchmark_llamacpp_cli.py
│   ├── benchmark_openai_stream.py
│   ├── benchmark_openai_stream_live.py
│   ├── evaluate_mandatory_outputs.py
│   ├── make_report_artifacts.py
│   ├── organize_a1_results.py
│   ├── parse_readable_a1_outputs.py
│   ├── summarize_a1_results.py
│   ├── validate_prompt_outputs.py
│   └── test_*.py
├── slurm/
│   ├── config/                    # Build jobs and smaller comparison jobs
│   └── sweep_benchmark/           # Main SLURM benchmark wrappers
├── vllm/                          # vLLM-related experimentation path
├── analysis_a1_benchmark.ipynb    # Notebook for result analysis
├── report.pdf                     # Final report
└── README.md
```

---

## Benchmarking Methodology

The benchmark methodology is based on executing a controlled matrix of inference configurations and collecting timing, throughput, memory, and quality-related metrics.

The main experiments use an OpenAI-compatible streaming benchmark client against `llama-server`. This makes it possible to evaluate different engine variants using a consistent request protocol.

Typical sweep dimensions include:

| Dimension | Example Values |
|---|---|
| Engine | TurboQuant, vanilla `llama.cpp` |
| Cache type | `f16:f16`, `turbo3:turbo3`, `turbo4:turbo4` |
| Threads | `1`, `4`, `8`, `16`, `24`, `48` |
| Request concurrency | `1`, `2`, `4`, `8`, `16` |
| Context size | `512`, `1024`, `2048`, `4096` |
| Decode length | `64`, `128`, `256`, `512` |
| Trials | Multiple trials with warm-up runs |

The default benchmark keeps the mandatory prompts while also using a broader fixed prompt set for throughput and quality analysis.

---

## What Is Measured

Each benchmark request records:

| Metric | Meaning |
|---|---|
| **TTFT** | Time to first token; measures initial response latency |
| **TPOT** | Time per output token; measures decode efficiency |
| **Decode throughput** | Generated output tokens per second |
| **Goodput** | Fraction of requests satisfying the configured latency SLA |
| **Total latency** | End-to-end request time |
| **Memory usage** | Process-level and node-level memory behaviour |
| **CPU utilisation** | Whole-node resource usage when enabled |
| **Generated text** | Captured output for mandatory prompt evaluation |
| **Quality score** | Rubric-based evaluation for selected mandatory prompts |

The mandatory prompt IDs are:

```text
mandatory_short_capital_france
mandatory_medium_ml_vs_dl
mandatory_long_transformer_cpu
```

---

## Inference Engines and Variants

The repository is organised around several CPU inference paths:

| Path | Purpose |
|---|---|
| `llamacpp-vanilla/` | Baseline vanilla `llama.cpp` implementation |
| `llamacpp-tq/` | TurboQuant-enabled `llama.cpp` implementation |
| `mclllm/` | Alternative inference/runtime experimentation |
| `TinyLlama-MLC/` | Additional model/runtime exploration |
| `vllm/` | vLLM-related experimentation path |

`llama.cpp` is the central reference engine because it provides efficient local inference, GGUF model support, CPU portability, and a practical `llama-server` interface for benchmark automation.

---

## Quantization and Cache Experiments

Quantization and cache format selection are central to the project.

The experiments consider the effect of:

- GGUF quantized models;
- reduced memory footprint;
- prompt processing speed;
- decode speed;
- cache format;
- runtime memory behaviour;
- output quality;
- scalability with CPU threads and concurrent requests.

Important cache/precision configurations include:

```text
vanilla: f16:f16
TurboQuant: turbo3:turbo3
TurboQuant: turbo4:turbo4
```

The goal is to understand the trade-off between:

- inference speed;
- memory efficiency;
- latency;
- throughput;
- quality degradation;
- practical deployability on CPU systems.

---

## One-Time Setup

From the repository root, make the benchmark scripts executable:

```bash
find slurm -name '*.sh' -exec chmod +x {} \;
chmod +x scripts/*.py
```

Large model files, credentials, virtual environments, generated caches, SLURM logs, and raw runtime artifacts should not be committed to version control.

---

## Model Configuration

The default Track A1 model set expects GGUF files under:

```text
llamacpp-tq/models/
```

Set `MODEL_SPECS` according to the models available on the target system:

```bash
export MODEL_SPECS="model-1-mandatory=$PWD/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PWD/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PWD/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"
```

A fourth model can be added as follows:

```bash
export MODEL_SPECS="$MODEL_SPECS;model-4=$PWD/llamacpp-tq/models/gpt-oss-20b-Q4_K_M.gguf"
```

---

## Running the Main ARM Sweep

The main ARM sweep is launched through SLURM.

```bash
MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

The default sweep covers:

- TurboQuant and vanilla `llama.cpp`;
- vanilla `f16:f16`;
- TurboQuant `turbo3:turbo3` and `turbo4:turbo4`;
- thread scaling from 1 to 48 threads;
- request concurrency from 1 to 16;
- context sizes from 512 to 4096;
- decode lengths from 64 to 512 tokens.

For a quick development smoke test:

```bash
MODEL_SPECS="$MODEL_SPECS" \
THREADS_LIST="4 8" CONCURRENCY_LIST="1" CTX_SIZE_LIST="512" MAX_TOKENS_LIST="64" \
TRIALS=1 WARMUP_TRIALS=0 \
sbatch --export=ALL --partition=dev-arm slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

---

## Running Follow-Up Experiments

### x86 CPU Sweep

```bash
MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh
```

### Focused ARM / REMORA Sweep

```bash
MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-remora-llamacpp-sweep.sh
```

### TurboQuant Stress Sweep

```bash
MODEL_SPECS="$MODEL_SPECS" \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-turboquant-stress-sweep.sh
```

### Vanilla OpenBLAS versus BLIS

```bash
sbatch --export=ALL slurm/config/build-vanilla-openblas-fujitsu.sh
sbatch --export=ALL slurm/config/build-vanilla-blis.sh

MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh
```

---

## RPC Node Scaling

RPC node scaling launches one `llama-server` coordinator on rank 0 and `rpc-server` workers on the remaining ranks.

This experiment is intended to test distributed model loading and remote execution behaviour, not simple data-parallel serving throughput.

```bash
SUBMIT_NODE_SWEEP=1 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-node-scaling.sh
```

With auto-build and an explicit local RPC server-count override:

```bash
SUBMIT_NODE_SWEEP=1 \
AUTO_BUILD=1 \
RPC_MAX_SERVERS=255 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-node-scaling.sh
```

Before interpreting RPC scaling results, inspect:

- `server.log`;
- worker logs;
- `system.txt`;
- connection acceptance messages;
- model-buffer allocation across workers.

---

## Outputs

A standard benchmark run writes results to:

```text
measurements/<jobid>-track-a1-server-sweep/
```

Important output files include:

| File | Purpose |
|---|---|
| `requests.jsonl` | Raw measured requests and generated responses |
| `summary.json` | Per-prompt timing means from the benchmark client |
| `resources.csv` | Process CPU and memory samples |
| `node_resources.csv` | Whole-node CPU and memory samples |
| `server.log` | Engine startup and runtime logs |
| `system.txt` | Resolved engine, model, cache, context, thread, and generation settings |
| `a1_server_summary.csv` | Per-prompt timing and memory summary |
| `a1_server_scaling_summary.csv` | Aggregate scaling table for plots |
| `mandatory_answer_quality.csv` | Mandatory-prompt quality scores |
| `mandatory_answer_quality_summary.json` | Mandatory-prompt quality summary |
| `model_inventory.csv` | Model reproducibility metadata |
| `engine_inventory.csv` | Engine reproducibility metadata |

---

## Post-Processing

Regenerate timing summaries:

```bash
python3 scripts/summarize_a1_results.py measurements/<run-dir> \
  --out measurements/<run-dir>/a1_server_summary.csv \
  --scaling-out measurements/<run-dir>/a1_server_scaling_summary.csv
```

Regenerate mandatory answer quality:

```bash
python3 scripts/evaluate_mandatory_outputs.py measurements/<run-dir> \
  --out measurements/<run-dir>/mandatory_answer_quality.csv \
  --summary-out measurements/<run-dir>/mandatory_answer_quality_summary.json
```

Validate generated answers:

```bash
python3 scripts/validate_prompt_outputs.py measurements/<run-dir> \
  --out measurements/<run-dir>/prompt_output_validation.csv \
  --summary-out measurements/<run-dir>/prompt_output_validation_summary.json
```

Create combined CSVs and report plots:

```bash
python3 scripts/make_report_artifacts.py measurements \
  --out-dir measurements/report_artifacts
```

If a `memory_bandwidth.csv` file with `triad_gb_s` exists, generate the TPOT bottleneck model:

```bash
python3 scripts/analyze_bottleneck_model.py measurements/<run-dir> \
  --out measurements/<run-dir>/bottleneck_model.csv
```

---

## Useful Environment Overrides

| Variable | Purpose |
|---|---|
| `AUTO_BUILD=0|1` | Require existing binaries or build before running |
| `MODEL_SPECS="label=/path/model.gguf;..."` | Model labels and paths |
| `ENGINE_SPECS="tq=/path/to/turboquant;vanilla=/path/to/llama.cpp"` | Engine repositories |
| `ENGINE_BIN_SPECS="label=/path/to/llama-server;..."` | Prebuilt server binaries |
| `ENGINE_PRELOAD_SPECS="label=/path/to/libblas.so;..."` | Per-engine `LD_PRELOAD` |
| `THREADS` | Baseline thread count |
| `CTX_SIZE` | Baseline context size |
| `PARALLEL_REQUESTS` | Baseline request concurrency |
| `MAX_TOKENS` | Baseline generation length |
| `THREADS_LIST` | Thread sweep values |
| `CONCURRENCY_LIST` | Concurrency sweep values |
| `CTX_SIZE_LIST` | Context-size sweep values |
| `MAX_TOKENS_LIST` | Decode-length sweep values |
| `ENGINE_CACHE_SWEEPS` | Engine/cache matrix |
| `MANDATORY_ONLY=1` | Run only the mandatory prompts |
| `LIMIT_PER_CATEGORY=5` | Cap prompt count per category while keeping mandatory prompts |
| `SUBMIT_ARRAY=1` | Submit derived SLURM array jobs |
| `ARRAY_CONCURRENCY=8` | Limit simultaneous array tasks |
| `SUMMARY_ONLY=1` | Regenerate summaries without rerunning benchmarks |
| `LIST_CONFIGS=1` | Print or write the planned configuration matrix |
| `EXCLUDE_CONFIGS` | Skip known-bad configurations |
| `SERVER_EXTRA_ARGS="--numa"` | Pass extra flags to `llama-server` |

---

## Validation and Tests

Before submitting full benchmark campaigns, validate the Python utilities and benchmark entrypoints:

```bash
python3 -m py_compile scripts/*.py
python3 scripts/test_evaluate_mandatory_outputs.py
python3 scripts/test_validate_prompt_outputs.py
python3 scripts/test_benchmark_protocol.py
python3 scripts/test_slurm_entrypoints.py
```

Additional tests present in the repository include:

```bash
python3 scripts/test_analyze_bottleneck_model.py
python3 scripts/test_benchmark_llamacpp_cli.py
python3 scripts/test_notebook_readable_loader.py
python3 scripts/test_parse_readable_a1_outputs.py
python3 scripts/test_slurm_accounts.py
python3 scripts/test_summarize_a1_results.py
python3 scripts/test_ultimate_report_suite.py
```

For SLURM-related modifications, run a small development-partition smoke test before launching a full sweep.

---

## Technologies

- Python
- Shell scripting
- C/C++
- `llama.cpp`
- TurboQuant-style inference/cache experiments
- GGUF quantized models
- OpenAI-compatible streaming benchmark protocol
- SLURM
- Linux
- Deucalion HPC environment
- CPU benchmarking and resource monitoring
- CSV/JSON result processing
- Jupyter notebooks

---

## Learning Outcomes

This project developed practical and research-level understanding of:

- LLM inference pipelines;
- CPU-based model serving;
- `llama.cpp` server execution;
- GGUF model deployment;
- quantization trade-offs;
- cache-format effects;
- time-to-first-token and token-generation behaviour;
- request concurrency and throughput scaling;
- memory footprint and runtime resource usage;
- SLURM-based benchmark automation;
- reproducible HPC experimentation;
- benchmark summarisation and report artifact generation;
- answer-quality validation for mandatory prompts.

The project achieved a **final grade of 20/20**.

---

## Future Work

Possible extensions include:

- deeper NUMA-aware scheduling with `numactl`;
- more representative benchmark sets per model family;
- expanded comparison between cache quantization strategies;
- broader comparison between CPU architectures;
- detailed profiling with `perf`, hardware counters, and memory-bandwidth measurements;
- explicit roofline-style modelling for decode bottlenecks;
- integration with REMORA for richer node-level monitoring;
- evaluation of FlashAttention-like CPU techniques where applicable;
- comparison with additional inference engines;
- automated generation of reproducible benchmark reports;
- heterogeneous CPU/GPU comparison;
- lightweight local assistant deployment using the optimized CPU inference pipeline.

---

## Codebase Hygiene

The repository may contain generated or local-only artifacts, such as:

- `.DS_Store`;
- macOS AppleDouble files such as `._*`;
- Python `__pycache__/`;
- local virtual environments;
- SLURM logs;
- generated benchmark outputs;
- large model/runtime folders.

These files should not be committed unless they are intentionally required for reproducibility. Large GGUF models should normally remain outside version control.

---

## Project Status

This repository is an academic and research-oriented exploration of **efficient LLM inference on CPU systems**.

It serves as a practical benchmark harness for studying how modern LLM workloads behave outside GPU-centric infrastructures, especially under HPC-style execution, quantization, runtime tuning, and scaling constraints.
