# Track A1 llama.cpp Server Sweep


This repository contains the Track A1 benchmark harness for evaluating the `llama.cpp` engine family on Deucalion CPU nodes. It compares vanilla `llama.cpp`, TurboQuant, selected BLAS builds, ARM versus x86 wrappers, and RPC node-scaling runs using one OpenAI-compatible streaming benchmark protocol.

The main entrypoints are under `slurm/sweep_benchmark/`; Python utilities under `scripts/` benchmark, validate, summarize, and plot the raw outputs written to `measurements/`.

## Repository layout

- `slurm/sweep_benchmark/`: Track A1 SLURM sweep wrappers.
- `slurm/config/`: build jobs, single-engine run jobs, and smaller comparison jobs.
- `scripts/benchmark_openai_stream.py`: OpenAI-compatible streaming benchmark client.
- `scripts/summarize_a1_results.py`: builds timing and scaling CSV summaries from raw runs.
- `scripts/evaluate_mandatory_outputs.py`: scores the mandatory assignment prompts.
- `scripts/validate_prompt_outputs.py`: optional reference/rubric validation for generated answers.
- `scripts/make_report_artifacts.py`: creates combined CSVs and SVG report plots.
- `prompts/track_a_prompts.json`: fixed prompt set, including the three mandatory prompts.
- `configs/`: reusable experiment configuration files.
- `measurements/`: raw benchmark outputs and generated summaries.
- `docs/analysis_a1_benchmark.ipynb`: notebook that reads parsed results from `measurements/readable/`.
- `docs/final-reproduction-guide.md`: longer reproduction guide kept for reference.
- `docs/run-notes.md`: short command notes from previous runs.
- `docs/prompt-answers.typ`: report material for prompt-answer discussion.
- `bibliography/`: bibliography files used by the report.
- `llamacpp-tq/llama-cpp-turboquant`: TurboQuant engine tree.
- `llamacpp-vanilla/llama.cpp`: vanilla `llama.cpp` engine tree.

## What is measured

Each benchmark request records:

- TTFT: time to first token.
- TPOT: time per output token.
- Decode throughput: generated output tokens per second.
- Goodput: fraction of requests satisfying the configured latency SLA.
- Generated text, especially for mandatory-prompt reporting.
- Process memory samples from `resources.csv`.
- Whole-node CPU and memory samples from `node_resources.csv` when enabled.

The mandatory prompt IDs are:

- `mandatory_short_capital_france`
- `mandatory_medium_ml_vs_dl`
- `mandatory_long_transformer_cpu`

## One-time setup

Run from the repository root on Deucalion:

```bash
find slurm -name '*.sh' -exec chmod +x {} \;
chmod +x scripts/*.py
```

Keep large GGUF model files, credentials, virtual environments, raw SLURM logs, and generated caches out of version control.

## Model paths

The default Track A1 model set expects GGUF files under `llamacpp-tq/models/`. Override `MODEL_SPECS` whenever your model paths differ:

```bash
export MODEL_SPECS="model-1-mandatory=$PWD/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PWD/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PWD/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"
```

Add a fourth model if needed, for example:

```bash
export MODEL_SPECS="$MODEL_SPECS;model-4=$PWD/llamacpp-tq/models/gpt-oss-20b-Q4_K_M.gguf"
```

## Replicate the main ARM sweep

Submit one SLURM array task per benchmark configuration. The wrapper can derive the array range automatically, or you can submit an explicit range such as `sbatch --array=0-63%8 --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh` after checking the config matrix.

```bash
MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

The default sweep uses five prompts per category, keeps the three mandatory prompts, runs vanilla only `f16:f16`, and runs TurboQuant with `turbo3:turbo3` and `turbo4:turbo4`.

The default sweep covers:

- Engines: TurboQuant and vanilla `llama.cpp`.
- Cache types: vanilla `f16:f16`; TurboQuant `turbo3:turbo3` and `turbo4:turbo4`.
- Thread scaling: `1 4 8 16 24 48`.
- Request concurrency: `1 2 4 8 16`.
- Context sizes: `512 1024 2048 4096`.
- Decode lengths: `64 128 256 512`.

For a quick development smoke test, reduce the sweep:

```bash
MODEL_SPECS="$MODEL_SPECS" \
THREADS_LIST="4 8" CONCURRENCY_LIST="1" CTX_SIZE_LIST="512" MAX_TOKENS_LIST="64" \
TRIALS=1 WARMUP_TRIALS=0 \
sbatch --export=ALL --partition=dev-arm slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

## Replicate focused follow-up runs

x86 CPU sweep:

```bash
MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh
```

Focused ARM throughput sweep:

```bash
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-focused.sh
```

Vanilla OpenBLAS versus BLIS:

```bash
sbatch --export=ALL slurm/config/build-vanilla-openblas-fujitsu.sh
sbatch --export=ALL slurm/config/build-vanilla-blis.sh

MODEL_SPECS="$MODEL_SPECS" \
TRIALS=3 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh
```

## Replicate RPC node scaling

RPC node scaling launches one `llama-server` coordinator on rank 0 and `rpc-server` workers on the other ranks. Use it to test distributed model-parallel loading/scaling, not data-parallel serving throughput.

```bash
SUBMIT_NODE_SWEEP=1 sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-node-scaling.sh
```

With auto-build and the local RPC server-count override:

```bash
SUBMIT_NODE_SWEEP=1 \
AUTO_BUILD=1 \
RPC_MAX_SERVERS=255 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-node-scaling.sh
```

By default, the generic node-scaling sweep runs 1, 2, 4, 6, 8, 16, 24, 32, 64, 128, and 256 nodes and writes `a1_node_scaling_summary.csv` plus `a1_node_scaling_scaling_summary.csv` after the chain completes.

For the GPT-OSS 120B Q8_0 wrapper:

```bash
NODE_COUNTS="8 12 16" \
SERVER_STARTUP_WAIT_SECONDS=3600 \
NODE_JOB_TIME=08:00:00 \
MAX_TOKENS=16 \
TRIALS=1 WARMUP_TRIALS=0 \
MANDATORY_ONLY=1 LIMIT_PER_CATEGORY=1 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-gptoss-node-scaling.sh
```

Check `server.log`, worker logs, and `system.txt` before interpreting scaling results. Confirm that RPC workers accepted connections and that each worker contributed model-buffer storage.

## Outputs

A normal sweep writes to:

```text
measurements/<jobid>-track-a1-server-sweep/
```

Important outputs include:

- `requests.jsonl`: raw measured requests, including each response `generated_text`.
- `summary.json`: per-prompt timing means from the benchmark client.
- `resources.csv`: process CPU and memory samples.
- `node_resources.csv`: whole-node CPU and memory samples.
- `server.log`: engine startup and runtime logs.
- `system.txt`: resolved model, engine, cache, context, thread, and generation settings.
- `a1_server_summary.csv`: per-prompt timing and memory summary.
- `a1_server_scaling_summary.csv`: aggregate scaling table for plots.
- `mandatory_answer_quality.csv`: mandatory-prompt quality scores.
- `mandatory_answer_quality_summary.json`: mandatory-prompt quality summary.
- `model_inventory.csv` and `engine_inventory.csv`: reproducibility metadata.

## Post-processing

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

Validate generated answers with the built-in mandatory rubric or a custom rubric:

```bash
python3 scripts/validate_prompt_outputs.py measurements/<run-dir> \
  --out measurements/<run-dir>/prompt_output_validation.csv \
  --summary-out measurements/<run-dir>/prompt_output_validation_summary.json
```

Create combined CSVs and report plots from all current result folders:

```bash
python3 scripts/make_report_artifacts.py measurements \
  --out-dir measurements/report_artifacts
```

If a `memory_bandwidth.csv` file with `triad_gb_s` exists, build the TPOT bottleneck model table:

```bash
python3 scripts/analyze_bottleneck_model.py measurements/<run-dir> \
  --out measurements/<run-dir>/bottleneck_model.csv
```

## Useful environment overrides

- `AUTO_BUILD=0|1`: require existing binaries or build before running.
- `MODEL_SPECS="label=/path/model.gguf;..."`: model labels and paths.
- `ENGINE_SPECS="tq=/path/to/turboquant;vanilla=/path/to/llama.cpp"`: engine repositories.
- `ENGINE_BIN_SPECS="label=/path/to/llama-server;..."`: prebuilt server binaries.
- `ENGINE_PRELOAD_SPECS="label=/path/to/libblas.so;..."`: per-engine `LD_PRELOAD`.
- `THREADS`, `CTX_SIZE`, `PARALLEL_REQUESTS`, `MAX_TOKENS`: baseline values.
- `THREADS_LIST`, `CONCURRENCY_LIST`, `CTX_SIZE_LIST`, `MAX_TOKENS_LIST`: sweep dimensions.
- `ENGINE_CACHE_SWEEPS="tq=turbo3:turbo3 turbo4:turbo4;vanilla=f16:f16"`: cache sweeps.
- `MANDATORY_ONLY=1`: run only the three mandatory prompts.
- `LIMIT_PER_CATEGORY=5`: cap prompt count per category while keeping mandatory prompts.
- `SUBMIT_ARRAY=1`: submit derived SLURM array jobs.
- `ARRAY_CONCURRENCY=8`: cap simultaneous array tasks.
- `SUMMARY_ONLY=1`: regenerate summaries without rerunning benchmarks.
- `LIST_CONFIGS=1`: print/write the planned configuration matrix.
- `EXCLUDE_CONFIGS="engine:model_label:cache_k[:cache_v]"`: skip known-bad configurations.
- `SERVER_EXTRA_ARGS="--numa"`: pass extra flags to `llama-server`.

## Validation after edits

Run these before submitting benchmark jobs after modifying Python or entrypoint logic:

```bash
python3 -m py_compile scripts/*.py
python3 scripts/test_evaluate_mandatory_outputs.py
python3 scripts/test_validate_prompt_outputs.py
python3 scripts/test_benchmark_protocol.py
python3 scripts/test_slurm_entrypoints.py
```

For SLURM changes, also run a small dev-partition smoke test before generating report data.

## Codebase hygiene notes

The repository currently contains generated or local-only artifacts such as macOS AppleDouble files (`._*`), `.DS_Store`, Python `__pycache__/`, local virtual environments, SLURM logs, and large generated model/runtime folders. These should not be committed. Move or delete them only after confirming they are not needed for reproducibility.

## Future works

For the future work we are going to use numactl, and test with more representative benchmarks for each model optimization and flash-attention.
