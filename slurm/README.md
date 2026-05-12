# SLURM Scripts For Track A1 CPU Benchmarks

The project assignment is CPU-only. Use ARM/x86 scripts for report data. A100 scripts exist for separate engineering tests, but do not use GPU results as Track A measurements.

Run these commands on a Deucalion login node where `sbatch` is available.

## CPU Scripts

Build, run, and helper jobs are in `slurm/config/`:

- `config/build-llama.sh`: build the TurboQuant llama.cpp fork on ARM CPU nodes.
- `config/run-llama.sh`: run `llama-server` on ARM CPU nodes.
- `config/build-llama-x86.sh`: build on x86 CPU nodes via the matrix builder.
- `config/run-llama-x86.sh`: run on x86 CPU nodes via the matrix runner.
- `config/build-llama-matrix.sh`: target-aware incremental build for `arm-cpu`, `x86-cpu`, and optional `a100-cuda`.
- `config/run-llama-matrix.sh`: target-aware server runner.
- `config/benchmark-llama-matrix.sh`: `llama-bench` throughput sweep.
- `config/build-vanilla-openblas-fujitsu.sh`: build vanilla `llama.cpp` with Fujitsu OpenBLAS.
- `config/build-vanilla-blis.sh`: build vanilla `llama.cpp` with BLIS.
- `config/submit-llama.sh`: wrapper for build, run, and `llama-bench` benchmark jobs.

Track A1 sweep benchmark jobs are in `slurm/sweep_benchmark/`:

- `sweep_benchmark/run-track-a1-server-sweep.sh`: Track A1 HTTP sweep; starts `llama-server`, records TTFT/TPOT/output/memory, and scores mandatory prompt answers.
- `sweep_benchmark/run-track-a1-server-sweep-x86.sh`: x86 wrapper for the Track A1 HTTP sweep.
- `sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh`: compare the two vanilla BLAS builds with the same HTTP sweep.

## Build

ARM:

```bash
sbatch slurm/config/build-llama.sh
```

x86:

```bash
sbatch slurm/config/build-llama-x86.sh
```

Vanilla BLAS variants:

```bash
sbatch slurm/config/build-vanilla-openblas-fujitsu.sh
sbatch slurm/config/build-vanilla-blis.sh
```

Matrix build with direct `sbatch`:

```bash
TARGET=arm-cpu sbatch slurm/config/build-llama-matrix.sh
TARGET=x86-cpu sbatch slurm/config/build-llama-matrix.sh
```

Build scripts compile `llama-server`, `llama-bench`, and `llama-perplexity` by default.

## Run Server

ARM:

```bash
MODEL_PATH=/absolute/path/model.gguf sbatch slurm/config/run-llama.sh
```

x86:

```bash
MODEL_PATH=/absolute/path/model.gguf sbatch slurm/config/run-llama-x86.sh
```

Useful CPU overrides:

```bash
THREADS=24 CTX_SIZE=2048 PARALLEL_REQUESTS=1 \
CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 \
MODEL_PATH=/absolute/path/model.gguf \
sbatch slurm/config/run-llama.sh
```

## Track A1 HTTP Sweep

Use this for the requested Track A1 run over TurboQuant and vanilla `llama.cpp`, three models, thread scaling, concurrency, context length, decode length, and KV-cache quantisation. By default, vanilla uses `f16:f16`, TurboQuant uses `turbo3:turbo3` and `turbo4:turbo4`, and the prompt set is the three mandatory prompts plus six additional prompts:

```bash
MODEL_SPECS="model-1-mandatory=llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf" \
TRIALS=1 WARMUP_TRIALS=1 \
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 \
sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

The default `EXCLUDE_CONFIGS=tq:model-2:*` skips the known TurboQuant/SmolLM2 startup crash. Use `EXCLUDE_CONFIGS=none` only when intentionally retesting that combination.

Outputs go to `measurements/<jobid>-track-a1-server-sweep/`:

- `requests.jsonl`: per request raw metrics and generated text.
- `a1_server_summary.csv`: TTFT, TPOT, throughput, goodput, and peak memory summary.
- `mandatory_answer_quality.csv`: answer-quality checks for the three mandatory prompts.
- `engine_inventory.csv`: compared engine repositories and server binaries.
- `resources.csv`: sampled server RSS and CPU percent inside each configuration directory.
- `server.log`: engine startup, cache type, and KV memory lines inside each configuration directory.
- `system.txt`: node and configuration metadata inside each configuration directory.

For faster collection, run the same script as a SLURM array. The script can derive the array range and submit it:

```bash
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

Check the planned count first if needed:

```bash
ALLOW_DIRECT=1 LIST_CONFIGS=1 bash slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

Manual array submission still works. With the current defaults, use:

```bash
sbatch --array=0-63%8 --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

After the array completes, submit one summary pass over the array output directory:

```bash
OUTPUT_DIR=$PWD/measurements/<array-jobid>-track-a1-server-sweep \
SUMMARY_ONLY=1 \
sbatch --dependency=afterany:<array-jobid> --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh
```

For x86, use:

```bash
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh
```

The x86 wrapper uses `ENGINE_BUILD_DIR_NAME=build-slurm-x86`, requests 128 CPUs, and sweeps `THREADS_LIST=8 16 24 32 48 64 96 128` by default.

For the vanilla BLAS comparison, build both binaries first and then run:

```bash
SUBMIT_ARRAY=1 ARRAY_CONCURRENCY=8 sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh
```

The BLAS sweep compares `vanilla-openblas-fujitsu` and `vanilla-blis`. The OpenBLAS build uses `OpenBLAS/0.3.26-GCC-13.3.0-Fujitsu`; the BLIS build uses `BLIS/1.0-GCC-13.3.0`. Runtime library injection is controlled by `ENGINE_PRELOAD_SPECS`, with convenience overrides `OPENBLAS_PRELOAD` and `BLIS_PRELOAD`.

## llama-bench Throughput Sweep

Use this for prompt tokens/s and generation tokens/s:

```bash
CACHE_SWEEP="f16:f16 turbo3:turbo3 turbo4:turbo4" \
PROMPT_TOKENS="128,512,1024,2048" \
GEN_TOKENS="64,128,256,512" \
REPETITIONS=5 \
MODEL_PATH=/absolute/path/model.gguf \
sbatch slurm/config/benchmark-llama-matrix.sh
```

For x86:

```bash
TARGET=x86-cpu MODEL_PATH=/absolute/path/model.gguf sbatch slurm/config/benchmark-llama-matrix.sh
```

Useful overrides:

- `CACHE_SWEEP`: space-separated `K:V` cache pairs.
- `PROMPT_TOKENS`: prefill token counts passed to `llama-bench -p`.
- `GEN_TOKENS`: decode token counts passed to `llama-bench -n`.
- `BATCH_SIZE`, `UBATCH_SIZE`: batch and micro-batch sizes.
- `REPETITIONS`: repetitions per test point.
- `OUTPUT_FORMAT`: `jsonl`, `json`, `csv`, `md`, or `sql`.
- `OUTPUT_DIR`: destination directory for raw benchmark output.

## Report Artifacts

Regenerate combined CSVs and SVG plots from all current measurement folders:

```bash
python3 scripts/make_report_artifacts.py measurements \
  --out-dir measurements/report_artifacts
```

This writes:

- `measurements/a1_all_results_summary.csv`
- `measurements/a1_all_results_scaling_summary.csv`
- `measurements/report_artifacts/all_scaling_rows.csv`
- SVG plots for throughput, TTFT, TPOT, goodput, memory, engines, cache types, threads, concurrency, context length, and decode length.

## Suggested A1 Dimensions

Vary one thing at a time:

- Threading: `THREADS=12,24,36,48` on ARM to align with 12-core NUMA CMGs.
- Context length: `CTX_SIZE=512,1024,2048,4096`, with matching prompt categories.
- Decode length: `MAX_TOKENS=64,128,256,512`.
- Concurrent requests: `PARALLEL_REQUESTS=1,2,4,8` and `BENCH_CONCURRENCY=1,2,4,8`.
- Node architecture: same model/config on `normal-arm` and `normal-x86`.
- Cache type in this fork: `f16`, `turbo2`, `turbo3`, `turbo4` for K/V where supported.

## Notes

- Use `dev-arm` or `dev-x86` for setup and short tests; use `normal-arm` or `normal-x86` for final benchmark runs.
- Keep raw outputs for every run. The report should show mean plus standard deviation over at least three trials.
- The prompt dataset is `prompts/track_a_prompts.json` and includes the three mandatory assignment prompts.
- Post-process completed HTTP benchmark runs with `python3 scripts/summarize_a1_results.py measurements --out measurements/a1_summary.csv`.
