#!/usr/bin/env bash
#SBATCH --job-name=a1-tq-stress
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
BASE_SWEEP_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep.sh"

export PROJECT_DIR
export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-turboquant-stress-sweep.sh}"

# TurboQuant-only stress sweep. The base sweep still validates that the model
# set contains the mandatory Llama 3.1 8B model plus at least two other models.
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-turboquant-stress-sweep}"
export ENGINE_SPECS="${ENGINE_SPECS:-tq=$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant}"
export ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-tq=$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant/build-slurm/bin/llama-server}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-tq=turbo2:turbo2 turbo3:turbo3 turbo4:turbo4}"
export AUTO_BUILD="${AUTO_BUILD:-1}"

# Recommended ARM baseline from the latest sweeps: 46 useful worker threads.
export THREADS="${THREADS:-46}"
export THREADS_BATCH="${THREADS_BATCH:-46}"
export THREADS_LIST="${THREADS_LIST:-24 36 46}"

# Stress dimensions: longer contexts, higher concurrency, and longer decode.
export CTX_SIZE="${CTX_SIZE:-8192}"
export CTX_SIZE_LIST="${CTX_SIZE_LIST:-4096 8192 12288 16384}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-$PARALLEL_REQUESTS}"
export CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16 24}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-128 256 512 768}"

export RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"
export RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-1}"
export RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-1}"
export RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-1}"
export RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-1}"
export RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-1}"

export LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"
export TRIALS="${TRIALS:-3}"
export WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
export ENABLE_NODE_RESOURCE_MONITOR="${ENABLE_NODE_RESOURCE_MONITOR:-1}"
export RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-1}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export FLASH_ATTN="${FLASH_ATTN:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"

cd "$SUBMIT_DIR"
exec "$BASE_SWEEP_SCRIPT" "$@"
