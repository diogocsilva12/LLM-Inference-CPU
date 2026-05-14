#!/usr/bin/env bash
#SBATCH --job-name=a1-server-sweep-focused
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${PROJECT_DIR:-}" ]]; then
  for candidate in "$SUBMIT_DIR" "$SUBMIT_DIR/.." "$SCRIPT_DIR/../.."; do
    if [[ -f "$candidate/prompts/track_a_prompts.json" ]]; then
      PROJECT_DIR="$(cd -- "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "Unable to locate project directory; set PROJECT_DIR=/path/to/aded-project-llamacpp." >&2
  exit 2
fi

BASE_SWEEP_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep.sh"
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep-focused.sh"
if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base sweep script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
  echo "Missing focused wrapper script: $WRAPPER_SCRIPT" >&2
  exit 2
fi

# Use absolute model paths so submissions are robust to SSHFS mount context and
# submit-directory differences.
default_model_specs="model-1-mandatory=$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PROJECT_DIR/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PROJECT_DIR/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"

export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-server-sweep-focused}"
export MODEL_SPECS="${MODEL_SPECS:-$default_model_specs}"
export THREADS="${THREADS:-46}"
export THREADS_BATCH="${THREADS_BATCH:-46}"
export CTX_SIZE="${CTX_SIZE:-2048}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export THREADS_LIST="${THREADS_LIST:-24 36 46 48}"
export CONCURRENCY_LIST="${CONCURRENCY_LIST:-1}"
export CTX_SIZE_LIST="${CTX_SIZE_LIST:-1024 2048}"
export MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-128 256}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-tq=turbo2:turbo2 turbo4:turbo4;vanilla=f16:f16}"
export MANDATORY_ONLY="${MANDATORY_ONLY:-1}"
export LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-}"
export TRIALS="${TRIALS:-3}"
export WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
export RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-0}"
export RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-1}"
export RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-0}"
export RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-1}"
export RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-1}"
export RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"
export PROJECT_DIR
export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

exec "$BASE_SWEEP_SCRIPT" "$@"
