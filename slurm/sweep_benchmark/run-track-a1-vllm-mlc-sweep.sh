#!/usr/bin/env bash
#SBATCH --job-name=a1-vllm-mlc-sweep
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
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
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-vllm-mlc-sweep.sh"
if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base sweep script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi

VLLM_ADAPTER="${VLLM_ADAPTER:-$PROJECT_DIR/slurm/config/serve-vllm-openai-adapter.sh}"
MLC_ADAPTER="${MLC_ADAPTER:-$PROJECT_DIR/slurm/config/serve-mlc-openai-adapter.sh}"

SETUP_HELPER="${VLLM_MLC_SETUP_HELPER:-$PROJECT_DIR/slurm/config/setup-vllm-mlc-env.sh}"
if [[ ! -f "$SETUP_HELPER" ]]; then
  echo "Missing vLLM/MLC setup helper: $SETUP_HELPER" >&2
  exit 2
fi
# Optional build/setup phase. Set VLLM_BUILD=1 and/or MLC_BUILD=1 to
# create/update the corresponding venvs and export VLLM_PYTHON/MLC_PYTHON.
# Set VLLM_BUILD_ONLY=1 or MLC_BUILD_ONLY=1 to prepare envs and exit.
if [[ "${VLLM_BUILD_ONLY:-0}" == "1" || "${VLLM_BUILD_ONLY:-0}" == "true" ]]; then
  export VLLM_BUILD=1
fi
if [[ "${MLC_BUILD_ONLY:-0}" == "1" || "${MLC_BUILD_ONLY:-0}" == "true" ]]; then
  export MLC_BUILD=1
fi
export VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/venvs/vllm-cpu-arm}"
export MLC_VENV_DIR="${MLC_VENV_DIR:-$HOME/venvs/mlc-cpu-arm}"
source "$SETUP_HELPER"
if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  setup_vllm_env
  setup_mlc_env
fi
if [[ "${VLLM_BUILD_ONLY:-0}" == "1" || "${VLLM_BUILD_ONLY:-0}" == "true" || "${MLC_BUILD_ONLY:-0}" == "1" || "${MLC_BUILD_ONLY:-0}" == "true" ]]; then
  echo "[OK] vLLM/MLC build/setup phase complete: VLLM_PYTHON=${VLLM_PYTHON:-python3} MLC_PYTHON=${MLC_PYTHON:-python3}"
  exit 0
fi

default_model_specs="model-1-mandatory=$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PROJECT_DIR/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PROJECT_DIR/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf;model-4=$PROJECT_DIR/llamacpp-tq/models/gpt-oss-20b-Q4_K_M.gguf"

export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-vllm-mlc-sweep}"
export MODEL_SPECS="${MODEL_SPECS:-$default_model_specs}"
export ENGINE_SPECS="${ENGINE_SPECS:-vllm=$PROJECT_DIR/vllm;mlc=$PROJECT_DIR/mclllm}"
export ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-vllm=$VLLM_ADAPTER;mlc=$MLC_ADAPTER}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vllm=f16:f16;mlc=f16:f16}"
export AUTO_BUILD="${AUTO_BUILD:-0}"

# ARM-optimal defaults from earlier sweeps.
export THREADS="${THREADS:-46}"
export THREADS_BATCH="${THREADS_BATCH:-46}"
export CTX_SIZE="${CTX_SIZE:-1024}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1}"
export MAX_TOKENS="${MAX_TOKENS:-128}"

export MANDATORY_ONLY="${MANDATORY_ONLY:-1}"
export LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-}"
export TRIALS="${TRIALS:-3}"
export WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"

# Keep this as a model-only sweep on fixed "best known" runtime settings.
export RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"
export RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-0}"
export RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-0}"
export RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-0}"
export RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-0}"
export RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"

export PROJECT_DIR
export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

exec "$BASE_SWEEP_SCRIPT" "$@"
