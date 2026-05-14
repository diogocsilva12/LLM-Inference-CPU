#!/usr/bin/env bash
#SBATCH --job-name=a1-mlc-sweep-x86
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=126
#SBATCH --time=08:00:00
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
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-mlc-sweep-x86.sh"
ADAPTER="${MLC_ADAPTER:-$PROJECT_DIR/slurm/config/serve-mlc-openai-adapter.sh}"

if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base sweep script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -x "$ADAPTER" ]]; then
  echo "Missing or non-executable MLC adapter: $ADAPTER" >&2
  echo "Run: chmod +x $PROJECT_DIR/slurm/config/serve-mlc-openai-adapter.sh" >&2
  exit 2
fi

SETUP_HELPER="${VLLM_MLC_SETUP_HELPER:-$PROJECT_DIR/slurm/config/setup-vllm-mlc-env.sh}"
if [[ ! -f "$SETUP_HELPER" ]]; then
  echo "Missing vLLM/MLC setup helper: $SETUP_HELPER" >&2
  exit 2
fi
# Optional build/setup phase. Set MLC_BUILD=1 to create/update MLC_VENV_DIR
# and automatically export MLC_PYTHON before the sweep runs.
# Set MLC_BUILD_ONLY=1 to prepare the environment and exit without benchmarking.
if [[ "${MLC_BUILD_ONLY:-0}" == "1" || "${MLC_BUILD_ONLY:-0}" == "true" ]]; then
  export MLC_BUILD=1
fi
export MLC_VENV_DIR="${MLC_VENV_DIR:-$HOME/venvs/mlc-cpu-x86}"
source "$SETUP_HELPER"
if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  setup_mlc_env
fi
if [[ "${MLC_BUILD_ONLY:-0}" == "1" || "${MLC_BUILD_ONLY:-0}" == "true" ]]; then
  echo "[OK] MLC build/setup phase complete: MLC_PYTHON=${MLC_PYTHON:-python3}"
  exit 0
fi

# Same four labels as the llama.cpp Track A1 run. Override MLC_MODEL_SPECS or MODEL_SPECS
# when MLC needs Hugging Face directories or MLC-compiled model directories
# instead of the GGUF files used by llama.cpp.
default_model_specs="model-1-mandatory=$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PROJECT_DIR/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PROJECT_DIR/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf;model-4=$PROJECT_DIR/llamacpp-tq/models/gpt-oss-20b-Q4_K_M.gguf"

export PROJECT_DIR
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-mlc-sweep-x86}"
export MODEL_SPECS="${MODEL_SPECS:-${MLC_MODEL_SPECS:-$default_model_specs}}"
export ENGINE_SPECS="${ENGINE_SPECS:-mlc=$PROJECT_DIR/mclllm}"
export ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-mlc=$ADAPTER}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-mlc=f16:f16}"
export AUTO_BUILD="${AUTO_BUILD:-0}"
export EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-none}"

# X86 defaults. The base harness still lets you override every dimension.
export THREADS="${THREADS:-126}"
export THREADS_BATCH="${THREADS_BATCH:-126}"
export THREADS_LIST="${THREADS_LIST:-8 16 24 32 48 64 96 126}"
export CTX_SIZE="${CTX_SIZE:-2048}"
export CTX_SIZE_LIST="${CTX_SIZE_LIST:-512 1024 2048 4096}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-$PARALLEL_REQUESTS}"
export CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-64 128 256 512}"

# Full Track A1 prompt protocol by default: 5 prompts/category + mandatory prompts.
export MANDATORY_ONLY="${MANDATORY_ONLY:-0}"
export LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"
export TRIALS="${TRIALS:-3}"
export WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"

# Full single-engine optimisation sweep, matching the llama.cpp harness shape.
export RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"
export RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-1}"
export RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-1}"
export RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-1}"
export RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-1}"
export RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"

export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

if [[ -z "${MLC_PYTHON:-}" ]]; then
  echo "[WARN] MLC_PYTHON is not set; adapter will use python3 from the job environment." >&2
fi

exec "$BASE_SWEEP_SCRIPT" "$@"
