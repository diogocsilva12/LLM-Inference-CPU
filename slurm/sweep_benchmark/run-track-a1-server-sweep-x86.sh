#!/usr/bin/env bash
#SBATCH --job-name=a1-server-sweep-x86
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
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
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh"
if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base sweep script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
  echo "Missing x86 wrapper script: $WRAPPER_SCRIPT" >&2
  exit 2
fi

export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-server-sweep-x86}"
export ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm-x86}"
export ENGINE_SPECS="${ENGINE_SPECS:-vanilla-x86=$PROJECT_DIR/llamacpp-vanilla/llama.cpp}"
export ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-vanilla-x86=$PROJECT_DIR/llamacpp-vanilla/llama.cpp/build-slurm-x86/bin/llama-server}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vanilla-x86=f16:f16}"
export AUTO_BUILD="${AUTO_BUILD:-0}"
export THREADS="${THREADS:-32}"
export THREADS_LIST="${THREADS_LIST:-8 16 24 32 48 64 96 128}"
export CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16}"
export CTX_SIZE_LIST="${CTX_SIZE_LIST:-512 1024 2048 4096}"
export MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-64 128 256 512}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export PROJECT_DIR
export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

exec "$BASE_SWEEP_SCRIPT" "$@"
