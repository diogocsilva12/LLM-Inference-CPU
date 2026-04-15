#!/usr/bin/env bash
#SBATCH --job-name=serve-llamacpp-tq
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Update these for your cluster if needed.
# SBATCH --partition=gpu
# SBATCH --account=your-account
# module purge
# module load cuda/12.4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-2}"

cd "$REPO_DIR"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing llama-server binary: $SERVER_BIN" >&2
  echo "Submit build job first: sbatch scripts/slurm/build-llamacpp-turboquant.sbatch" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 1
fi

# TurboQuant KV cache is enabled via cache types at runtime.
exec "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$N_GPU_LAYERS" \
  -c "$CTX_SIZE" \
  -np "$PARALLEL_REQUESTS" \
  --flash-attn \
  --cache-type-k turbo3_0 \
  --cache-type-v turbo3_0
