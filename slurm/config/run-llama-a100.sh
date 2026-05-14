#!/usr/bin/env bash
#SBATCH --job-name=run-llama-a100
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "[STARTING] Loading modules"
modules=(
  "GCC/13.3.0"
  "cmake/3.21.3"
  "CUDA"
)

if command -v module >/dev/null 2>&1; then
  module --ignore_cache purge
else
  ml purge
fi
for module in "${modules[@]}"; do
  echo "[LOAD_MODULE] $module"
  if command -v module >/dev/null 2>&1; then
    module --ignore_cache load "$module"
  else
    ml "$module"
  fi
done

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
  CANDIDATES=(
    "$SUBMIT_DIR/llamacpp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/llamacpp-tq"
    "$SUBMIT_DIR/llamaccp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/llamaccp-tq"
    "$SUBMIT_DIR/../llamacpp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/../llamacpp-tq"
    "$SUBMIT_DIR/../llamaccp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/../llamaccp-tq"
    "$SCRIPT_DIR/../../llamacpp-tq/llama-cpp-turboquant"
    "$SCRIPT_DIR/../../llamacpp-tq"
    "$SCRIPT_DIR/../../llamaccp-tq/llama-cpp-turboquant"
    "$SCRIPT_DIR/../../llamaccp-tq"
  )

  REPO_DIR=""
  for c in "${CANDIDATES[@]}"; do
    if [[ -f "$c/CMakeLists.txt" ]]; then
      REPO_DIR="$c"
      break
    fi
  done
fi

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-a100}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/bin/llama-server}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_CANDIDATES=(
    "$REPO_DIR/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$(dirname "$REPO_DIR")/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/llamaccp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/../llamaccp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SCRIPT_DIR/../../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SCRIPT_DIR/../../llamaccp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  )

  MODEL_PATH=""
  for m in "${MODEL_CANDIDATES[@]}"; do
    if [[ -f "$m" ]]; then
      MODEL_PATH="$m"
      break
    fi
  done

  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$SUBMIT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-2}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

GUI_HOST="localhost"
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "0.0.0.0" ]]; then
  GUI_HOST="$HOST"
fi
GUI_URL="http://${GUI_HOST}:${PORT}"
CHAT_API_URL="${GUI_URL}/v1/chat/completions"
NODE_HOST="${SLURMD_NODENAME:-$(hostname -s)}"
TUNNEL_CMD="ssh deucalion -L ${LOCAL_TUNNEL_PORT}:${NODE_HOST}:${PORT}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  echo "SLURM_SUBMIT_DIR was: $SUBMIT_DIR" >&2
  echo "Submit from project root or pass REPO_DIR=/path/to/llama-cpp-turboquant" >&2
  exit 1
fi

cd "$REPO_DIR"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing llama-server binary: $SERVER_BIN" >&2
  echo "Submit A100 build first: sbatch slurm/config/build-llama-a100.sh" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  echo "If needed, submit with: --export=ALL,MODEL_PATH=/absolute/path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" >&2
  exit 1
fi

if [[ -z "$CACHE_TYPE_K" || -z "$CACHE_TYPE_V" ]]; then
  HELP_TEXT="$($SERVER_BIN -h 2>&1 || true)"
  if [[ "$HELP_TEXT" == *"turbo3_0"* ]]; then
    CACHE_TYPE_K="${CACHE_TYPE_K:-turbo3_0}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3_0}"
  elif [[ "$HELP_TEXT" == *"turbo3"* ]]; then
    CACHE_TYPE_K="${CACHE_TYPE_K:-turbo3}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3}"
  else
    CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
    echo "[WARN] Turbo cache type not available in this binary; falling back to f16." >&2
  fi
fi

echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Node: $NODE_HOST"
echo "[INFO] Threads: $THREADS"
echo "[INFO] GUI: $GUI_URL"
echo "[INFO] Chat API: $CHAT_API_URL"
echo "[INFO] Cache types: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "[INFO] One-hop tunnel: $TUNNEL_CMD"
echo "[INFO] Open in browser: http://localhost:${LOCAL_TUNNEL_PORT}"

SRUN_BASE=(srun -n1 --gpus=1)

exec "${SRUN_BASE[@]}" "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  -ngl "$N_GPU_LAYERS" \
  -c "$CTX_SIZE" \
  -np "$PARALLEL_REQUESTS" \
  --flash-attn "$FLASH_ATTN" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V"
