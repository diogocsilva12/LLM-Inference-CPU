#!/usr/bin/env bash
#SBATCH -A f202500001hpcvlabepicurea
#SBATCH -p normal-arm
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --acctg-freq=energy=10

set -euo pipefail

echo "[STARTING] Loading modules"
modules=(
  "GCC/13.3.0"
  "cmake/3.21.3"
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
    "$SUBMIT_DIR/llamacpp-vanilla/llama.cpp"
    "$SUBMIT_DIR/../llamacpp-vanilla/llama.cpp"
    "$SCRIPT_DIR/../llamacpp-vanilla/llama.cpp"
  )

  REPO_DIR=""
  for c in "${CANDIDATES[@]}"; do
    if [[ -f "$c/CMakeLists.txt" ]]; then
      REPO_DIR="$c"
      break
    fi
  done
fi

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/bin/llama-server}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_CANDIDATES=(
    "$REPO_DIR/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$(dirname "$REPO_DIR")/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/llamacpp-vanilla/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/../llamacpp-vanilla/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SCRIPT_DIR/../llamacpp-vanilla/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SCRIPT_DIR/../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
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
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
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
  echo "Vanilla repository directory not found: $REPO_DIR" >&2
  echo "Run: bash slurm/clone-vanilla-llama.sh" >&2
  echo "Or pass REPO_DIR=/path/to/llama.cpp" >&2
  exit 1
fi

cd "$REPO_DIR"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing llama-server binary: $SERVER_BIN" >&2
  echo "Submit build job first: sbatch slurm/build-vanilla-llama.sh" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  echo "If needed, submit with: --export=ALL,MODEL_PATH=/absolute/path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" >&2
  exit 1
fi

echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Node: $NODE_HOST"
echo "[INFO] Threads: $THREADS"
echo "[INFO] GUI: $GUI_URL"
echo "[INFO] Chat API: $CHAT_API_URL"
echo "[INFO] Cache types: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "[INFO] One-hop tunnel: $TUNNEL_CMD"
echo "[INFO] Open in browser: http://localhost:${LOCAL_TUNNEL_PORT}"

exec "$SERVER_BIN" \
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
