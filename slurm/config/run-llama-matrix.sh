#!/usr/bin/env bash
set -euo pipefail

# Target options:
# - auto (default): detect from node
# - arm-cpu
# - x86-cpu
# - a100-cuda
TARGET="${TARGET:-auto}"

ARCH="$(uname -m)"
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi

if [[ "$TARGET" == "auto" ]]; then
  if [[ "$HAS_NVIDIA" == "1" ]]; then
    TARGET="a100-cuda"
  elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    TARGET="arm-cpu"
  else
    TARGET="x86-cpu"
  fi
fi

echo "[INFO] Run target: $TARGET"
echo "[INFO] Node arch: $ARCH"
echo "[INFO] NVIDIA detected: $HAS_NVIDIA"

echo "[STARTING] Loading modules"
modules=(
  "GCC/13.3.0"
  "cmake/3.21.3"
)
if [[ "$TARGET" == "a100-cuda" ]]; then
  modules+=("CUDA")
fi

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

case "$TARGET" in
  arm-cpu)
    BUILD_SUBDIR="build-matrix/arm64-cpu"
    DEFAULT_NGL=0
    ;;
  x86-cpu)
    BUILD_SUBDIR="build-matrix/x86_64-cpu"
    DEFAULT_NGL=0
    ;;
  a100-cuda)
    BUILD_SUBDIR="build-matrix/x86_64-cuda"
    DEFAULT_NGL=-1
    ;;
  *)
    echo "Unsupported TARGET: $TARGET" >&2
    exit 1
    ;;
esac

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/$BUILD_SUBDIR}"
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
N_GPU_LAYERS="${N_GPU_LAYERS:-$DEFAULT_NGL}"
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
  echo "Set REPO_DIR to the llama.cpp project path." >&2
  exit 1
fi

cd "$REPO_DIR"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing llama-server binary: $SERVER_BIN" >&2
  echo "Build first with build-llama-matrix.sh for TARGET=$TARGET" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/absolute/path/to/model.gguf" >&2
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
echo "[INFO] Build dir: $BUILD_DIR"
echo "[INFO] Node: $NODE_HOST"
echo "[INFO] Threads: $THREADS"
echo "[INFO] GUI: $GUI_URL"
echo "[INFO] Chat API: $CHAT_API_URL"
echo "[INFO] Cache types: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "[INFO] One-hop tunnel: $TUNNEL_CMD"
echo "[INFO] Open in browser: http://localhost:${LOCAL_TUNNEL_PORT}"

CMD=(
  "$SERVER_BIN"
  -m "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --threads "$THREADS"
  --threads-batch "$THREADS"
  -ngl "$N_GPU_LAYERS"
  -c "$CTX_SIZE"
  -np "$PARALLEL_REQUESTS"
  --flash-attn "$FLASH_ATTN"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
)

if [[ "$TARGET" == "a100-cuda" ]]; then
  exec srun -n1 --gpus=1 "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
