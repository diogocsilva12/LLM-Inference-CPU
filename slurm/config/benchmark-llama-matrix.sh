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

echo "[INFO] Benchmark target: $TARGET"
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
    "$SUBMIT_DIR/../llamacpp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/../llamacpp-tq"
    "$SCRIPT_DIR/../../llamacpp-tq/llama-cpp-turboquant"
    "$SCRIPT_DIR/../../llamacpp-tq"
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
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/llama-bench}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_CANDIDATES=(
    "$REPO_DIR/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$(dirname "$REPO_DIR")/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SUBMIT_DIR/../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$SCRIPT_DIR/../../llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
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

N_GPU_LAYERS="${N_GPU_LAYERS:-$DEFAULT_NGL}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
FLASH_ATTN="${FLASH_ATTN:-1}"
REPETITIONS="${REPETITIONS:-5}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512,2048,8192}"
GEN_TOKENS="${GEN_TOKENS:-128,512}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_SWEEP="${CACHE_SWEEP:-f16:f16 turbo2:turbo2 turbo3:turbo3 turbo4:turbo4}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUBMIT_DIR/benchmarks/$(date +%Y%m%d-%H%M%S)-$TARGET}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  echo "Set REPO_DIR to the llama.cpp project path." >&2
  exit 1
fi

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Missing llama-bench binary: $BENCH_BIN" >&2
  echo "Build first with: BUILD_TARGETS=\"llama-server llama-bench llama-perplexity\" slurm/config/submit-llama.sh build $TARGET" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/absolute/path/to/model.gguf" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
HELP_TEXT="$($BENCH_BIN -h 2>&1 || true)"

echo "[INFO] Repo: $REPO_DIR"
echo "[INFO] Build dir: $BUILD_DIR"
echo "[INFO] Benchmark bin: $BENCH_BIN"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Threads: $THREADS"
echo "[INFO] Prompt tokens: $PROMPT_TOKENS"
echo "[INFO] Generation tokens: $GEN_TOKENS"
echo "[INFO] Cache sweep: $CACHE_SWEEP"
echo "[INFO] Output dir: $OUTPUT_DIR"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "target=$TARGET"
  echo "node=${SLURMD_NODENAME:-$(hostname -s)}"
  echo "arch=$ARCH"
  echo "threads=$THREADS"
  echo "model=$MODEL_PATH"
  echo "build_dir=$BUILD_DIR"
  "$BENCH_BIN" --list-devices 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
} > "$OUTPUT_DIR/system.txt"

for pair in $CACHE_SWEEP; do
  CACHE_TYPE_K="${pair%%:*}"
  CACHE_TYPE_V="${pair##*:}"
  if [[ "$HELP_TEXT" != *"$CACHE_TYPE_K"* || "$HELP_TEXT" != *"$CACHE_TYPE_V"* ]]; then
    echo "[WARN] Skipping unsupported cache pair K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
    continue
  fi

  OUT_FILE="$OUTPUT_DIR/llama-bench-k_${CACHE_TYPE_K}-v_${CACHE_TYPE_V}.${OUTPUT_FORMAT}"
  echo "[INFO] Running llama-bench K=$CACHE_TYPE_K V=$CACHE_TYPE_V -> $OUT_FILE"

  CMD=(
    "$BENCH_BIN"
    -m "$MODEL_PATH"
    -p "$PROMPT_TOKENS"
    -n "$GEN_TOKENS"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -t "$THREADS"
    -ngl "$N_GPU_LAYERS"
    -fa "$FLASH_ATTN"
    -ctk "$CACHE_TYPE_K"
    -ctv "$CACHE_TYPE_V"
    -r "$REPETITIONS"
    -o "$OUTPUT_FORMAT"
  )

  if [[ "$TARGET" == "a100-cuda" ]]; then
    /usr/bin/time -v srun -n1 --gpus=1 "${CMD[@]}" > "$OUT_FILE" 2> "$OUT_FILE.stderr"
  else
    /usr/bin/time -v "${CMD[@]}" > "$OUT_FILE" 2> "$OUT_FILE.stderr"
  fi
done

echo "[OK] Benchmark outputs written to $OUTPUT_DIR"
