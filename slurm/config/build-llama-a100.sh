#!/usr/bin/env bash
#SBATCH --job-name=build-llama-a100
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

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

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  echo "SLURM_SUBMIT_DIR was: $SUBMIT_DIR" >&2
  echo "Submit from project root or pass REPO_DIR=/path/to/llama-cpp-turboquant" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake command not found after module load" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-a100}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80-real}"
read -r -a BUILD_TARGETS <<< "${BUILD_TARGETS:-llama-server llama-bench llama-perplexity}"

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  CACHED_CUDA_ARCHS="$(grep '^CMAKE_CUDA_ARCHITECTURES:STRING=' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "$CACHED_CUDA_ARCHS" && "$CACHED_CUDA_ARCHS" != "$CUDA_ARCHITECTURES" ]]; then
    echo "[INFO] Cleaning stale CUDA build dir due to arch change: ${CACHED_CUDA_ARCHS} -> ${CUDA_ARCHITECTURES}"
    rm -rf "$BUILD_DIR"
  fi
fi

cd "$REPO_DIR"

echo "[INFO] CUDA architectures: $CUDA_ARCHITECTURES"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGGML_CUDA=ON

cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target "${BUILD_TARGETS[@]}"

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "Build completed, but llama-server binary was not found at expected path." >&2
  exit 1
fi
