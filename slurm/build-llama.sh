#!/usr/bin/env bash
#SBATCH -A f202500010hpcvlabuminhoa
#SBATCH -p normal-arm
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --acctg-freq=energy=10

set -euo pipefail

# --------- LOAD MODULES -----------------
echo "[STARTING] Loading modules"
modules=(
    "GCC/13.3.0"
  "cmake/3.21.3"
)

ml purge

for module in "${modules[@]}"; do
    echo "[LOAD_MODULE] $module"
    ml "$module"
done

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -z "${REPO_DIR:-}" ]]; then
  CANDIDATES=(
    "$SUBMIT_DIR/llamacpp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/llamaccp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/../llamacpp-tq/llama-cpp-turboquant"
    "$SUBMIT_DIR/../llamaccp-tq/llama-cpp-turboquant"
  )

  REPO_DIR=""
  for c in "${CANDIDATES[@]}"; do
    if [[ -d "$c" ]]; then
      REPO_DIR="$c"
      break
    fi
  done
fi

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_CUDA="${ENABLE_CUDA:-0}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  echo "SLURM_SUBMIT_DIR was: $SUBMIT_DIR" >&2
  echo "Submit from project root or pass REPO_DIR=/path/to/llama-cpp-turboquant" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake command not found after module load. Check module availability with: module spider cmake" >&2
  exit 1
fi

cd "$REPO_DIR"

CMAKE_ARGS=(
  -S .
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ "$ENABLE_CUDA" == "1" ]]; then
  CMAKE_ARGS+=( -DGGML_CUDA=ON )
fi

# TurboQuant support is already compiled in this fork.
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$SLURM_CPUS_PER_TASK" --target llama-server

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "Build completed, but llama-server binary was not found at expected path." >&2
  exit 1
fi
