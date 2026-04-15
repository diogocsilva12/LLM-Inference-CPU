#!/usr/bin/env bash
#SBATCH --job-name=build-llamacpp-tq
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

# Update these for your cluster if needed.
# SBATCH --partition=compute
# SBATCH --account=your-account
# module purge
# module load gcc/13 cmake/3.29

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_CUDA="${ENABLE_CUDA:-0}"

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
