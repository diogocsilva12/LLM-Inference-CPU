#!/usr/bin/env bash
#SBATCH --job-name=build-vanilla-blis
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

load_module() {
  local module_name="$1"
  echo "[LOAD_MODULE] $module_name"
  if command -v module >/dev/null 2>&1; then
    module --ignore_cache load "$module_name"
  elif command -v ml >/dev/null 2>&1; then
    ml "$module_name"
  else
    echo "Neither module nor ml is available in this shell." >&2
    exit 1
  fi
}

if command -v module >/dev/null 2>&1; then
  module --ignore_cache purge
elif command -v ml >/dev/null 2>&1; then
  ml purge
fi

BLIS_MODULE="${BLIS_MODULE:-BLIS/1.0-GCC-13.3.0}"
CMAKE_MODULE="${CMAKE_MODULE:-cmake/3.21.3}"
load_module "$BLIS_MODULE"
load_module "$CMAKE_MODULE"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SUBMIT_DIR}"
if [[ ! -f "$PROJECT_DIR/prompts/track_a_prompts.json" ]]; then
  PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
fi

REPO_DIR="${REPO_DIR:-$PROJECT_DIR/llamacpp-vanilla/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-blas/blis}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
read -r -a BUILD_TARGETS <<< "${BUILD_TARGETS:-llama-server llama-bench}"
read -r -a EXTRA_CMAKE_ARGS <<< "${CMAKE_EXTRA_ARGS:-}"

if [[ ! -f "$REPO_DIR/CMakeLists.txt" ]]; then
  echo "Missing vanilla llama.cpp repository: $REPO_DIR" >&2
  exit 1
fi

CMAKE_ARGS=(
  -S "$REPO_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DGGML_BLAS=ON
  -DGGML_BLAS_VENDOR=FLAME
)
if [[ -n "${EBROOTBLIS:-}" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_PREFIX_PATH="$EBROOTBLIS"
    -DCMAKE_BUILD_RPATH="$EBROOTBLIS/lib"
  )
fi
CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")

echo "[INFO] Repo: $REPO_DIR"
echo "[INFO] Build dir: $BUILD_DIR"
echo "[INFO] BLAS module: $BLIS_MODULE"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target "${BUILD_TARGETS[@]}"

if [[ ! -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "Build finished but missing $BUILD_DIR/bin/llama-server" >&2
  exit 1
fi
echo "[OK] BLIS vanilla build: $BUILD_DIR/bin/llama-server"
