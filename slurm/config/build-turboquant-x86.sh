#!/usr/bin/env bash
#SBATCH --job-name=build-tq-x86
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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

REPO_DIR="${REPO_DIR:-$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm-x86}"
BUILD_MATRIX_SCRIPT="${BUILD_MATRIX_SCRIPT:-$PROJECT_DIR/slurm/config/build-llama-matrix.sh}"

if [[ ! -f "$REPO_DIR/CMakeLists.txt" ]]; then
  echo "Missing TurboQuant llama.cpp repository: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$BUILD_MATRIX_SCRIPT" ]]; then
  echo "Missing matrix build script: $BUILD_MATRIX_SCRIPT" >&2
  exit 1
fi

export TARGET="x86-cpu"
export PROJECT_DIR
export REPO_DIR
export BUILD_DIR
export BUILD_TARGETS="${BUILD_TARGETS:-llama-cli llama-server llama-bench llama-perplexity}"
export GCC_MODULE_CANDIDATES="${GCC_MODULE_CANDIDATES:-GCC/13.3.0 GCC/12.3.0 GCC/12.2.0 GCC/11.3.0 GCC/10.3.0 GCCcore/13.3.0 GCCcore/12.3.0 GCCcore/12.2.0 GCCcore/11.3.0 GCCcore/10.3.0}"
export CMAKE_MODULE_CANDIDATES="${CMAKE_MODULE_CANDIDATES:-CMake/4.2.1 CMake/4.0.3 CMake/3.31.8 CMake/3.29.3 CMake/3.27.6 CMake/3.26.3 CMake/3.24.3 cmake/3.21.3}"
export CFLAGS="${CFLAGS:-} -g0"
export CXXFLAGS="${CXXFLAGS:-} -g0"
export CMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE:--O3 -DNDEBUG -g0}"
export CMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE:--O3 -DNDEBUG -g0}"

if [[ "${FRESH_BUILD:-1}" == "1" || "${FRESH_BUILD:-1}" == "true" ]]; then
  case "$BUILD_DIR" in
    "$REPO_DIR"/*)
      echo "[INFO] Removing x86 TurboQuant build dir for a fresh compiler/configure pass: $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      ;;
    *)
      echo "Refusing to remove BUILD_DIR outside REPO_DIR: $BUILD_DIR" >&2
      exit 2
      ;;
  esac
fi

echo "[INFO] Project dir: $PROJECT_DIR"
echo "[INFO] TurboQuant repo: $REPO_DIR"
echo "[INFO] Build dir: $BUILD_DIR"
echo "[INFO] Target: $TARGET"
echo "[INFO] GCC candidates: $GCC_MODULE_CANDIDATES"
echo "[INFO] Release C flags: $CMAKE_C_FLAGS_RELEASE"
echo "[INFO] Release CXX flags: $CMAKE_CXX_FLAGS_RELEASE"

exec "$BUILD_MATRIX_SCRIPT"
