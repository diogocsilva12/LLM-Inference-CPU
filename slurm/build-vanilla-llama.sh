#!/usr/bin/env bash
#SBATCH -A f202500001hpcvlabepicurea
#SBATCH -p normal-arm
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --acctg-freq=energy=10

set -euo pipefail

ensure_module_commands() {
  if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
    return
  fi
  if [[ -f /etc/profile ]]; then
    # Load site shell setup in non-login batch shells.
    source /etc/profile
  fi
  if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
    return
  fi
  for init_script in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
    if [[ -f "$init_script" ]]; then
      source "$init_script"
      break
    fi
  done
}

ensure_module_commands
if ! command -v module >/dev/null 2>&1 && ! command -v ml >/dev/null 2>&1; then
  echo "Neither 'module' nor 'ml' is available in this batch shell on $(hostname)." >&2
  exit 1
fi

echo "[STARTING] Loading modules"
GCC_MODULE_CANDIDATES="${GCC_MODULE_CANDIDATES:-GCC/13.3.0 GCCcore/13.3.0}"
CMAKE_MODULE_CANDIDATES="${CMAKE_MODULE_CANDIDATES:-CMake/4.2.1 CMake/4.0.3 CMake/3.31.8 CMake/3.31.3 CMake/3.29.3 CMake/3.27.6 CMake/3.26.3 CMake/3.24.3 CMake/3.23.1 cmake/3.21.3}"

if command -v module >/dev/null 2>&1; then
  module --ignore_cache purge
else
  ml purge
fi

LOADED_GCC_MODULE=""
for gcc_module in $GCC_MODULE_CANDIDATES; do
  if command -v module >/dev/null 2>&1; then
    if module --ignore_cache load "$gcc_module" >/dev/null 2>&1; then
      LOADED_GCC_MODULE="$gcc_module"
      break
    fi
  elif ml "$gcc_module" >/dev/null 2>&1; then
    LOADED_GCC_MODULE="$gcc_module"
    break
  fi
done

if [[ -z "$LOADED_GCC_MODULE" ]]; then
  echo "No usable GCC module found. Tried: $GCC_MODULE_CANDIDATES" >&2
  exit 1
fi
echo "[LOAD_MODULE] $LOADED_GCC_MODULE"

LOADED_CMAKE_MODULE=""
for cmake_module in $CMAKE_MODULE_CANDIDATES; do
  if command -v module >/dev/null 2>&1; then
    if module --ignore_cache load "$cmake_module" >/dev/null 2>&1; then
      LOADED_CMAKE_MODULE="$cmake_module"
      break
    fi
  elif ml "$cmake_module" >/dev/null 2>&1; then
    LOADED_CMAKE_MODULE="$cmake_module"
    break
  fi
done

if [[ -z "$LOADED_CMAKE_MODULE" ]]; then
  echo "No usable CMake module found. Tried: $CMAKE_MODULE_CANDIDATES" >&2
  exit 1
fi
echo "[LOAD_MODULE] $LOADED_CMAKE_MODULE"

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

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Vanilla llama.cpp repository not found: $REPO_DIR" >&2
  echo "Run: bash slurm/clone-vanilla-llama.sh" >&2
  echo "Or pass REPO_DIR=/path/to/llama.cpp" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake command not found after module load. Check module availability with: module spider cmake" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-slurm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
read -r -a BUILD_TARGETS <<< "${BUILD_TARGETS:-llama-server llama-bench llama-perplexity}"

cd "$REPO_DIR"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target "${BUILD_TARGETS[@]}"

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "Build completed, but llama-server binary was not found at expected path." >&2
  exit 1
fi
