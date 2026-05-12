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

echo "[INFO] Build target: $TARGET"
echo "[INFO] Node arch: $ARCH"
echo "[INFO] NVIDIA detected: $HAS_NVIDIA"

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
if [[ "$TARGET" == "arm-cpu" ]]; then
  DEFAULT_GCC_MODULE_CANDIDATES="GCC/13.3.0 GCCcore/13.3.0"
else
  DEFAULT_GCC_MODULE_CANDIDATES="GCC/15.2.0 GCC/14.3.0 GCC/14.2.0 GCC/13.3.0 GCC/13.2.0 GCC/12.3.0 GCC/12.2.0 GCC/11.3.0 GCC/10.3.0 GCC/8.3.0 GCCcore/15.2.0 GCCcore/14.3.0 GCCcore/14.2.0 GCCcore/13.3.0 GCCcore/13.2.0 GCCcore/12.3.0 GCCcore/12.2.0 GCCcore/11.3.0 GCCcore/10.3.0 GCCcore/8.3.0"
fi
GCC_MODULE_CANDIDATES="${GCC_MODULE_CANDIDATES:-$DEFAULT_GCC_MODULE_CANDIDATES}"
CMAKE_MODULE_CANDIDATES="${CMAKE_MODULE_CANDIDATES:-CMake/4.2.1 CMake/4.0.3 CMake/3.31.8 CMake/3.31.3 CMake/3.29.3 CMake/3.27.6 CMake/3.26.3 CMake/3.24.3 CMake/3.23.1 cmake/3.21.3}"
EXTRA_MODULES=()
if [[ "$TARGET" == "a100-cuda" ]]; then
  EXTRA_MODULES+=("CUDA")
fi

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

for module_name in "${EXTRA_MODULES[@]}"; do
  echo "[LOAD_MODULE] $module_name"
  if command -v module >/dev/null 2>&1; then
    module --ignore_cache load "$module_name"
  else
    ml "$module_name"
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
    "$SCRIPT_DIR/../llamacpp-tq/llama-cpp-turboquant"
    "$SCRIPT_DIR/../llamacpp-tq"
    "$SCRIPT_DIR/../llamaccp-tq/llama-cpp-turboquant"
    "$SCRIPT_DIR/../llamaccp-tq"
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
  echo "Set REPO_DIR to the llama.cpp project path." >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake command not found after module load" >&2
  exit 1
fi

BUILD_TYPE="${BUILD_TYPE:-Release}"
read -r -a BUILD_TARGETS <<< "${BUILD_TARGETS:-llama-cli llama-server llama-bench llama-perplexity}"
case "$TARGET" in
  arm-cpu)
    BUILD_SUBDIR="build-matrix/arm64-cpu"
    ENABLE_CUDA=0
    ;;
  x86-cpu)
    BUILD_SUBDIR="build-matrix/x86_64-cpu"
    ENABLE_CUDA=0
    ;;
  a100-cuda)
    BUILD_SUBDIR="build-matrix/x86_64-cuda"
    ENABLE_CUDA=1
    CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80-real}"
    ;;
  *)
    echo "Unsupported TARGET: $TARGET" >&2
    exit 1
    ;;
esac

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/$BUILD_SUBDIR}"

if [[ "$ENABLE_CUDA" == "1" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  CACHED_CUDA_ARCHS="$(grep '^CMAKE_CUDA_ARCHITECTURES:STRING=' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "$CACHED_CUDA_ARCHS" && "$CACHED_CUDA_ARCHS" != "$CUDA_ARCHITECTURES" ]]; then
    echo "[INFO] Cleaning stale CUDA build dir due to arch change: ${CACHED_CUDA_ARCHS} -> ${CUDA_ARCHITECTURES}"
    rm -rf "$BUILD_DIR"
  fi
fi

CMAKE_ARGS=(
  -S .
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)
if [[ "$ENABLE_CUDA" == "1" ]]; then
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
  )
fi
if command -v ccache >/dev/null 2>&1; then
  CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

cd "$REPO_DIR"

echo "[INFO] Repo: $REPO_DIR"
echo "[INFO] Build dir: $BUILD_DIR"
echo "[INFO] Build type: $BUILD_TYPE"
if [[ "$ENABLE_CUDA" == "1" ]]; then
  echo "[INFO] CUDA architectures: $CUDA_ARCHITECTURES"
fi
echo "[INFO] Incremental build enabled (reuses existing build dir)"

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target "${BUILD_TARGETS[@]}"

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "[OK] Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "[ERROR] Build completed, but llama-server was not found in $BUILD_DIR/bin" >&2
  exit 1
fi
