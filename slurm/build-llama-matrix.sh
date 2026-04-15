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

echo "[STARTING] Loading modules"
modules=(
  "GCC/13.3.0"
  "cmake/3.21.3"
)
if [[ "$TARGET" == "a100-cuda" ]]; then
  modules+=("CUDA")
fi

ml purge
for module in "${modules[@]}"; do
  echo "[LOAD_MODULE] $module"
  ml "$module"
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
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target llama-server

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "[OK] Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "[ERROR] Build completed, but llama-server was not found in $BUILD_DIR/bin" >&2
  exit 1
fi
