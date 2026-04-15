#!/usr/bin/env bash
#SBATCH -A f202500010hpcvlabuminhoa
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

cd "$REPO_DIR"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-8}" --target llama-server

if [[ -x "$BUILD_DIR/bin/llama-server" ]]; then
  echo "Build successful: $BUILD_DIR/bin/llama-server"
else
  echo "Build completed, but llama-server binary was not found at expected path." >&2
  exit 1
fi
