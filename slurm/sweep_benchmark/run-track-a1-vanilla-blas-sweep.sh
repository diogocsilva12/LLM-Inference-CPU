#!/usr/bin/env bash
#SBATCH --job-name=a1-vanilla-blas-sweep
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
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

BASE_SWEEP_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep.sh"
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh"
if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base sweep script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
  echo "Missing BLAS wrapper script: $WRAPPER_SCRIPT" >&2
  exit 2
fi

VANILLA_REPO="${VANILLA_REPO:-$PROJECT_DIR/llamacpp-vanilla/llama.cpp}"
OPENBLAS_MODULE="${OPENBLAS_MODULE:-OpenBLAS/0.3.26-GCC-13.3.0-Fujitsu}"
BLIS_MODULE="${BLIS_MODULE:-BLIS/1.0-GCC-13.3.0}"
OPENBLAS_BUILD_DIR="${OPENBLAS_BUILD_DIR:-$VANILLA_REPO/build-blas/openblas-fujitsu}"
BLIS_BUILD_DIR="${BLIS_BUILD_DIR:-$VANILLA_REPO/build-blas/blis}"
OPENBLAS_PRELOAD="${OPENBLAS_PRELOAD:-/eb/aarch64/software/OpenBLAS/0.3.26-GCC-13.3.0-Fujitsu/lib/libopenblas.so}"
BLIS_PRELOAD="${BLIS_PRELOAD:-/eb/aarch64/software/BLIS/1.0-GCC-13.3.0/lib/libblis.so}"

export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-vanilla-blas-sweep}"
export ENGINE_SPECS="${ENGINE_SPECS:-vanilla-openblas-fujitsu=$VANILLA_REPO;vanilla-blis=$VANILLA_REPO}"
export ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-vanilla-openblas-fujitsu=$OPENBLAS_BUILD_DIR/bin/llama-server;vanilla-blis=$BLIS_BUILD_DIR/bin/llama-server}"
export ENGINE_PRELOAD_SPECS="${ENGINE_PRELOAD_SPECS:-vanilla-openblas-fujitsu=$OPENBLAS_PRELOAD;vanilla-blis=$BLIS_PRELOAD}"
export ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vanilla-openblas-fujitsu=f16:f16;vanilla-blis=f16:f16}"
export AUTO_BUILD="${AUTO_BUILD:-0}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
export PROJECT_DIR
export A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

echo "[INFO] Expected OpenBLAS build module: $OPENBLAS_MODULE"
echo "[INFO] Expected BLIS build module: $BLIS_MODULE"
echo "[INFO] OpenBLAS binary: $OPENBLAS_BUILD_DIR/bin/llama-server"
echo "[INFO] BLIS binary: $BLIS_BUILD_DIR/bin/llama-server"
echo "[INFO] Runtime LD_PRELOAD specs: $ENGINE_PRELOAD_SPECS"

exec "$BASE_SWEEP_SCRIPT" "$@"
