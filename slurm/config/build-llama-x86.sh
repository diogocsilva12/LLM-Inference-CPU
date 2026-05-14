#!/usr/bin/env bash
#SBATCH --job-name=build-llama-x86
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

BUILD_MATRIX_SCRIPT="${BUILD_MATRIX_SCRIPT:-}"
if [[ -z "$BUILD_MATRIX_SCRIPT" ]]; then
  for candidate in \
    "$SUBMIT_DIR/slurm/config/build-llama-matrix.sh" \
    "$SUBMIT_DIR/build-llama-matrix.sh" \
    "$SCRIPT_DIR/build-llama-matrix.sh"; do
    if [[ -f "$candidate" ]]; then
      BUILD_MATRIX_SCRIPT="$candidate"
      break
    fi
  done
fi

if [[ -z "$BUILD_MATRIX_SCRIPT" ]]; then
  echo "Unable to locate build-llama-matrix.sh." >&2
  echo "SLURM_SUBMIT_DIR was: $SUBMIT_DIR" >&2
  echo "Submit from the project root or set BUILD_MATRIX_SCRIPT=/path/to/slurm/config/build-llama-matrix.sh." >&2
  exit 2
fi

export TARGET="x86-cpu"

exec "$BUILD_MATRIX_SCRIPT"
