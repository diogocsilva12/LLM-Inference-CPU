#!/usr/bin/env bash
#SBATCH --job-name=run-llama-x86
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TARGET="x86-cpu"
export N_GPU_LAYERS="${N_GPU_LAYERS:-0}"

exec "$SCRIPT_DIR/run-llama-matrix.sh"
