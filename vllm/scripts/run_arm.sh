#!/bin/bash
#SBATCH --job-name=arm-venv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:06:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=dev-arm
#SBATCH --output=slurm_logs/job_out_%j
#SBATCH --error=slurm_logs/job_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.13.1-GCCcore-14.2.0
module load numactl/2.0.19-GCCcore-14.2.0 
module load gperftools/2.16-GCCcore-14.2.0


INSTALL=$(git rev-parse --show-toplevel)
VLLM_SRC="$INSTALL/vllm/vllm"

cd "$INSTALL"
VENV=.venv-vllm
source "$VENV/bin/activate"

export OMP_NUM_THREADS=48
export VLLM_CPU_OMP_THREADS_BIND=auto
export LD_PRELOAD=/eb/aarch64/software/GCCcore/14.2.0/lib64/libstdc++.so.6

cd "$VLLM_SRC"
python ../scripts/test.py