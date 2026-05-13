#!/bin/bash
#SBATCH --job-name=test-mcl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=dev-arm
#SBATCH --output=slurm_logs/job_out_%j
#SBATCH --error=slurm_logs/job_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.13.1-GCCcore-14.2.0


PROJECT=/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion

cd "$PROJECT"
VENV=.venv-arm-mcl
source "$VENV/bin/activate"

export OMP_NUM_THREADS=48


python -c "import mlc_llm; print(mlc_llm)"