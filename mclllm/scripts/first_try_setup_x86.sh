#!/bin/bash
#SBATCH --job-name=test-mcl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --output=slurm_logs/job_out_%j
#SBATCH --error=slurm_logs/job_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.13.1-GCCcore-14.2.0


PROJECT=/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion

cd "$PROJECT"

VENV=.venv-x86-mcl
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools git-lfs pytest


python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu


python -c "import mlc_llm; print(mlc_llm)"