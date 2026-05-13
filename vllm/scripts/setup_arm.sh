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

INSTALL=/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion
VLLM_SRC="$INSTALL/vllm/vllm"

cd "$INSTALL"
VENV=.venv-vllm
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

export MAX_JOBS=12
export CPATH=$EBROOTNUMACTL/include:$CPATH
export LIBRARY_PATH=$EBROOTNUMACTL/lib:$LIBRARY_PATH
export PATH=/eb/aarch64/software/binutils/2.42-GCCcore-13.3.0/bin:$PATH
export OMP_NUM_THREADS=48
export VLLM_CPU_OMP_THREADS_BIND=auto
export LD_PRELOAD=/eb/aarch64/software/GCCcore/13.3.0/lib64/libstdc++.so.6
export CC=$(which gcc)
export CXX=$(which g++)

# Base tooling
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy

# PyTorch CPU + restantes deps — versão ditada pelo vLLM
pip install -v -r "$VLLM_SRC/requirements/cpu.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Build vLLM
cd "$VLLM_SRC"
rm -rf .deps build _skbuild
VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation