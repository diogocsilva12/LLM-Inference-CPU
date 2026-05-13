#!/bin/bash
#SBATCH --job-name=x86-venv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:06:00
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --output=slurm_logs/job_out_%j
#SBATCH --error=slurm_logs/job_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.12.3-GCCcore-13.3.0
#module load Python/3.13.1-GCCcore-14.2.0
#module load AOCL-BLAS/5.0-GCC-14.2.0
module load gperftools/2.16-GCCcore-13.3.0 
#module load gperftools/2.16-GCCcore-14.2.0

INSTALL=$(git rev-parse --show-toplevel)
VLLM_SRC="$INSTALL/vllm/vllm"


cd "$INSTALL"
VENV=.venv-x86
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

export CC=$(which gcc)
export CXX=$(which g++)
export CXXFLAGS="-O3 -march=znver2"
export CFLAGS="-O3 -march=znver2"

# Base tooling
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy

# PyTorch CPU + restantes deps — versão ditada pelo vLLM
pip install -v -r "$VLLM_SRC/requirements/cpu.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Build vLLM
# Build vLLM (Forçando compilação bruta e visível)
cd "$VLLM_SRC"


rm -rf .deps build _skbuild dist/ 

# 1. Usar --no-deps para compilar APENAS a wheel do vLLM, poupando tempo e evitando o erro do PyTorch
VLLM_TARGET_DEVICE=cpu pip wheel . --no-build-isolation --no-deps -w dist/

# 2. Instalar a wheel gerada. Aqui sim, passamos o link do repositório para o pip validar silenciosamente que o PyTorch já lá está.
pip install dist/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/cpu
