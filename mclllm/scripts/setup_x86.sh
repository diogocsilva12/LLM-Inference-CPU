#!/bin/bash
#SBATCH --job-name=test-mcl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
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
export RUSTUP_HOME="/projects/F202500010HPCVLABUMINHO/uminhocp030/.rustup"
export CARGO_HOME="/projects/F202500010HPCVLABUMINHO/uminhocp030/.cargo"

cd "$PROJECT"

VENV=.venv-x86-mcl
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools "cmake>=3.24"  pytest
source "$CARGO_HOME/env"
rustup default stable

cd "$PROJECT"
cd mclllm/mlc-llm

mkdir -p build && cd build
#python ../cmake/gen_cmake_config.py
#cmake .. && make -j $(nproc) && cd ..
cmake ..
make -j $(nproc)
make tvm -j $(nproc) 

cd ../python
pip install -e .
pip install apache-tvm-ffi huggingface_hub hf_transfer
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TVM_DIR="$SITE_PACKAGES/tvm"
echo "export LD_LIBRARY_PATH=\"$TVM_DIR:$TVM_DIR/lib:\$LD_LIBRARY_PATH\"" >> "$VENV/bin/activate"