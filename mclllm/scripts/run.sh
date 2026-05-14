#!/bin/bash
#SBATCH --job-name=mlc-benchmark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --output=slurm_logs/bench_out_%j
#SBATCH --error=slurm_logs/bench_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.13.1-GCCcore-14.2.0

PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"
VENV="$PROJECT/.venv-x86-mcl"

source "$VENV/bin/activate"

cd "$PROJECT"

echo "========================================================="
echo ">>> A INICIAR BENCHMARK MLC-LLM"
echo ">>> Nó alocado: $SLURMD_NODENAME"
echo ">>> Data/Hora: $(date)"
echo "========================================================="

python benchmark.py

echo "========================================================="
echo ">>> BENCHMARK TERMINADO COM SUCESSO!"
echo ">>> Data/Hora: $(date)"
echo "========================================================="