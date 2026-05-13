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

# 1. Carregar Módulos do Sistema
source /share/env/module_select.sh
module purge
module load Python/3.13.1-GCCcore-14.2.0

# 2. Definir Variáveis de Ambiente e Caminhos
PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"
VENV="$PROJECT/.venv-x86-mcl"

# 3. Ativar o Ambiente Virtual (A magia do LD_LIBRARY_PATH acontece aqui!)
source "$VENV/bin/activate"

# 4. Navegar para a diretoria onde está o código Python
cd "$PROJECT"

# 5. Imprimir estado e executar o script de JSON
echo "========================================================="
echo ">>> A INICIAR BENCHMARK MLC-LLM"
echo ">>> Nó alocado: $SLURMD_NODENAME"
echo ">>> Data/Hora: $(date)"
echo "========================================================="

# Correr o ficheiro Python que vai ler o JSON, processar os Batches e cuspir o novo JSON
python benchmark.py

echo "========================================================="
echo ">>> BENCHMARK TERMINADO COM SUCESSO!"
echo ">>> Data/Hora: $(date)"
echo "========================================================="