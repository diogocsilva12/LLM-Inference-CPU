#!/bin/bash
#SBATCH --job-name=setup-mlc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:15:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=dev-arm
#SBATCH --output=slurm_logs/mlc_out_%j
#SBATCH --error=slurm_logs/mlc_error_%j
#SBATCH --exclusive

set -euo pipefail

# 1. Carregar a versão de Python segura
source /share/env/module_select.sh
module purge
module load Python/3.12.3-GCCcore-13.3.0

PROJECT=/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion
cd "$PROJECT"

# 2. Destruir o ambiente antigo e criar um novo e limpo
VENV=.venv-arm-mcl
echo "A limpar ambiente antigo..."
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"

# 3. Atualizar o PIP e ferramentas base
pip install --upgrade pip wheel setuptools

# 4. Instalar o MLC-LLM
# A documentação do MLC recomenda usar o canal deles para ter as versões mais recentes (nightly)
echo "A instalar MLC-LLM..."
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

# Se o comando acima falhar porque eles não têm "wheels" para Linux ARM (AArch64),
# o pip vai automaticamente tentar compilar a partir do código-fonte, ou podemos tentar a via normal:
# pip install mlc-llm

# 5. O Grande Teste
echo "A testar importação..."
python -c "import mlc_llm; print('SUCESSO! O MLC-LLM está vivo e mora em:', mlc_llm.__path__)"
