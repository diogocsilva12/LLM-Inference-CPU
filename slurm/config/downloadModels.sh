#!/usr/bin/env bash
#SBATCH --job-name=download-models-hf
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
set -euo pipefail

USER=$(whoami)
PROJECT="/projects/F202500010HPCVLABUMINHO/${USER}/heretic-deucalion"

export HF_HOME="$PROJECT/hf_cache"
HF_TOKEN="INSERIR_TOKEN_AQUI"
MLC_DIR="$PROJECT/Llama3.1-MLC"
GGUF_DIR="$PROJECT/gguf_models"

mkdir -p "$GGUF_DIR"
mkdir -p "slurm_logs"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ ERRO: HF_TOKEN não definido. Corre 'export HF_TOKEN=...' antes do sbatch."
    exit 1
fi


echo "[!] A garantir que o huggingface_hub está instalado..."
pip install -q -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1 

echo "---------------------------------------------------------"
echo ">>> [1/3] A descarregar Llama-3.1-8B-Instruct (Para o vLLM)"
echo "---------------------------------------------------------"

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --token "$HF_TOKEN"



if [ ! -d "$MLC_DIR" ]; then
    huggingface-cli download mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC \
        --local-dir "$MLC_DIR" \
        --token "$HF_TOKEN"
else
    echo "✅ A pasta $MLC_DIR já existe. Download do MLC ignorado."
fi

echo "A sacar SmolLM2-360M..."
huggingface-cli download unsloth/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q4_K_M.gguf \
    --local-dir "$GGUF_DIR" \
    --local-dir-use-symlinks False

echo "A sacar Gemma-4B..."
huggingface-cli download bartowski/gemma-2-4b-it-GGUF gemma-2-4b-it-Q4_K_M.gguf \
    --local-dir "$GGUF_DIR" \
    --local-dir-use-symlinks False


