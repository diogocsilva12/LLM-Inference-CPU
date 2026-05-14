#!/usr/bin/env bash
#SBATCH --job-name=vllm-sweep-x86
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --exclusive

set -euo pipefail

# 1. Setup do Ambiente
source /share/env/module_select.sh
module purge
module load Python/3.12.3-GCCcore-13.3.0

PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"
source "$PROJECT/.venv-x86/bin/activate"

# Instalar pacotes para o cliente
pip install -q openai numpy


MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" 
MODEL_LABEL="llama3.1-8b"
ENGINE="vllm-x86"
PORT=8000
CONCURRENCY_LIST="1 4 8 16 32"
PROMPTS_FILE="$PROJECT/prompts/track_a_prompts.json"

export OMP_NUM_THREADS=128

# [!!! NOVAS LINHAS AQUI !!!]
# Obriga o download a ir para o disco do Projeto (foge da quota do teu /home/)
export HF_HOME="$PROJECT/hf_cache"

# Token do Hugging Face para teres permissão de sacar o Llama 3.1
# (Substitui 'hf_AQUI_O_TEU_TOKEN_REAL' pelo teu token do site Hugging Face)
export HF_TOKEN="hf_AQUI_O_TEU_TOKEN_REAL"
#

# 3. Criar a hierarquia de diretorias majestosa
MEASUREMENTS_DIR="$PROJECT/measurements/${SLURM_JOB_ID}-vllm-server-sweep-x86"
mkdir -p "$MEASUREMENTS_DIR"

echo "========================================================="
echo "🚀 SWEEP BENCHMARK INICIADO"
echo "📂 Pasta de saída: $MEASUREMENTS_DIR"
echo "========================================================="

# Loop da Concorrência
for C in $CONCURRENCY_LIST; do
    echo "[!] A testar Concorrência: $C"
    
    # Criar a pasta exata para este teste (ex: t_128_c_2048_p_4...)
    RUN_DIR="$MEASUREMENTS_DIR/concurrency/$ENGINE/$MODEL_LABEL/t_128_c_2048_p_${C}"
    mkdir -p "$RUN_DIR"
    
    # Gerar o system.txt
    lscpu > "$RUN_DIR/system.txt"

    # Iniciar o Servidor vLLM
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --enforce-eager \
        --max-model-len 2048 > "$RUN_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
    
    # Iniciar o Monitor de Recursos em Background (grava CPU/RAM a cada segundo)
    echo "timestamp,cpu_percent,rss_kb" > "$RUN_DIR/resources.csv"
    while kill -0 $SERVER_PID 2>/dev/null; do
        TIMESTAMP=$(date -Iseconds)
        # Extrai CPU% e Memória RSS do processo do servidor
        STATS=$(ps -p $SERVER_PID -o %cpu,rss | tail -n 1 | awk '{print $1","$2}')
        echo "$TIMESTAMP,$STATS" >> "$RUN_DIR/resources.csv"
        sleep 1
    done &
    MONITOR_PID=$!

    # Esperar o servidor acordar
    while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do sleep 5; done
    
    # Correr o Cliente (usando o comando `time` nativo do Linux para recriar o client.time.log)
    /usr/bin/time -v -o "$RUN_DIR/client.time.log" python "$PROJECT/benchmark_cliente.py" \
        --url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL" \
        --prompts "$PROMPTS_FILE" \
        --out-jsonl "$RUN_DIR/requests.jsonl" \
        --out-summary "$RUN_DIR/summary.json" \
        --concurrency $C

    # Limpar a casa para a próxima iteração
    kill $SERVER_PID
    kill $MONITOR_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
done

echo "========================================================="
echo "✅ SWEEP TERMINADO COM SUCESSO!"
echo "========================================================="