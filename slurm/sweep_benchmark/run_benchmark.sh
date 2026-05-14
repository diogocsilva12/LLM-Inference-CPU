#!/bin/bash
#SBATCH --job-name=vllm-sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --output=slurm_logs/sweep_out_%j
#SBATCH --error=slurm_logs/sweep_error_%j
#SBATCH --exclusive

set -euo pipefail

source /share/env/module_select.sh
module purge
module load Python/3.12.3-GCCcore-13.3.0

PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"

cd "$PROJECT"
mkdir -p slurm_logs

VENV="$PROJECT/.venv-x86"
source "$VENV/bin/activate"

# Variáveis do teu Sweep
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PORT=8000
CONCURRENCY_LIST="1 4 8 16 32"
export OMP_NUM_THREADS=128

PROMPTS_FILE="$PROJECT/prompts/track_a_prompts.json"
MODEL_LABEL="llama3.1-8b"
ENGINE="vllm-x86"

export HF_HOME="$PROJECT/hf_cache"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

# Criar a pasta base para este job
MEASUREMENTS_DIR="$PROJECT/measurements/${SLURM_JOB_ID}-vllm-server-sweep"
mkdir -p "$MEASUREMENTS_DIR"

echo "========================================================="
echo ">>> A INICIAR SWEEP BENCHMARK (vLLM SERVER)"
echo "========================================================="

for CONCURRENCY in $CONCURRENCY_LIST; do
    echo "---------------------------------------------------------"
    echo "[!] A INICIAR TESTE COM CONCORRÊNCIA: $CONCURRENCY"
    
    # Criar a pasta exata para esta métrica (Igual ao do teu amigo)
    RUN_DIR="$MEASUREMENTS_DIR/concurrency/$ENGINE/$MODEL_LABEL/t_128_c_2048_p_${CONCURRENCY}"
    mkdir -p "$RUN_DIR"

    # 1. Iniciar o servidor em background
    # NOTA: Agora o log vai direitinho para a pasta do Run!
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --enforce-eager \
        --max-model-len 2048 > "$RUN_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
    
    # 2. Esperar que o servidor esteja pronto (Loop Seguro)
    echo "A aguardar o arranque do servidor..."
    while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do 
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERRO: Servidor vLLM crashou. Lê $RUN_DIR/server.log"
            exit 1
        fi
        sleep 5 
    done
    echo "✅ Servidor online e pronto!"

    # 3. Correr o VERDADEIRO script cliente!
    /usr/bin/time -v -o "$RUN_DIR/client.time.log" python scripts/benchmark_cliente.py \
        --url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL" \
        --prompts "$PROMPTS_FILE" \
        --out-jsonl "$RUN_DIR/requests.jsonl" \
        --out-summary "$RUN_DIR/summary.json" \
        --concurrency $CONCURRENCY
    
    # 4. Matar o servidor suavemente
    echo "A desligar servidor PID: $SERVER_PID"
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    
    sleep 5
done

echo "========================================================="
echo ">>> SWEEP TERMINADO COM SUCESSO!"
echo "========================================================="
