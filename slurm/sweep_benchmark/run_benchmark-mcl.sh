#!/usr/bin/env bash
#SBATCH --job-name=mlc-sweep-x86
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
module load Python/3.13.1-GCCcore-14.2.0

PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"

# GARANTIR QUE ESTAMOS NA PASTA CERTA
cd "$PROJECT"
mkdir -p slurm_logs

# IMPORTANTE: Usamos o venv do MLC (não o do vLLM)
VENV="$PROJECT/.venv-x86-mcl"
source "$VENV/bin/activate"

# Garantir que o cliente tem o OpenAI instalado neste ambiente também
pip install -q openai numpy

# 2. Configurações da Experiência
# Usamos o modelo pré-compilado do MLC (Quantizado a 4 bits)
MODEL_REPO="mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC"
LOCAL_MODEL_DIR="$PROJECT/Llama3.1-MLC"
PORT=8000
CONCURRENCY_LIST="1 4 8 16 32"
export OMP_NUM_THREADS=128
export TVM_NUM_THREADS=128   
export TVM_BIND_THREADS=1

PROMPTS_FILE="$PROJECT/prompts/track_a_prompts.json"
MODEL_LABEL="llama3.1-8b-mlc"
ENGINE="mlc-x86"

export HF_HOME="$PROJECT/hf_cache"

export MLC_LLM_LIBRARY_PATH="$PROJECT/mclllm/mlc-llm/build/libmlc_llm_module.so"
export LD_LIBRARY_PATH="$PROJECT/mclllm/mlc-llm/build:${LD_LIBRARY_PATH:-}"

# 3. Garantir que o modelo MLC está descarregado localmente
# Como o MLC não saca "automagicamente" pela flag do python como o vllm, fazemos nós.
echo "[!] A verificar ficheiros do modelo MLC..."
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    hf download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"
fi

# 4. Criar a hierarquia de diretorias
MEASUREMENTS_DIR="$PROJECT/measurements/${SLURM_JOB_ID}-mlc-server-sweep"
mkdir -p "$MEASUREMENTS_DIR"

echo "========================================================="
echo "🚀 SWEEP BENCHMARK INICIADO (MLC-LLM SERVER)"
echo "📂 Pasta de saída: $MEASUREMENTS_DIR"
echo "========================================================="

# Loop da Concorrência
for CONCURRENCY in $CONCURRENCY_LIST; do
    echo "---------------------------------------------------------"
    echo "[!] A INICIAR TESTE COM CONCORRÊNCIA: $CONCURRENCY"
    
    # Criar a pasta exata para este teste
    RUN_DIR="$MEASUREMENTS_DIR/concurrency/$ENGINE/$MODEL_LABEL/t_128_c_2048_p_${CONCURRENCY}"
    mkdir -p "$RUN_DIR"
    
    # Gerar o system.txt
    lscpu > "$RUN_DIR/system.txt"

    # Iniciar o Servidor MLC-LLM em background
    python -m mlc_llm serve "$LOCAL_MODEL_DIR" \
        --device cpu \
        --host 127.0.0.1 \
        --port $PORT > "$RUN_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
    
    # Iniciar o Monitor de Recursos em Background
    echo "timestamp,cpu_percent,rss_kb" > "$RUN_DIR/resources.csv"
    while kill -0 $SERVER_PID 2>/dev/null; do
        TIMESTAMP=$(date -Iseconds)
        STATS=$(ps -p $SERVER_PID -o %cpu,rss | tail -n 1 | awk '{print $1","$2}')
        echo "$TIMESTAMP,$STATS" >> "$RUN_DIR/resources.csv"
        sleep 1
    done &
    MONITOR_PID=$!

    # Esperar o servidor acordar com Loop Seguro
    # O MLC pode demorar 1 a 2 mins a compilar a JIT cache na 1ª vez que corre
    echo "A aguardar o arranque do servidor MLC..."
    while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do 
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ ERRO: Servidor MLC-LLM crashou. Lê $RUN_DIR/server.log"
            exit 1
        fi
        sleep 5 
    done
    echo "✅ Servidor MLC online e pronto!"
    
    # Correr o Cliente Python
    /usr/bin/time -v -o "$RUN_DIR/client.time.log" python scripts/benchmark_cliente.py \
        --url "http://127.0.0.1:$PORT/v1" \
        --model "$LOCAL_MODEL_DIR" \
        --prompts "$PROMPTS_FILE" \
        --out-jsonl "$RUN_DIR/requests.jsonl" \
        --out-summary "$RUN_DIR/summary.json" \
        --concurrency $CONCURRENCY

    # Limpar a casa para a próxima iteração
    echo "A desligar servidor PID: $SERVER_PID"
    kill $SERVER_PID
    kill $MONITOR_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
done

echo "========================================================="
echo "✅ SWEEP TERMINADO COM SUCESSO!"
echo "========================================================="