#!/usr/bin/env bash
#SBATCH --job-name=a1-tq-vs-vanilla
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --acctg-freq=energy=10

set -euo pipefail

echo "[STARTING] Loading modules"
modules=(
  "GCC/13.3.0"
  "cmake/3.21.3"
)

if command -v module >/dev/null 2>&1; then
  module --ignore_cache purge
else
  ml purge
fi
for module in "${modules[@]}"; do
  echo "[LOAD_MODULE] $module"
  if command -v module >/dev/null 2>&1; then
    module --ignore_cache load "$module"
  else
    ml "$module"
  fi
done

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -f "$SUBMIT_DIR/prompts/track_a_prompts.json" ]]; then
    PROJECT_DIR="$SUBMIT_DIR"
  else
    PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
  fi
fi

TQ_REPO="${TQ_REPO:-$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant}"
VANILLA_REPO="${VANILLA_REPO:-$PROJECT_DIR/llamacpp-vanilla/llama.cpp}"
MTP_REPO="${MTP_REPO:-$PROJECT_DIR/llmacpp-mtp/atomic-llama-cpp-turboquant}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
MTP_MODEL_PATH="${MTP_MODEL_PATH:-$MODEL_PATH}"
MTP_HEAD_PATH="${MTP_HEAD_PATH:-}"
PROMPTS_FILE="${PROMPTS_FILE:-$PROJECT_DIR/prompts/track_a_prompts.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUBMIT_DIR/measurements/${SLURM_JOB_ID:-manual}-tq-vs-vanilla-mandatory}"
BENCH_SCRIPT="${BENCH_SCRIPT:-$PROJECT_DIR/scripts/benchmark_openai_stream.py}"
LIVE_PROGRESS="${LIVE_PROGRESS:-0}"
LIVE_PROGRESS_FILE="${LIVE_PROGRESS_FILE:-live_progress.json}"

AUTO_BUILD="${AUTO_BUILD:-1}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_TARGETS="${BUILD_TARGETS:-llama-server}"
ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm}"
INCLUDE_TQ="${INCLUDE_TQ:-1}"
INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"
INCLUDE_MTP="${INCLUDE_MTP:-0}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-18080}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
THREADS_BATCH="${THREADS_BATCH:-}"
CTX_SIZE="${CTX_SIZE:-2048}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TRIALS="${TRIALS:-2}"
WARMUP_TRIALS="${WARMUP_TRIALS:-0}"
MANDATORY_ONLY="${MANDATORY_ONLY:-1}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-}"

# Deucalion ARM nodes are organized in 12-core NUMA CMGs. Sweep whole-CMG
# thread counts by default: 1, 2, 3, and 4 CMGs.
THREADS_LIST="${THREADS_LIST:-12 24 36 48}"
CTX_SIZE_LIST="${CTX_SIZE_LIST:-$CTX_SIZE}"
PARALLEL_REQUESTS_LIST="${PARALLEL_REQUESTS_LIST:-$PARALLEL_REQUESTS}"
TQ_CACHE_SWEEP="${TQ_CACHE_SWEEP:-$CACHE_TYPE_K:$CACHE_TYPE_V}"
VANILLA_CACHE_SWEEP="${VANILLA_CACHE_SWEEP:-$CACHE_TYPE_K:$CACHE_TYPE_V}"
MTP_CACHE_SWEEP="${MTP_CACHE_SWEEP:-$CACHE_TYPE_K:$CACHE_TYPE_V}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
TQ_EXTRA_ARGS="${TQ_EXTRA_ARGS:-}"
VANILLA_EXTRA_ARGS="${VANILLA_EXTRA_ARGS:-}"
MTP_EXTRA_ARGS="${MTP_EXTRA_ARGS:-}"
MTP_DRAFT_BLOCK_SIZE="${MTP_DRAFT_BLOCK_SIZE:-3}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-8}"
MTP_DRAFT_MIN="${MTP_DRAFT_MIN:-0}"

THREADS_LIST="${THREADS_LIST//,/ }"
THREADS_LIST="${THREADS_LIST//;/ }"
CTX_SIZE_LIST="${CTX_SIZE_LIST//,/ }"
CTX_SIZE_LIST="${CTX_SIZE_LIST//;/ }"
PARALLEL_REQUESTS_LIST="${PARALLEL_REQUESTS_LIST//,/ }"
PARALLEL_REQUESTS_LIST="${PARALLEL_REQUESTS_LIST//;/ }"
TQ_CACHE_SWEEP="${TQ_CACHE_SWEEP//,/ }"
TQ_CACHE_SWEEP="${TQ_CACHE_SWEEP//;/ }"
VANILLA_CACHE_SWEEP="${VANILLA_CACHE_SWEEP//,/ }"
VANILLA_CACHE_SWEEP="${VANILLA_CACHE_SWEEP//;/ }"
MTP_CACHE_SWEEP="${MTP_CACHE_SWEEP//,/ }"
MTP_CACHE_SWEEP="${MTP_CACHE_SWEEP//;/ }"

if [[ ( "$INCLUDE_TQ" == "1" || "$INCLUDE_TQ" == "true" || "$INCLUDE_VANILLA" == "1" || "$INCLUDE_VANILLA" == "true" ) && ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$INCLUDE_MTP" == "1" || "$INCLUDE_MTP" == "true" ]]; then
  if [[ ! -f "$MTP_MODEL_PATH" ]]; then
    echo "Missing MTP target model file: $MTP_MODEL_PATH" >&2
    exit 1
  fi
  if [[ -n "$MTP_HEAD_PATH" && ! -f "$MTP_HEAD_PATH" ]]; then
    echo "Missing MTP assistant/head file: $MTP_HEAD_PATH" >&2
    exit 1
  fi
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Missing prompts file: $PROMPTS_FILE" >&2
  exit 1
fi

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Missing benchmark script: $BENCH_SCRIPT" >&2
  exit 1
fi

build_if_needed() {
  local repo_dir="$1"
  local build_dir="$repo_dir/$ENGINE_BUILD_DIR_NAME"
  local server_bin="$build_dir/bin/llama-server"

  if [[ ! -f "$repo_dir/CMakeLists.txt" ]]; then
    echo "Missing engine repository: $repo_dir" >&2
    exit 1
  fi

  if [[ -x "$server_bin" ]]; then
    echo "[BUILD] Reusing $server_bin"
    return
  fi

  if [[ "$AUTO_BUILD" != "1" && "$AUTO_BUILD" != "true" ]]; then
    echo "Missing llama-server binary: $server_bin" >&2
    echo "Set AUTO_BUILD=1 or build the engine first." >&2
    exit 1
  fi

  echo "[BUILD] Building $repo_dir"
  cmake -S "$repo_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  cmake --build "$build_dir" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target $BUILD_TARGETS
}

supports_cache_pair() {
  local server_bin="$1"
  local cache_k="$2"
  local cache_v="$3"
  local help_text

  help_text="$("$server_bin" -h 2>&1 || true)"
  [[ "$help_text" == *"$cache_k"* && "$help_text" == *"$cache_v"* ]]
}

run_engine() {
  local engine="$1"
  local repo_dir="$2"
  local port="$3"
  local threads="$4"
  local ctx_size="$5"
  local parallel_requests="$6"
  local cache_type_k="$7"
  local cache_type_v="$8"
  local model_path="$9"
  local extra_arg_string="${10}"
  local run_dir="${11}"
  local build_dir="$repo_dir/$ENGINE_BUILD_DIR_NAME"
  local server_bin="$build_dir/bin/llama-server"
  local threads_batch="${THREADS_BATCH:-$threads}"
  local bench_concurrency="${BENCH_CONCURRENCY:-$parallel_requests}"
  local server_args=()

  if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
    read -r -a server_args <<< "$SERVER_EXTRA_ARGS"
  fi
  if [[ -n "$extra_arg_string" ]]; then
    local engine_args=()
    read -r -a engine_args <<< "$extra_arg_string"
    server_args+=("${engine_args[@]}")
  fi

  mkdir -p "$run_dir"

  local server_log="$run_dir/server.log"
  local resource_log="$run_dir/resources.csv"
  local raw_out="$run_dir/requests.jsonl"
  local summary_out="$run_dir/summary.json"
  local system_out="$run_dir/system.txt"

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "engine=$engine"
    echo "node=${SLURMD_NODENAME:-$(hostname -s)}"
    echo "model=$model_path"
    echo "repo_dir=$repo_dir"
    echo "server_bin=$server_bin"
    echo "threads=$threads"
    echo "threads_batch=$threads_batch"
    echo "ctx_size=$ctx_size"
    echo "parallel_requests=$parallel_requests"
    echo "cache_type_k=$cache_type_k"
    echo "cache_type_v=$cache_type_v"
    echo "max_tokens=$MAX_TOKENS"
    echo "trials=$TRIALS"
    echo "warmup_trials=$WARMUP_TRIALS"
    echo "bench_concurrency=$bench_concurrency"
    echo "mandatory_only=$MANDATORY_ONLY"
    echo "limit_per_category=${LIMIT_PER_CATEGORY:-<unset>}"
    echo "server_extra_args=${server_args[*]}"
    lscpu || true
  } > "$system_out"

  export OMP_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"

  echo "[RUN] Starting $engine threads=$threads ctx=$ctx_size parallel=$parallel_requests cache=$cache_type_k:$cache_type_v on port $port"
  "$server_bin" \
    -m "$model_path" \
    --host "$HOST" \
    --port "$port" \
    --threads "$threads" \
    --threads-batch "$threads_batch" \
    -ngl "$N_GPU_LAYERS" \
    -c "$ctx_size" \
    -np "$parallel_requests" \
    --flash-attn "$FLASH_ATTN" \
    --cache-type-k "$cache_type_k" \
    --cache-type-v "$cache_type_v" \
    "${server_args[@]}" \
    > "$server_log" 2>&1 &
  local server_pid=$!
  local monitor_pid=""

  cleanup_engine() {
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }

  for _ in $(seq 1 120); do
    if curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
    echo "$engine did not become healthy. See $server_log" >&2
    cleanup_engine
    exit 1
  fi

  echo "timestamp,cpu_percent,rss_kb,vmhwm_kb" > "$resource_log"
  (
    while kill -0 "$server_pid" 2>/dev/null; do
      ts="$(date --iso-8601=seconds)"
      cpu="$(ps -p "$server_pid" -o %cpu= | awk '{print $1}')"
      rss="$(awk '/VmRSS:/ {print $2}' "/proc/$server_pid/status" 2>/dev/null || echo 0)"
      hwm="$(awk '/VmHWM:/ {print $2}' "/proc/$server_pid/status" 2>/dev/null || echo 0)"
      echo "$ts,${cpu:-0},${rss:-0},${hwm:-0}"
      sleep 1
    done
  ) >> "$resource_log" &
  monitor_pid=$!

  set +e
  bench_args=(
    "$BENCH_SCRIPT"
    --url "http://$HOST:$port/v1/chat/completions" \
    --model local-model \
    --prompts "$PROMPTS_FILE" \
    --out "$raw_out" \
    --summary-out "$summary_out" \
    --trials "$TRIALS" \
    --warmup-trials "$WARMUP_TRIALS" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$bench_concurrency"
  )
  if [[ "$MANDATORY_ONLY" == "1" || "$MANDATORY_ONLY" == "true" ]]; then
    bench_args+=(--mandatory-only)
  fi
  if [[ -n "$LIMIT_PER_CATEGORY" ]]; then
    bench_args+=(--limit-per-category "$LIMIT_PER_CATEGORY")
  fi
  if [[ "$LIVE_PROGRESS" == "1" || "$LIVE_PROGRESS" == "true" ]]; then
    bench_args+=(--live-out "$run_dir/$LIVE_PROGRESS_FILE")
  fi
  /usr/bin/time -v python3 "${bench_args[@]}" 2> "$run_dir/client.time.log"
  local bench_status=$?
  set -e

  cleanup_engine
  if [[ "$bench_status" -ne 0 ]]; then
    echo "$engine benchmark failed with exit code $bench_status" >&2
    exit "$bench_status"
  fi
  echo "[RUN] Finished $engine"
}

run_sweep() {
  local engine="$1"
  local repo_dir="$2"
  local base_port="$3"
  local cache_sweep="$4"
  local run_count="$5"
  local model_path="$6"
  local extra_arg_string="$7"
  local build_dir="$repo_dir/$ENGINE_BUILD_DIR_NAME"
  local server_bin="$build_dir/bin/llama-server"
  local port="$base_port"

  for threads in $THREADS_LIST; do
    for ctx_size in $CTX_SIZE_LIST; do
      for parallel_requests in $PARALLEL_REQUESTS_LIST; do
        for pair in $cache_sweep; do
          local cache_type_k="${pair%%:*}"
          local cache_type_v="${pair##*:}"

          if ! supports_cache_pair "$server_bin" "$cache_type_k" "$cache_type_v"; then
            echo "[WARN] Skipping unsupported $engine cache pair K=$cache_type_k V=$cache_type_v"
            continue
          fi

          if [[ "$run_count" -eq 1 ]]; then
            run_dir="$OUTPUT_DIR/$engine"
          else
            run_dir="$OUTPUT_DIR/$engine/t_${threads}_c_${ctx_size}_p_${parallel_requests}_k_${cache_type_k}_v_${cache_type_v}"
          fi

          run_engine "$engine" "$repo_dir" "$port" "$threads" "$ctx_size" "$parallel_requests" "$cache_type_k" "$cache_type_v" "$model_path" "$extra_arg_string" "$run_dir"
          port="$((port + 2))"
        done
      done
    done
  done
}

count_requested_runs() {
  local cache_sweep="$1"
  local threads_count=0
  local ctx_count=0
  local parallel_count=0
  local cache_count=0

  for _ in $THREADS_LIST; do
    threads_count="$((threads_count + 1))"
  done
  for _ in $CTX_SIZE_LIST; do
    ctx_count="$((ctx_count + 1))"
  done
  for _ in $PARALLEL_REQUESTS_LIST; do
    parallel_count="$((parallel_count + 1))"
  done
  for _ in $cache_sweep; do
    cache_count="$((cache_count + 1))"
  done

  echo "$((threads_count * ctx_count * parallel_count * cache_count))"
}

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] TQ repo: $TQ_REPO"
echo "[INFO] Vanilla repo: $VANILLA_REPO"
echo "[INFO] MTP repo: $MTP_REPO"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] MTP model: $MTP_MODEL_PATH"
echo "[INFO] MTP head: ${MTP_HEAD_PATH:-<unset>}"
echo "[INFO] Engine build dir name: $ENGINE_BUILD_DIR_NAME"
echo "[INFO] Iterations per prompt: $TRIALS"
echo "[INFO] Threads list: $THREADS_LIST"
echo "[INFO] Context size list: $CTX_SIZE_LIST"
echo "[INFO] Parallel requests list: $PARALLEL_REQUESTS_LIST"
echo "[INFO] TQ cache sweep: $TQ_CACHE_SWEEP"
echo "[INFO] Vanilla cache sweep: $VANILLA_CACHE_SWEEP"
echo "[INFO] MTP cache sweep: $MTP_CACHE_SWEEP"
echo "[INFO] Include engines: tq=$INCLUDE_TQ vanilla=$INCLUDE_VANILLA mtp=$INCLUDE_MTP"

if [[ "$INCLUDE_TQ" == "1" || "$INCLUDE_TQ" == "true" ]]; then
  build_if_needed "$TQ_REPO"
fi
if [[ "$INCLUDE_VANILLA" == "1" || "$INCLUDE_VANILLA" == "true" ]]; then
  build_if_needed "$VANILLA_REPO"
fi
if [[ "$INCLUDE_MTP" == "1" || "$INCLUDE_MTP" == "true" ]]; then
  build_if_needed "$MTP_REPO"
fi

if [[ "$INCLUDE_TQ" == "1" || "$INCLUDE_TQ" == "true" ]]; then
  run_sweep "tq" "$TQ_REPO" "$BASE_PORT" "$TQ_CACHE_SWEEP" "$(count_requested_runs "$TQ_CACHE_SWEEP")" "$MODEL_PATH" "$TQ_EXTRA_ARGS"
fi
if [[ "$INCLUDE_VANILLA" == "1" || "$INCLUDE_VANILLA" == "true" ]]; then
  run_sweep "vanilla" "$VANILLA_REPO" "$((BASE_PORT + 1))" "$VANILLA_CACHE_SWEEP" "$(count_requested_runs "$VANILLA_CACHE_SWEEP")" "$MODEL_PATH" "$VANILLA_EXTRA_ARGS"
fi
if [[ "$INCLUDE_MTP" == "1" || "$INCLUDE_MTP" == "true" ]]; then
  mtp_args="$MTP_EXTRA_ARGS"
  if [[ -n "$MTP_HEAD_PATH" ]]; then
    mtp_args="--mtp-head $MTP_HEAD_PATH --spec-type mtp --draft-block-size $MTP_DRAFT_BLOCK_SIZE --draft-max $MTP_DRAFT_MAX --draft-min $MTP_DRAFT_MIN $mtp_args"
  fi
  run_sweep "mtp" "$MTP_REPO" "$((BASE_PORT + 2))" "$MTP_CACHE_SWEEP" "$(count_requested_runs "$MTP_CACHE_SWEEP")" "$MTP_MODEL_PATH" "$mtp_args"
fi

python3 "$PROJECT_DIR/scripts/summarize_a1_results.py" "$OUTPUT_DIR" \
  --out "$OUTPUT_DIR/comparison_summary.csv" \
  --scaling-out "$OUTPUT_DIR/scaling_summary.csv"

echo "[OK] Comparison complete: $OUTPUT_DIR"
echo "[OK] Summary: $OUTPUT_DIR/comparison_summary.csv"
echo "[OK] Scaling summary: $OUTPUT_DIR/scaling_summary.csv"
