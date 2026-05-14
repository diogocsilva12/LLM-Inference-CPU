#!/usr/bin/env bash
#SBATCH --job-name=a1-node-scaling
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=46
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL

set -euo pipefail

load_modules() {
  local modules=(
    "GCC/13.3.0"
    "cmake/3.21.3"
  )

  if command -v module >/dev/null 2>&1; then
    module --ignore_cache purge
  elif command -v ml >/dev/null 2>&1; then
    ml purge
  fi

  for module_name in "${modules[@]}"; do
    echo "[LOAD_MODULE] $module_name"
    if command -v module >/dev/null 2>&1; then
      module --ignore_cache load "$module_name"
    elif command -v ml >/dev/null 2>&1; then
      ml "$module_name"
    fi
  done
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

normalize_list() {
  local value="$1"
  value="${value//,/ }"
  value="${value//;/ }"
  printf '%s' "$value"
}

count_words() {
  local count=0
  for _ in $1; do
    count="$((count + 1))"
  done
  printf '%s' "$count"
}

start_node_resource_monitor() {
  local output_path="$1"
  local watched_pid="$2"
  local role="${3:-node}"

  if [[ "$ENABLE_NODE_RESOURCE_MONITOR" != "1" && "$ENABLE_NODE_RESOURCE_MONITOR" != "true" ]]; then
    return 0
  fi

  if command -v dstat >/dev/null 2>&1; then
    dstat --time --cpu --mem --output "$output_path" "$RESOURCE_MONITOR_INTERVAL" >/dev/null 2>&1 &
    printf '%s\n' "$!"
    return 0
  fi

  (
    echo "timestamp,role,cpu_user_jiffies,cpu_nice_jiffies,cpu_system_jiffies,cpu_idle_jiffies,cpu_iowait_jiffies,cpu_irq_jiffies,cpu_softirq_jiffies,cpu_steal_jiffies,mem_total_kb,mem_available_kb,mem_used_kb,mem_free_kb,swap_total_kb,swap_free_kb"
    while kill -0 "$watched_pid" 2>/dev/null; do
      local ts
      local cpu_label cpu_user cpu_nice cpu_system cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal cpu_guest cpu_guest_nice
      local mem_values
      ts="$(date --iso-8601=seconds)"
      read -r cpu_label cpu_user cpu_nice cpu_system cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal cpu_guest cpu_guest_nice < /proc/stat
      mem_values="$(awk '
        /^MemTotal:/ {total=$2}
        /^MemAvailable:/ {available=$2}
        /^MemFree:/ {free=$2}
        /^SwapTotal:/ {swap_total=$2}
        /^SwapFree:/ {swap_free=$2}
        END {
          if (available == "") available = free
          used = total - available
          printf "%s,%s,%s,%s,%s,%s", total+0, available+0, used+0, free+0, swap_total+0, swap_free+0
        }
      ' /proc/meminfo)"
      echo "$ts,$role,${cpu_user:-0},${cpu_nice:-0},${cpu_system:-0},${cpu_idle:-0},${cpu_iowait:-0},${cpu_irq:-0},${cpu_softirq:-0},${cpu_steal:-0},$mem_values"
      sleep "$RESOURCE_MONITOR_INTERVAL"
    done
  ) > "$output_path" &
  printf '%s\n' "$!"
}

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_DIRECT:-0}" != "1" ]]; then
  if command -v sbatch >/dev/null 2>&1; then
    echo "[INFO] Resubmitting through sbatch. Set ALLOW_DIRECT=1 to run without SLURM."
    exec sbatch --export=ALL "$0" "$@"
  fi
  echo "This benchmark is intended to run under SLURM with sbatch." >&2
  exit 2
fi

load_modules

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -f "$SUBMIT_DIR/prompts/track_a_prompts.json" ]]; then
    PROJECT_DIR="$SUBMIT_DIR"
  else
    PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
A1_NODE_SCALING_SUBMIT_SCRIPT="${A1_NODE_SCALING_SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-node-scaling.sh}"

LLAMA_REPO="${LLAMA_REPO:-$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant}"
ENGINE_LABEL="${ENGINE_LABEL:-llamacpp-optimal}"
ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm-rpc}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
AUTO_BUILD="${AUTO_BUILD:-0}"
BUILD_TARGETS="${BUILD_TARGETS:-llama-server rpc-server}"
SERVER_BIN="${SERVER_BIN:-$LLAMA_REPO/$ENGINE_BUILD_DIR_NAME/bin/llama-server}"
RPC_SERVER_BIN="${RPC_SERVER_BIN:-$LLAMA_REPO/$ENGINE_BUILD_DIR_NAME/bin/rpc-server}"

PROMPTS_FILE="${PROMPTS_FILE:-$PROJECT_DIR/prompts/track_a_prompts.json}"
BENCH_SCRIPT="${BENCH_SCRIPT:-$PROJECT_DIR/scripts/benchmark_openai_stream.py}"
QUALITY_SCRIPT="${QUALITY_SCRIPT:-$PROJECT_DIR/scripts/evaluate_mandatory_outputs.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$PROJECT_DIR/scripts/summarize_a1_results.py}"
OUTPUT_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-node-scaling}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUBMIT_DIR/measurements/${OUTPUT_JOB_ID}-${OUTPUT_SUFFIX}}"

MODEL_LABEL="${MODEL_LABEL:-model-1-mandatory}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-19080}"
RPC_BIND_HOST="${RPC_BIND_HOST:-0.0.0.0}"
RPC_PORT_BASE="${RPC_PORT_BASE:-50052}"
RPC_USE_CACHE="${RPC_USE_CACHE:-1}"
RPC_MAX_SERVERS="${RPC_MAX_SERVERS:-255}"
NODE_COUNTS="${NODE_COUNTS:-1 2 4 6 8 16 24 32 64 128 256}"
NODE_COUNT="${NODE_COUNT:-${SLURM_NNODES:-1}}"
THREADS="${THREADS:-46}"
THREADS_BATCH="${THREADS_BATCH:-46}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-2}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-$PARALLEL_REQUESTS}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TRIALS="${TRIALS:-1}"
WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
TIMEOUT="${TIMEOUT:-1200}"
SERVER_STARTUP_WAIT_SECONDS="${SERVER_STARTUP_WAIT_SECONDS:-180}"
MANDATORY_ONLY="${MANDATORY_ONLY:-0}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"
ENABLE_NODE_RESOURCE_MONITOR="${ENABLE_NODE_RESOURCE_MONITOR:-1}"
RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
FIT_PARAMS="${FIT_PARAMS:-off}"
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
SUBMIT_NODE_SWEEP="${SUBMIT_NODE_SWEEP:-0}"
SUBMIT_SUMMARY_JOB="${SUBMIT_SUMMARY_JOB:-1}"
NODE_SWEEP_DEPENDENCY_TYPE="${NODE_SWEEP_DEPENDENCY_TYPE:-afterany}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
LIST_CONFIGS="${LIST_CONFIGS:-0}"
WRITE_SUMMARY_AFTER_RUN="${WRITE_SUMMARY_AFTER_RUN:-1}"

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Missing prompts file: $PROMPTS_FILE" >&2
  exit 1
fi
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Missing benchmark script: $BENCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$QUALITY_SCRIPT" ]]; then
  echo "Missing quality script: $QUALITY_SCRIPT" >&2
  exit 1
fi

write_final_outputs() {
  python3 "$SUMMARY_SCRIPT" "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/a1_node_scaling_summary.csv" \
    --scaling-out "$OUTPUT_DIR/a1_node_scaling_scaling_summary.csv"

  python3 "$QUALITY_SCRIPT" "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/mandatory_answer_quality.csv" \
    --summary-out "$OUTPUT_DIR/mandatory_answer_quality_summary.json"

  cat > "$OUTPUT_DIR/run_manifest.txt" <<EOF
track=A1
suite=node-scaling
engine=$ENGINE_LABEL
repo_dir=$LLAMA_REPO
server_bin=$SERVER_BIN
rpc_server_bin=$RPC_SERVER_BIN
rpc_bind_host=$RPC_BIND_HOST
rpc_port_base=$RPC_PORT_BASE
rpc_use_cache=$RPC_USE_CACHE
rpc_max_servers=$RPC_MAX_SERVERS
model_label=$MODEL_LABEL
model=$MODEL_PATH
node_counts=$NODE_COUNTS
node_count_config_count=$(count_words "$NODE_COUNTS")
threads=$THREADS
threads_batch=$THREADS_BATCH
ctx_size=$CTX_SIZE
parallel_requests=$PARALLEL_REQUESTS
bench_concurrency=$BENCH_CONCURRENCY
server_startup_wait_seconds=$SERVER_STARTUP_WAIT_SECONDS
max_tokens=$MAX_TOKENS
n_gpu_layers=$N_GPU_LAYERS
cache_type_k=$CACHE_TYPE_K
cache_type_v=$CACHE_TYPE_V
fit_params=$FIT_PARAMS
flash_attn=$FLASH_ATTN
prompt_scope=limit_per_category_${LIMIT_PER_CATEGORY:-all}
mandatory_only=$MANDATORY_ONLY
trials=$TRIALS
warmup_trials=$WARMUP_TRIALS
summary=$OUTPUT_DIR/a1_node_scaling_summary.csv
scaling_summary=$OUTPUT_DIR/a1_node_scaling_scaling_summary.csv
quality=$OUTPUT_DIR/mandatory_answer_quality.csv
quality_summary=$OUTPUT_DIR/mandatory_answer_quality_summary.json
EOF
}

if [[ "$SUMMARY_ONLY" == "1" || "$SUMMARY_ONLY" == "true" ]]; then
  mkdir -p "$OUTPUT_DIR"
  write_final_outputs
  echo "[OK] Node-scaling summary complete: $OUTPUT_DIR"
  echo "[OK] Timing summary: $OUTPUT_DIR/a1_node_scaling_summary.csv"
  echo "[OK] Scaling summary: $OUTPUT_DIR/a1_node_scaling_scaling_summary.csv"
  exit 0
fi

if [[ "$LIST_CONFIGS" == "1" || "$LIST_CONFIGS" == "true" ]]; then
  echo "NODE_COUNTS=$(normalize_list "$NODE_COUNTS")"
  echo "NODE_COUNT_CONFIG_COUNT=$(count_words "$NODE_COUNTS")"
  echo "THREADS=$THREADS"
  echo "CTX_SIZE=$CTX_SIZE"
  echo "PARALLEL_REQUESTS=$PARALLEL_REQUESTS"
  echo "BENCH_CONCURRENCY=$BENCH_CONCURRENCY"
  echo "LIMIT_PER_CATEGORY=$LIMIT_PER_CATEGORY"
  echo "SERVER_STARTUP_WAIT_SECONDS=$SERVER_STARTUP_WAIT_SECONDS"
  echo "RPC_BIND_HOST=$RPC_BIND_HOST"
  echo "RPC_PORT_BASE=$RPC_PORT_BASE"
  echo "RPC_MAX_SERVERS=$RPC_MAX_SERVERS"
  exit 0
fi

if [[ ! -d "$LLAMA_REPO" || ! -f "$LLAMA_REPO/CMakeLists.txt" ]]; then
  echo "Missing llama.cpp repository: $LLAMA_REPO" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 1
fi

acquire_build_lock() {
  local lock_dir="$1"
  local waited=0

  while ! mkdir "$lock_dir" 2>/dev/null; do
    if [[ "$waited" -eq 0 ]]; then
      echo "[BUILD] Waiting for build lock: $lock_dir"
    fi
    waited="$((waited + 1))"
    sleep 5
  done
}

release_build_lock() {
  local lock_dir="$1"
  rmdir "$lock_dir" 2>/dev/null || true
}

build_engine_if_needed() {
  local lock_dir="$LLAMA_REPO/.node-scaling-build.lock"
  local rpc_c_flags="-DGGML_RPC_MAX_SERVERS=$RPC_MAX_SERVERS ${CMAKE_C_FLAGS:-}"
  local rpc_cxx_flags="-DGGML_RPC_MAX_SERVERS=$RPC_MAX_SERVERS ${CMAKE_CXX_FLAGS:-}"

  if [[ -x "$SERVER_BIN" && -x "$RPC_SERVER_BIN" ]]; then
    echo "[BUILD] Reusing $SERVER_BIN and $RPC_SERVER_BIN"
    return
  fi

  if [[ "$AUTO_BUILD" != "1" && "$AUTO_BUILD" != "true" ]]; then
    echo "Missing RPC-enabled binaries: $SERVER_BIN and/or $RPC_SERVER_BIN" >&2
    echo "Build first with -DGGML_RPC=ON or set AUTO_BUILD=1." >&2
    exit 1
  fi

  acquire_build_lock "$lock_dir"
  if [[ -x "$SERVER_BIN" && -x "$RPC_SERVER_BIN" ]]; then
    release_build_lock "$lock_dir"
    echo "[BUILD] Reusing $SERVER_BIN and $RPC_SERVER_BIN"
    return
  fi

  echo "[BUILD] Building RPC-enabled $SERVER_BIN and $RPC_SERVER_BIN"
  set +e
  cmake -S "$LLAMA_REPO" -B "$LLAMA_REPO/$ENGINE_BUILD_DIR_NAME" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_RPC=ON \
    -DCMAKE_C_FLAGS="$rpc_c_flags" \
    -DCMAKE_CXX_FLAGS="$rpc_cxx_flags"
  local configure_status=$?
  if [[ "$configure_status" -eq 0 ]]; then
    cmake --build "$LLAMA_REPO/$ENGINE_BUILD_DIR_NAME" -j "${SLURM_CPUS_PER_TASK:-$THREADS}" --target $BUILD_TARGETS
  fi
  local build_status=$?
  set -e
  release_build_lock "$lock_dir"
  if [[ "$configure_status" -ne 0 ]]; then
    exit "$configure_status"
  fi
  if [[ "$build_status" -ne 0 ]]; then
    exit "$build_status"
  fi
}

resolve_cache_types() {
  if [[ -n "$CACHE_TYPE_K" && -n "$CACHE_TYPE_V" ]]; then
    return
  fi

  local help_text
  help_text="$("$SERVER_BIN" -h 2>&1 || true)"
  if [[ "$help_text" == *"turbo3_0"* ]]; then
    CACHE_TYPE_K="${CACHE_TYPE_K:-turbo3_0}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3_0}"
  elif [[ "$help_text" == *"turbo3"* ]]; then
    CACHE_TYPE_K="${CACHE_TYPE_K:-turbo3}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3}"
  else
    CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
    CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
    echo "[WARN] Turbo cache type not available in this binary; falling back to f16." >&2
  fi
}

supports_cache_pair() {
  local help_text
  help_text="$("$SERVER_BIN" -h 2>&1 || true)"
  [[ "$help_text" == *"$CACHE_TYPE_K"* && "$help_text" == *"$CACHE_TYPE_V"* ]]
}

supports_rpc() {
  local help_text
  help_text="$("$SERVER_BIN" -h 2>&1 || true)"
  [[ "$help_text" == *"--rpc"* ]]
}

submit_node_sweep_if_requested() {
  local submitted_ids=()
  local node_count
  local submit_output
  local submit_status
  local job_id
  local previous_job_id=""
  local dependency_arg=()
  local node_time_arg=()
  local summary_time_arg=()

  if [[ "$SUBMIT_NODE_SWEEP" != "1" && "$SUBMIT_NODE_SWEEP" != "true" ]]; then
    return
  fi
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "SUBMIT_NODE_SWEEP=1 requires sbatch." >&2
    exit 2
  fi

  mkdir -p "$OUTPUT_DIR"
  echo "[INFO] Submitting node-count jobs into $OUTPUT_DIR"
  for node_count in $(normalize_list "$NODE_COUNTS"); do
    dependency_arg=()
    if [[ -n "$previous_job_id" ]]; then
      dependency_arg=(--dependency="${NODE_SWEEP_DEPENDENCY_TYPE}:$previous_job_id")
      echo "[INFO] Submitting node scaling job with $node_count nodes after job $previous_job_id"
    else
      echo "[INFO] Submitting node scaling job with $node_count nodes"
    fi
    node_time_arg=()
    if [[ -n "${NODE_JOB_TIME:-}" ]]; then
      node_time_arg=(--time="$NODE_JOB_TIME")
    fi
    set +e
    submit_output="$(sbatch \
      "${dependency_arg[@]}" \
      "${node_time_arg[@]}" \
      --nodes="$node_count" \
      --ntasks="$node_count" \
      --ntasks-per-node=1 \
      --cpus-per-task="$THREADS" \
      --export=ALL,A1_NODE_SCALING_SUBMIT_SCRIPT="$A1_NODE_SCALING_SUBMIT_SCRIPT",SUBMIT_NODE_SWEEP=0,NODE_COUNT="$node_count",OUTPUT_DIR="$OUTPUT_DIR",WRITE_SUMMARY_AFTER_RUN=0 \
      "$A1_NODE_SCALING_SUBMIT_SCRIPT" "$@" 2>&1)"
    submit_status="$?"
    set -e
    printf '%s\n' "$submit_output"
    if [[ "$submit_status" -ne 0 ]]; then
      exit "$submit_status"
    fi
    job_id="$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4; exit}')"
    if [[ -n "$job_id" ]]; then
      submitted_ids+=("$job_id")
      previous_job_id="$job_id"
    fi
  done

  if [[ "${#submitted_ids[@]}" -gt 0 ]]; then
    echo "[INFO] Submitted sequential node-scaling chain: ${submitted_ids[*]}"
    if [[ "$SUBMIT_SUMMARY_JOB" == "1" || "$SUBMIT_SUMMARY_JOB" == "true" ]]; then
      echo "[INFO] Submitting summary job after final node-scaling job: $previous_job_id"
      summary_time_arg=()
      if [[ -n "${SUMMARY_JOB_TIME:-}" ]]; then
        summary_time_arg=(--time="$SUMMARY_JOB_TIME")
      fi
      sbatch \
        --dependency="${NODE_SWEEP_DEPENDENCY_TYPE}:$previous_job_id" \
        "${summary_time_arg[@]}" \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --cpus-per-task="$THREADS" \
        --export=ALL,A1_NODE_SCALING_SUBMIT_SCRIPT="$A1_NODE_SCALING_SUBMIT_SCRIPT",SUBMIT_NODE_SWEEP=0,SUMMARY_ONLY=1,OUTPUT_DIR="$OUTPUT_DIR" \
        "$A1_NODE_SCALING_SUBMIT_SCRIPT" "$@"
    else
      echo "[INFO] After all jobs finish, summarize with:"
      echo "OUTPUT_DIR=$OUTPUT_DIR SUMMARY_ONLY=1 sbatch --dependency=${NODE_SWEEP_DEPENDENCY_TYPE}:$previous_job_id --export=ALL $A1_NODE_SCALING_SUBMIT_SCRIPT"
    fi
  fi

  exit 0
}

write_inventories() {
  mkdir -p "$OUTPUT_DIR"
  {
    echo "label,path,bytes"
    bytes="$(stat -c '%s' "$MODEL_PATH")"
    echo "$MODEL_LABEL,$MODEL_PATH,$bytes"
  } > "$OUTPUT_DIR/model_inventory.csv"

  {
    echo "label,repo,server_bin,rpc_server_bin,cache_type_k,cache_type_v,rpc_max_servers"
    echo "$ENGINE_LABEL,$LLAMA_REPO,$SERVER_BIN,$RPC_SERVER_BIN,$CACHE_TYPE_K,$CACHE_TYPE_V,$RPC_MAX_SERVERS"
  } > "$OUTPUT_DIR/engine_inventory.csv"
}

allocated_hosts() {
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_JOB_NODELIST"
  else
    hostname -s
  fi
}

build_rpc_endpoints() {
  local rank=0
  local host
  local port
  local endpoints=""

  while read -r host; do
    [[ -z "$host" ]] && continue
    if [[ "$rank" -gt 0 && "$rank" -lt "$NODE_COUNT" ]]; then
      port="$((RPC_PORT_BASE + rank))"
      if [[ -n "$endpoints" ]]; then
        endpoints+=","
      fi
      endpoints+="${host}:${port}"
    fi
    rank="$((rank + 1))"
  done < <(allocated_hosts)

  printf '%s' "$endpoints"
}

wait_for_rpc_endpoints() {
  local endpoints="$1"
  local endpoint
  local host
  local port
  local ready

  [[ -z "$endpoints" ]] && return

  for endpoint in ${endpoints//,/ }; do
    host="${endpoint%%:*}"
    port="${endpoint##*:}"
    ready=0
    for _ in $(seq 1 180); do
      if python3 -c 'import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()' "$host" "$port" >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
      echo "RPC endpoint did not become reachable: $endpoint" >&2
      exit 1
    fi
  done
}

rpc_done_file() {
  printf '%s/node-scaling-n%s/node_count_%s/rpc_done' "$OUTPUT_DIR" "$NODE_COUNT" "$NODE_COUNT"
}

run_rpc_worker() {
  local rank="${SLURM_PROCID:-0}"
  local host
  local host_safe
  local port
  local suite
  local run_dir
  local rpc_log
  local resource_log
  local node_resource_log
  local system_out
  local rpc_pid
  local monitor_pid=""
  local node_monitor_pid=""
  local cache_args=()
  local done_file

  host="$(hostname -s)"
  host_safe="$(safe_name "$host")"
  port="$((RPC_PORT_BASE + rank))"
  suite="node-scaling-n${NODE_COUNT}"
  run_dir="$OUTPUT_DIR/$suite/node_count_${NODE_COUNT}/rpc_worker_${rank}_${host_safe}"
  rpc_log="$run_dir/rpc-server.log"
  resource_log="$run_dir/resources.csv"
  node_resource_log="$run_dir/node_resources.csv"
  system_out="$run_dir/system.txt"
  done_file="$(rpc_done_file)"

  mkdir -p "$run_dir"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "track=A1"
    echo "suite=$suite"
    echo "engine=$ENGINE_LABEL"
    echo "role=rpc-worker"
    echo "rank=$rank"
    echo "node=$host"
    echo "node_count=$NODE_COUNT"
    echo "allocated_nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "repo_dir=$LLAMA_REPO"
    echo "rpc_server_bin=$RPC_SERVER_BIN"
    echo "rpc_bind_host=$RPC_BIND_HOST"
    echo "rpc_port=$port"
    echo "rpc_use_cache=$RPC_USE_CACHE"
    echo "threads=$THREADS"
    echo "enable_node_resource_monitor=$ENABLE_NODE_RESOURCE_MONITOR"
    echo "resource_monitor_interval=$RESOURCE_MONITOR_INTERVAL"
    lscpu || true
  } > "$system_out"

  export OMP_NUM_THREADS="$THREADS"
  export OPENBLAS_NUM_THREADS="$THREADS"
  export BLIS_NUM_THREADS="$THREADS"

  if [[ "$RPC_USE_CACHE" == "1" || "$RPC_USE_CACHE" == "true" ]]; then
    cache_args+=(--cache)
  fi

  echo "[RPC_WORKER] suite=$suite rank=$rank node=$host bind=$RPC_BIND_HOST:$port threads=$THREADS"
  "$RPC_SERVER_BIN" \
    --host "$RPC_BIND_HOST" \
    --port "$port" \
    --threads "$THREADS" \
    "${cache_args[@]}" \
    > "$rpc_log" 2>&1 &
  rpc_pid=$!

  cleanup_worker() {
    if [[ -n "$node_monitor_pid" ]]; then
      kill "$node_monitor_pid" 2>/dev/null || true
      wait "$node_monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    kill "$rpc_pid" 2>/dev/null || true
    wait "$rpc_pid" 2>/dev/null || true
  }

  echo "timestamp,cpu_percent,rss_kb,vmhwm_kb" > "$resource_log"
  (
    while kill -0 "$rpc_pid" 2>/dev/null; do
      ts="$(date --iso-8601=seconds)"
      cpu="$(ps -p "$rpc_pid" -o %cpu= | awk '{print $1}')"
      rss="$(awk '/VmRSS:/ {print $2}' "/proc/$rpc_pid/status" 2>/dev/null || echo 0)"
      hwm="$(awk '/VmHWM:/ {print $2}' "/proc/$rpc_pid/status" 2>/dev/null || echo 0)"
      echo "$ts,${cpu:-0},${rss:-0},${hwm:-0}"
      sleep 1
    done
  ) >> "$resource_log" &
  monitor_pid=$!
  node_monitor_pid="$(start_node_resource_monitor "$node_resource_log" "$rpc_pid" "rpc-worker")"

  while [[ ! -f "$done_file" ]]; do
    if ! kill -0 "$rpc_pid" 2>/dev/null; then
      echo "rpc-server exited before coordinator completed. See $rpc_log" >&2
      echo "----- rpc-server.log tail -----" >&2
      tail -n 80 "$rpc_log" >&2 || true
      echo "-------------------------------" >&2
      cleanup_worker
      exit 1
    fi
    sleep 1
  done

  cleanup_worker
  echo "[OK] RPC worker rank $rank complete: $run_dir"
}

run_rpc_coordinator() {
  local rank="${SLURM_PROCID:-0}"
  local host
  local host_safe
  local port="$BASE_PORT"
  local suite
  local run_dir
  local server_log
  local resource_log
  local node_resource_log
  local raw_out
  local summary_out
  local system_out
  local server_pid
  local monitor_pid=""
  local node_monitor_pid=""
  local server_args=()
  local bench_args=()
  local rpc_args=()
  local rpc_endpoints
  local done_file

  host="$(hostname -s)"
  host_safe="$(safe_name "$host")"
  suite="node-scaling-n${NODE_COUNT}"
  run_dir="$OUTPUT_DIR/$suite/node_count_${NODE_COUNT}/rpc_coordinator_${host_safe}/t_${THREADS}_c_${CTX_SIZE}_p_${PARALLEL_REQUESTS}_n_${MAX_TOKENS}_k_${CACHE_TYPE_K}_v_${CACHE_TYPE_V}"
  server_log="$run_dir/server.log"
  resource_log="$run_dir/resources.csv"
  node_resource_log="$run_dir/node_resources.csv"
  raw_out="$run_dir/requests.jsonl"
  summary_out="$run_dir/summary.json"
  system_out="$run_dir/system.txt"
  done_file="$(rpc_done_file)"
  rpc_endpoints="$(build_rpc_endpoints)"

  if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
    read -r -a server_args <<< "$SERVER_EXTRA_ARGS"
  fi
  if [[ -n "$rpc_endpoints" ]]; then
    rpc_args+=(--rpc "$rpc_endpoints")
  fi

  mkdir -p "$run_dir"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "track=A1"
    echo "suite=$suite"
    echo "engine=$ENGINE_LABEL"
    echo "role=rpc-coordinator"
    echo "rank=$rank"
    echo "node=$host"
    echo "node_count=$NODE_COUNT"
    echo "rpc_endpoints=$rpc_endpoints"
    echo "rpc_max_servers=$RPC_MAX_SERVERS"
    echo "allocated_nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "model_label=$MODEL_LABEL"
    echo "model=$MODEL_PATH"
    echo "repo_dir=$LLAMA_REPO"
    echo "server_bin=$SERVER_BIN"
    echo "rpc_server_bin=$RPC_SERVER_BIN"
    echo "threads=$THREADS"
    echo "threads_batch=$THREADS_BATCH"
    echo "ctx_size=$CTX_SIZE"
    echo "parallel_requests=$PARALLEL_REQUESTS"
    echo "bench_concurrency=$BENCH_CONCURRENCY"
    echo "server_startup_wait_seconds=$SERVER_STARTUP_WAIT_SECONDS"
    echo "cache_type_k=$CACHE_TYPE_K"
    echo "cache_type_v=$CACHE_TYPE_V"
    echo "max_tokens=$MAX_TOKENS"
    echo "n_gpu_layers=$N_GPU_LAYERS"
    echo "fit_params=$FIT_PARAMS"
    echo "flash_attn=$FLASH_ATTN"
    echo "trials=$TRIALS"
    echo "warmup_trials=$WARMUP_TRIALS"
    echo "mandatory_only=$MANDATORY_ONLY"
    echo "limit_per_category=${LIMIT_PER_CATEGORY:-<unset>}"
    echo "enable_node_resource_monitor=$ENABLE_NODE_RESOURCE_MONITOR"
    echo "resource_monitor_interval=$RESOURCE_MONITOR_INTERVAL"
    echo "server_extra_args=${server_args[*]}"
    lscpu || true
  } > "$system_out"

  export OMP_NUM_THREADS="$THREADS"
  export OPENBLAS_NUM_THREADS="$THREADS"
  export BLIS_NUM_THREADS="$THREADS"

  wait_for_rpc_endpoints "$rpc_endpoints"

  echo "[RPC_COORDINATOR] suite=$suite node=$host node_count=$NODE_COUNT rpc_endpoints=${rpc_endpoints:-<none>} threads=$THREADS ctx=$CTX_SIZE parallel=$PARALLEL_REQUESTS"
  # llama-server receives --rpc "$rpc_endpoints" through rpc_args when endpoints exist.
  "$SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$port" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    -ngl "$N_GPU_LAYERS" \
    -c "$CTX_SIZE" \
    -np "$PARALLEL_REQUESTS" \
    --flash-attn "$FLASH_ATTN" \
    -fit "$FIT_PARAMS" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    "${rpc_args[@]}" \
    "${server_args[@]}" \
    > "$server_log" 2>&1 &
  server_pid=$!

  cleanup_coordinator() {
    touch "$done_file" 2>/dev/null || true
    if [[ -n "$node_monitor_pid" ]]; then
      kill "$node_monitor_pid" 2>/dev/null || true
      wait "$node_monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }

  for _ in $(seq 1 "$SERVER_STARTUP_WAIT_SECONDS"); do
    if curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "llama-server exited before becoming healthy. See $server_log" >&2
      echo "----- server.log tail -----" >&2
      tail -n 80 "$server_log" >&2 || true
      echo "---------------------------" >&2
      cleanup_coordinator
      exit 1
    fi
    sleep 1
  done

  if ! curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
    echo "llama-server did not become healthy within ${SERVER_STARTUP_WAIT_SECONDS}s. See $server_log" >&2
    echo "----- server.log tail -----" >&2
    tail -n 80 "$server_log" >&2 || true
    echo "---------------------------" >&2
    cleanup_coordinator
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
  node_monitor_pid="$(start_node_resource_monitor "$node_resource_log" "$server_pid" "rpc-coordinator")"

  bench_args=(
    "$BENCH_SCRIPT"
    --url "http://$HOST:$port/v1/chat/completions"
    --model local-model
    --prompts "$PROMPTS_FILE"
    --out "$raw_out"
    --summary-out "$summary_out"
    --trials "$TRIALS"
    --warmup-trials "$WARMUP_TRIALS"
    --max-tokens "$MAX_TOKENS"
    --concurrency "$BENCH_CONCURRENCY"
    --timeout "$TIMEOUT"
  )
  if [[ "$MANDATORY_ONLY" == "1" || "$MANDATORY_ONLY" == "true" ]]; then
    bench_args+=(--mandatory-only)
  fi
  if [[ -n "$LIMIT_PER_CATEGORY" ]]; then
    bench_args+=(--limit-per-category "$LIMIT_PER_CATEGORY")
  fi

  set +e
  /usr/bin/time -v python3 "${bench_args[@]}" 2> "$run_dir/client.time.log"
  local bench_status=$?
  set -e

  cleanup_coordinator
  if [[ "$bench_status" -ne 0 ]]; then
    echo "Benchmark failed for $run_dir with exit code $bench_status" >&2
    exit "$bench_status"
  fi
  echo "[OK] RPC coordinator complete: $run_dir"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ ! -x "$SERVER_BIN" || ! -x "$RPC_SERVER_BIN" ]]; then
    echo "Missing RPC-enabled binaries in worker: $SERVER_BIN and/or $RPC_SERVER_BIN" >&2
    exit 1
  fi
  resolve_cache_types
  if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    run_rpc_coordinator
  else
    run_rpc_worker
  fi
  exit 0
fi

submit_node_sweep_if_requested "$@"
build_engine_if_needed
resolve_cache_types
if ! supports_cache_pair; then
  echo "Server binary does not support cache pair K=$CACHE_TYPE_K V=$CACHE_TYPE_V: $SERVER_BIN" >&2
  exit 1
fi
if ! supports_rpc; then
  echo "Server binary does not expose --rpc. Rebuild with -DGGML_RPC=ON: $SERVER_BIN" >&2
  exit 1
fi
if [[ "$((NODE_COUNT - 1))" -gt "$RPC_MAX_SERVERS" ]]; then
  echo "NODE_COUNT=$NODE_COUNT needs $((NODE_COUNT - 1)) RPC servers, but RPC_MAX_SERVERS=$RPC_MAX_SERVERS." >&2
  echo "Increase RPC_MAX_SERVERS and rebuild the RPC-enabled build directory." >&2
  exit 1
fi
write_inventories

mkdir -p "$OUTPUT_DIR/node-scaling-n${NODE_COUNT}/node_count_${NODE_COUNT}"
rm -f "$(rpc_done_file)"

echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Engine: $ENGINE_LABEL ($LLAMA_REPO)"
echo "[INFO] Server: $SERVER_BIN"
echo "[INFO] RPC server: $RPC_SERVER_BIN"
echo "[INFO] Model: $MODEL_LABEL ($MODEL_PATH)"
echo "[INFO] Node count: $NODE_COUNT"
echo "[INFO] RPC worker count: $((NODE_COUNT - 1))"
echo "[INFO] RPC max servers: $RPC_MAX_SERVERS"
echo "[INFO] Threads per node: $THREADS"
echo "[INFO] Context: $CTX_SIZE"
echo "[INFO] Parallel requests: $PARALLEL_REQUESTS"
echo "[INFO] Bench concurrency: $BENCH_CONCURRENCY"
echo "[INFO] Max tokens: $MAX_TOKENS"
echo "[INFO] Cache types: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "[INFO] Prompt protocol: mandatory_only=$MANDATORY_ONLY limit_per_category=${LIMIT_PER_CATEGORY:-<unset>}"

srun --nodes="$NODE_COUNT" --ntasks="$NODE_COUNT" --ntasks-per-node=1 \
  --cpus-per-task="$THREADS" --kill-on-bad-exit=1 bash "$A1_NODE_SCALING_SUBMIT_SCRIPT" --worker

if [[ "$WRITE_SUMMARY_AFTER_RUN" == "1" || "$WRITE_SUMMARY_AFTER_RUN" == "true" ]]; then
  write_final_outputs
  echo "[OK] Track A1 node-scaling run complete: $OUTPUT_DIR"
  echo "[OK] Timing summary: $OUTPUT_DIR/a1_node_scaling_summary.csv"
  echo "[OK] Scaling summary: $OUTPUT_DIR/a1_node_scaling_scaling_summary.csv"
else
  echo "[OK] Track A1 node-scaling workers complete: $OUTPUT_DIR"
fi
