#!/usr/bin/env bash
#SBATCH --job-name=a1-server-sweep
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_DIRECT:-0}" != "1" ]]; then
  if command -v sbatch >/dev/null 2>&1; then
    echo "[INFO] Resubmitting through sbatch. Set ALLOW_DIRECT=1 to run without SLURM."
    exec sbatch --export=ALL "$0" "$@"
  fi
  echo "This benchmark is intended to run under SLURM with sbatch." >&2
  exit 2
fi

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

load_modules

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
A1_SERVER_SWEEP_SUBMIT_SCRIPT="${A1_SERVER_SWEEP_SUBMIT_SCRIPT:-$0}"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -f "$SUBMIT_DIR/prompts/track_a_prompts.json" ]]; then
    PROJECT_DIR="$SUBMIT_DIR"
  else
    PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
  fi
fi

DEFAULT_ENGINE_SPECS="tq=$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant;vanilla=$PROJECT_DIR/llamacpp-vanilla/llama.cpp"
if [[ -z "${ENGINE_SPECS:-}" && -n "${LLAMA_REPO:-}" ]]; then
  ENGINE_SPECS="${ENGINE_NAME:-llamacpp-server}=$LLAMA_REPO"
else
  ENGINE_SPECS="${ENGINE_SPECS:-$DEFAULT_ENGINE_SPECS}"
fi
ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
AUTO_BUILD="${AUTO_BUILD:-1}"
BUILD_TARGETS="${BUILD_TARGETS:-llama-server}"

PROMPTS_FILE="${PROMPTS_FILE:-$PROJECT_DIR/prompts/track_a_prompts.json}"
BENCH_SCRIPT="${BENCH_SCRIPT:-$PROJECT_DIR/scripts/benchmark_openai_stream.py}"
QUALITY_SCRIPT="${QUALITY_SCRIPT:-$PROJECT_DIR/scripts/evaluate_mandatory_outputs.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$PROJECT_DIR/scripts/summarize_a1_results.py}"
OUTPUT_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-server-sweep}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUBMIT_DIR/measurements/${OUTPUT_JOB_ID}-${OUTPUT_SUFFIX}}"

DEFAULT_MODEL_SPECS="model-1-mandatory=$PROJECT_DIR/llamacpp-tq/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf;model-2=$PROJECT_DIR/llamacpp-tq/models/SmolLM2-360M-Instruct.Q4_K_M.gguf;model-3=$PROJECT_DIR/llamacpp-tq/models/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"
MODEL_SPECS="${MODEL_SPECS:-$DEFAULT_MODEL_SPECS}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-18080}"
THREADS="${THREADS:-24}"
THREADS_BATCH="${THREADS_BATCH:-}"
CTX_SIZE="${CTX_SIZE:-2048}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-$PARALLEL_REQUESTS}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TRIALS="${TRIALS:-3}"
WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
TIMEOUT="${TIMEOUT:-900}"
MANDATORY_ONLY="${MANDATORY_ONLY:-0}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"
ENABLE_NODE_RESOURCE_MONITOR="${ENABLE_NODE_RESOURCE_MONITOR:-1}"
RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
FLASH_ATTN="${FLASH_ATTN:-off}"
FIT_PARAMS="${FIT_PARAMS:-off}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
DEFAULT_ENGINE_CACHE_SWEEPS="tq=turbo2:turbo2 turbo3:turbo3 turbo4:turbo4;vanilla=f16:f16"
ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-$DEFAULT_ENGINE_CACHE_SWEEPS}"
CACHE_SWEEP="$(normalize_list "${CACHE_SWEEP:-$CACHE_TYPE_K:$CACHE_TYPE_V}")"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

THREADS_LIST="$(normalize_list "${THREADS_LIST:-12 24 36 46 48}")"
CONCURRENCY_LIST="$(normalize_list "${CONCURRENCY_LIST:-1 2 4 8 16}")"
CTX_SIZE_LIST="$(normalize_list "${CTX_SIZE_LIST:-512 1024 2048 4096}")"
MAX_TOKENS_LIST="$(normalize_list "${MAX_TOKENS_LIST:-64 128 256 512}")"

RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"
RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-1}"
RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-1}"
RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-1}"
RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-1}"
RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"
LIST_CONFIGS="${LIST_CONFIGS:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SUBMIT_ARRAY="${SUBMIT_ARRAY:-0}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-tq:model-2:*}"

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

build_engine_if_needed() {
  local repo_dir="$1"
  local server_bin="$2"
  local lock_dir="$repo_dir/.build-slurm.lock"

  if [[ -x "$server_bin" ]]; then
    echo "[BUILD] Reusing $server_bin"
    return
  fi

  if [[ ! -f "$repo_dir/CMakeLists.txt" ]]; then
    echo "Missing llama.cpp repository: $repo_dir" >&2
    exit 1
  fi

  if [[ "$AUTO_BUILD" != "1" && "$AUTO_BUILD" != "true" ]]; then
    echo "Missing llama-server binary: $server_bin" >&2
    echo "Set AUTO_BUILD=1 or SERVER_BIN=/path/to/llama-server." >&2
    exit 1
  fi

  acquire_build_lock "$lock_dir"

  if [[ -x "$server_bin" ]]; then
    echo "[BUILD] Reusing $server_bin"
    release_build_lock "$lock_dir"
    return
  fi

  echo "[BUILD] Building $server_bin"
  set +e
  cmake -S "$repo_dir" -B "$repo_dir/$ENGINE_BUILD_DIR_NAME" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  local configure_status=$?
  if [[ "$configure_status" -eq 0 ]]; then
    cmake --build "$repo_dir/$ENGINE_BUILD_DIR_NAME" -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target $BUILD_TARGETS
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

IFS=';' read -r -a ENGINE_ITEMS <<< "$ENGINE_SPECS"
ENGINE_LABELS=()
ENGINE_REPOS=()
ENGINE_BINS=()
ENGINE_PRELOADS=()

value_for_label_spec() {
  local specs="$1"
  local requested_label="$2"
  local default_value="$3"
  local items=()
  local item
  local label

  if [[ -z "$specs" ]]; then
    printf '%s' "$default_value"
    return
  fi

  IFS=';' read -r -a items <<< "$specs"
  for item in "${items[@]}"; do
    [[ -z "$item" ]] && continue
    if [[ "$item" != *=* ]]; then
      echo "Invalid label spec entry: $item" >&2
      echo "Use label=value entries separated by semicolons." >&2
      exit 1
    fi
    label="$(safe_name "${item%%=*}")"
    if [[ "$label" == "$requested_label" ]]; then
      printf '%s' "${item#*=}"
      return
    fi
  done

  printf '%s' "$default_value"
}

for item in "${ENGINE_ITEMS[@]}"; do
  [[ -z "$item" ]] && continue
  if [[ "$item" != *=* ]]; then
    echo "Invalid ENGINE_SPECS entry: $item" >&2
    echo "Use label=/absolute/path/llama.cpp entries separated by semicolons." >&2
    exit 1
  fi
  engine_label="${item%%=*}"
  engine_repo="${item#*=}"
  engine_label_safe="$(safe_name "$engine_label")"
  engine_bin="$engine_repo/$ENGINE_BUILD_DIR_NAME/bin/llama-server"
  if [[ "${#ENGINE_ITEMS[@]}" -eq 1 && -n "${SERVER_BIN:-}" ]]; then
    engine_bin="$SERVER_BIN"
  fi
  engine_bin="$(value_for_label_spec "${ENGINE_BIN_SPECS:-}" "$engine_label_safe" "$engine_bin")"
  engine_preload="$(value_for_label_spec "${ENGINE_PRELOAD_SPECS:-}" "$engine_label_safe" "")"
  ENGINE_LABELS+=("$engine_label_safe")
  ENGINE_REPOS+=("$engine_repo")
  ENGINE_BINS+=("$engine_bin")
  ENGINE_PRELOADS+=("$engine_preload")
done

if [[ "${#ENGINE_LABELS[@]}" -lt 1 ]]; then
  echo "ENGINE_SPECS must include at least one engine." >&2
  exit 1
fi

cache_sweep_for_engine() {
  local requested_label="$1"
  local fallback_sweep="$CACHE_SWEEP"
  local cache_items=()
  local item
  local label
  local sweep

  IFS=';' read -r -a cache_items <<< "$ENGINE_CACHE_SWEEPS"
  for item in "${cache_items[@]}"; do
    [[ -z "$item" ]] && continue
    if [[ "$item" != *=* ]]; then
      echo "Invalid ENGINE_CACHE_SWEEPS entry: $item" >&2
      echo "Use label=K:V entries separated by semicolons. Multiple pairs for one label are space-separated." >&2
      exit 1
    fi
    label="$(safe_name "${item%%=*}")"
    sweep="${item#*=}"
    if [[ "$label" == "$requested_label" ]]; then
      normalize_list "$sweep"
      return
    fi
  done

  normalize_list "$fallback_sweep"
}

ENGINE_CACHE_SWEEPS_BY_INDEX=()
for idx in "${!ENGINE_LABELS[@]}"; do
  ENGINE_CACHE_SWEEPS_BY_INDEX+=("$(cache_sweep_for_engine "${ENGINE_LABELS[$idx]}")")
done

IFS=';' read -r -a MODEL_ITEMS <<< "$MODEL_SPECS"
MODEL_LABELS=()
MODEL_PATHS=()
MANDATORY_INDEX=""
for item in "${MODEL_ITEMS[@]}"; do
  [[ -z "$item" ]] && continue
  if [[ "$item" != *=* ]]; then
    echo "Invalid MODEL_SPECS entry: $item" >&2
    echo "Use label=/absolute/path/model.gguf entries separated by semicolons." >&2
    exit 1
  fi
  label="${item%%=*}"
  model_path="${item#*=}"
  if [[ ! -e "$model_path" ]]; then
    echo "Missing model path for $label: $model_path" >&2
    exit 1
  fi
  MODEL_LABELS+=("$(safe_name "$label")")
  MODEL_PATHS+=("$model_path")
  if [[ "$label" == *mandatory* || "$model_path" == *Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf ]]; then
    MANDATORY_INDEX="$((${#MODEL_LABELS[@]} - 1))"
  fi
done

if [[ "${#MODEL_LABELS[@]}" -lt 3 ]]; then
  echo "Track A requires at least three models. MODEL_SPECS currently has ${#MODEL_LABELS[@]}." >&2
  exit 1
fi
if [[ -z "$MANDATORY_INDEX" ]]; then
  echo "MODEL_SPECS must include Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

write_inventories() {
  {
    echo "label,path,bytes"
    for idx in "${!MODEL_LABELS[@]}"; do
      if [[ -f "${MODEL_PATHS[$idx]}" ]]; then
        bytes="$(stat -c '%s' "${MODEL_PATHS[$idx]}")"
      elif [[ -d "${MODEL_PATHS[$idx]}" ]]; then
        bytes="$(find "${MODEL_PATHS[$idx]}" -type f -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s + 0}')"
      else
        bytes=0
      fi
      echo "${MODEL_LABELS[$idx]},${MODEL_PATHS[$idx]},$bytes"
    done
  } > "$OUTPUT_DIR/model_inventory.csv"

  {
    echo "label,repo,server_bin,cache_sweep,ld_preload"
    for idx in "${!ENGINE_LABELS[@]}"; do
      echo "${ENGINE_LABELS[$idx]},${ENGINE_REPOS[$idx]},${ENGINE_BINS[$idx]},${ENGINE_CACHE_SWEEPS_BY_INDEX[$idx]},${ENGINE_PRELOADS[$idx]}"
    done
  } > "$OUTPUT_DIR/engine_inventory.csv"
}

supports_cache_pair() {
  local server_bin="$1"
  local cache_type_k="$2"
  local cache_type_v="$3"
  local engine_preload="${4:-}"
  local help_text
  local help_env=()
  if [[ -n "$engine_preload" ]]; then
    help_env+=(LD_PRELOAD="$engine_preload${LD_PRELOAD:+:$LD_PRELOAD}")
  fi
  help_text="$(env "${help_env[@]}" "$server_bin" -h 2>&1 || true)"
  [[ "$help_text" == *"$cache_type_k"* && "$help_text" == *"$cache_type_v"* ]]
}

run_config() {
  local suite="$1"
  local engine_label="$2"
  local engine_repo="$3"
  local server_bin="$4"
  local model_label="$5"
  local model_path="$6"
  local threads="$7"
  local ctx_size="$8"
  local parallel_requests="$9"
  local bench_concurrency="${10}"
  local max_tokens="${11}"
  local cache_type_k="${12}"
  local cache_type_v="${13}"
  local port="${14}"
  local engine_preload="${15:-}"
  local run_dir="$OUTPUT_DIR/$suite/$engine_label/$model_label/t_${threads}_c_${ctx_size}_p_${parallel_requests}_n_${max_tokens}_k_${cache_type_k}_v_${cache_type_v}"
  local threads_batch="${THREADS_BATCH:-$threads}"
  local server_log="$run_dir/server.log"
  local resource_log="$run_dir/resources.csv"
  local node_resource_log="$run_dir/node_resources.csv"
  local raw_out="$run_dir/requests.jsonl"
  local summary_out="$run_dir/summary.json"
  local system_out="$run_dir/system.txt"
  local server_args=()

  if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
    read -r -a server_args <<< "$SERVER_EXTRA_ARGS"
  fi

  mkdir -p "$run_dir"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "track=A1"
    echo "suite=$suite"
    echo "engine=$engine_label"
    echo "node=${SLURMD_NODENAME:-$(hostname -s)}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "model_label=$model_label"
    echo "model=$model_path"
    echo "repo_dir=$engine_repo"
    echo "server_bin=$server_bin"
    echo "ld_preload=$engine_preload"
    echo "threads=$threads"
    echo "threads_batch=$threads_batch"
    echo "ctx_size=$ctx_size"
    echo "parallel_requests=$parallel_requests"
    echo "bench_concurrency=$bench_concurrency"
    echo "cache_type_k=$cache_type_k"
    echo "cache_type_v=$cache_type_v"
    echo "max_tokens=$max_tokens"
    echo "fit_params=$FIT_PARAMS"
    echo "trials=$TRIALS"
    echo "warmup_trials=$WARMUP_TRIALS"
    echo "mandatory_only=$MANDATORY_ONLY"
    echo "limit_per_category=${LIMIT_PER_CATEGORY:-<unset>}"
    echo "enable_node_resource_monitor=$ENABLE_NODE_RESOURCE_MONITOR"
    echo "resource_monitor_interval=$RESOURCE_MONITOR_INTERVAL"
    echo "server_extra_args=${server_args[*]}"
    lscpu || true
  } > "$system_out"

  export OMP_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"
  export BLIS_NUM_THREADS="$threads"

  echo "[RUN] suite=$suite engine=$engine_label model=$model_label threads=$threads ctx=$ctx_size parallel=$parallel_requests max_tokens=$max_tokens cache=$cache_type_k:$cache_type_v port=$port"
  server_env=()
  if [[ -n "$engine_preload" ]]; then
    server_env+=(LD_PRELOAD="$engine_preload${LD_PRELOAD:+:$LD_PRELOAD}")
  fi
  env "${server_env[@]}" "$server_bin" \
    -m "$model_path" \
    --host "$HOST" \
    --port "$port" \
    --threads "$threads" \
    --threads-batch "$threads_batch" \
    -ngl "$N_GPU_LAYERS" \
    -c "$ctx_size" \
    -np "$parallel_requests" \
    --flash-attn "$FLASH_ATTN" \
    -fit "$FIT_PARAMS" \
    --cache-type-k "$cache_type_k" \
    --cache-type-v "$cache_type_v" \
    "${server_args[@]}" \
    > "$server_log" 2>&1 &
  local server_pid=$!
  local monitor_pid=""
  local node_monitor_pid=""

  cleanup_config() {
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

  for _ in $(seq 1 180); do
    if curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "llama-server exited before becoming healthy. See $server_log" >&2
      echo "----- server.log tail -----" >&2
      tail -n 80 "$server_log" >&2 || true
      echo "---------------------------" >&2
      cleanup_config
      exit 1
    fi
    sleep 1
  done

  if ! curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
    echo "llama-server did not become healthy. See $server_log" >&2
    echo "----- server.log tail -----" >&2
    tail -n 80 "$server_log" >&2 || true
    echo "---------------------------" >&2
    cleanup_config
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
  node_monitor_pid="$(start_node_resource_monitor "$node_resource_log" "$server_pid" "server")"

  bench_args=(
    "$BENCH_SCRIPT"
    --url "http://$HOST:$port/v1/chat/completions"
    --model local-model
    --prompts "$PROMPTS_FILE"
    --out "$raw_out"
    --summary-out "$summary_out"
    --trials "$TRIALS"
    --warmup-trials "$WARMUP_TRIALS"
    --max-tokens "$max_tokens"
    --concurrency "$bench_concurrency"
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

  cleanup_config
  if [[ "$bench_status" -ne 0 ]]; then
    echo "Benchmark failed for $run_dir with exit code $bench_status" >&2
    exit "$bench_status"
  fi
}

run_config_if_supported() {
  local engine_idx="$1"
  local suite="$2"
  local model_label="$3"
  local model_path="$4"
  local threads="$5"
  local ctx_size="$6"
  local parallel_requests="$7"
  local bench_concurrency="$8"
  local max_tokens="$9"
  local cache_type_k="${10}"
  local cache_type_v="${11}"
  local engine_label="${ENGINE_LABELS[$engine_idx]}"
  local engine_repo="${ENGINE_REPOS[$engine_idx]}"
  local server_bin="${ENGINE_BINS[$engine_idx]}"
  local engine_preload="${ENGINE_PRELOADS[$engine_idx]}"

  build_engine_if_needed "$engine_repo" "$server_bin"
  if ! supports_cache_pair "$server_bin" "$cache_type_k" "$cache_type_v" "$engine_preload"; then
    echo "[WARN] Skipping unsupported cache pair for engine=$engine_label K=$cache_type_k V=$cache_type_v"
    return
  fi

  run_config "$suite" "$engine_label" "$engine_repo" "$server_bin" "$model_label" "$model_path" "$threads" "$ctx_size" "$parallel_requests" "$bench_concurrency" "$max_tokens" "$cache_type_k" "$cache_type_v" "$next_port" "$engine_preload"
  next_port="$((next_port + 1))"
}

CONFIG_SUITES=()
CONFIG_ENGINE_IDXS=()
CONFIG_MODEL_LABELS=()
CONFIG_MODEL_PATHS=()
CONFIG_THREADS=()
CONFIG_CTX_SIZES=()
CONFIG_PARALLEL_REQUESTS=()
CONFIG_BENCH_CONCURRENCIES=()
CONFIG_MAX_TOKENS=()
CONFIG_CACHE_TYPE_K=()
CONFIG_CACHE_TYPE_V=()
EXCLUDED_CONFIG_COUNT=0

field_matches() {
  local pattern="$1"
  local value="$2"
  [[ "$pattern" == "*" || "$pattern" == "$value" ]]
}

config_is_excluded() {
  local engine_idx="$1"
  local model_label="$2"
  local cache_type_k="$3"
  local cache_type_v="$4"
  local engine_label="${ENGINE_LABELS[$engine_idx]}"
  local patterns
  local pattern
  local rest
  local engine_pattern
  local model_pattern
  local cache_k_pattern
  local cache_v_pattern

  if [[ "$EXCLUDE_CONFIGS" == "0" || "$EXCLUDE_CONFIGS" == "false" || "$EXCLUDE_CONFIGS" == "none" || "$EXCLUDE_CONFIGS" == "off" ]]; then
    return 1
  fi

  patterns="$(normalize_list "$EXCLUDE_CONFIGS")"
  for pattern in $patterns; do
    [[ -z "$pattern" ]] && continue
    if [[ "$pattern" != *:*:* ]]; then
      echo "Invalid EXCLUDE_CONFIGS entry: $pattern" >&2
      echo "Use engine:model_label:cache_k[:cache_v], with * wildcards." >&2
      exit 1
    fi

    engine_pattern="${pattern%%:*}"
    rest="${pattern#*:}"
    model_pattern="${rest%%:*}"
    rest="${rest#*:}"
    cache_k_pattern="${rest%%:*}"
    if [[ "$rest" == "$cache_k_pattern" ]]; then
      cache_v_pattern="*"
    else
      cache_v_pattern="${rest#*:}"
    fi

    if field_matches "$engine_pattern" "$engine_label" \
      && field_matches "$model_pattern" "$model_label" \
      && field_matches "$cache_k_pattern" "$cache_type_k" \
      && field_matches "$cache_v_pattern" "$cache_type_v"; then
      return 0
    fi
  done

  return 1
}

add_config() {
  local suite="$1"
  local engine_idx="$2"
  local model_label="$3"
  local model_path="$4"
  local threads="$5"
  local ctx_size="$6"
  local parallel_requests="$7"
  local bench_concurrency="$8"
  local max_tokens="$9"
  local cache_type_k="${10}"
  local cache_type_v="${11}"

  if config_is_excluded "$engine_idx" "$model_label" "$cache_type_k" "$cache_type_v"; then
    EXCLUDED_CONFIG_COUNT="$((EXCLUDED_CONFIG_COUNT + 1))"
    echo "[SKIP_CONFIG] suite=$suite engine=${ENGINE_LABELS[$engine_idx]} model=$model_label cache=$cache_type_k:$cache_type_v matches EXCLUDE_CONFIGS=$EXCLUDE_CONFIGS"
    return
  fi

  CONFIG_SUITES+=("$suite")
  CONFIG_ENGINE_IDXS+=("$engine_idx")
  CONFIG_MODEL_LABELS+=("$model_label")
  CONFIG_MODEL_PATHS+=("$model_path")
  CONFIG_THREADS+=("$threads")
  CONFIG_CTX_SIZES+=("$ctx_size")
  CONFIG_PARALLEL_REQUESTS+=("$parallel_requests")
  CONFIG_BENCH_CONCURRENCIES+=("$bench_concurrency")
  CONFIG_MAX_TOKENS+=("$max_tokens")
  CONFIG_CACHE_TYPE_K+=("$cache_type_k")
  CONFIG_CACHE_TYPE_V+=("$cache_type_v")
}

add_configs_for_all_engines() {
  local suite="$1"
  local model_label="$2"
  local model_path="$3"
  local threads="$4"
  local ctx_size="$5"
  local parallel_requests="$6"
  local bench_concurrency="$7"
  local max_tokens="$8"
  local pair
  local cache_type_k
  local cache_type_v

  for engine_idx in "${!ENGINE_LABELS[@]}"; do
    for pair in ${ENGINE_CACHE_SWEEPS_BY_INDEX[$engine_idx]}; do
      cache_type_k="${pair%%:*}"
      cache_type_v="${pair##*:}"
      add_config "$suite" "$engine_idx" "$model_label" "$model_path" "$threads" "$ctx_size" "$parallel_requests" "$bench_concurrency" "$max_tokens" "$cache_type_k" "$cache_type_v"
    done
  done
}

build_config_matrix() {
  local idx
  local threads_value
  local concurrency_value
  local ctx_value
  local max_tokens_value
  local pair
  local cache_type_k
  local cache_type_v
  local engine_idx

  CONFIG_SUITES=()
  CONFIG_ENGINE_IDXS=()
  CONFIG_MODEL_LABELS=()
  CONFIG_MODEL_PATHS=()
  CONFIG_THREADS=()
  CONFIG_CTX_SIZES=()
  CONFIG_PARALLEL_REQUESTS=()
  CONFIG_BENCH_CONCURRENCIES=()
  CONFIG_MAX_TOKENS=()
  CONFIG_CACHE_TYPE_K=()
  CONFIG_CACHE_TYPE_V=()

  if [[ "$RUN_MODEL_SWEEP" == "1" || "$RUN_MODEL_SWEEP" == "true" ]]; then
    for idx in "${!MODEL_LABELS[@]}"; do
      add_configs_for_all_engines "model-set" "${MODEL_LABELS[$idx]}" "${MODEL_PATHS[$idx]}" "$THREADS" "$CTX_SIZE" "$PARALLEL_REQUESTS" "$BENCH_CONCURRENCY" "$MAX_TOKENS"
    done
  fi

  if [[ "$RUN_THREAD_SWEEP" == "1" || "$RUN_THREAD_SWEEP" == "true" ]]; then
    for threads_value in $THREADS_LIST; do
      add_configs_for_all_engines "thread-scaling" "$mandatory_label" "$mandatory_path" "$threads_value" "$CTX_SIZE" "$PARALLEL_REQUESTS" "$BENCH_CONCURRENCY" "$MAX_TOKENS"
    done
  fi

  if [[ "$RUN_CONCURRENCY_SWEEP" == "1" || "$RUN_CONCURRENCY_SWEEP" == "true" ]]; then
    for concurrency_value in $CONCURRENCY_LIST; do
      add_configs_for_all_engines "concurrency" "$mandatory_label" "$mandatory_path" "$THREADS" "$CTX_SIZE" "$concurrency_value" "$concurrency_value" "$MAX_TOKENS"
    done
  fi

  if [[ "$RUN_CONTEXT_SWEEP" == "1" || "$RUN_CONTEXT_SWEEP" == "true" ]]; then
    for ctx_value in $CTX_SIZE_LIST; do
      add_configs_for_all_engines "context-length" "$mandatory_label" "$mandatory_path" "$THREADS" "$ctx_value" "$PARALLEL_REQUESTS" "$BENCH_CONCURRENCY" "$MAX_TOKENS"
    done
  fi

  if [[ "$RUN_DECODE_SWEEP" == "1" || "$RUN_DECODE_SWEEP" == "true" ]]; then
    for max_tokens_value in $MAX_TOKENS_LIST; do
      add_configs_for_all_engines "decode-length" "$mandatory_label" "$mandatory_path" "$THREADS" "$CTX_SIZE" "$PARALLEL_REQUESTS" "$BENCH_CONCURRENCY" "$max_tokens_value"
    done
  fi

  if [[ "$RUN_CACHE_SWEEP" == "1" || "$RUN_CACHE_SWEEP" == "true" ]]; then
    for pair in $CACHE_SWEEP; do
      cache_type_k="${pair%%:*}"
      cache_type_v="${pair##*:}"
      for engine_idx in "${!ENGINE_LABELS[@]}"; do
        add_config "cache-sweep" "$engine_idx" "$mandatory_label" "$mandatory_path" "$THREADS" "$CTX_SIZE" "$PARALLEL_REQUESTS" "$BENCH_CONCURRENCY" "$MAX_TOKENS" "$cache_type_k" "$cache_type_v"
      done
    done
  fi
}

write_config_list() {
  local path="$1"
  local idx
  local engine_idx
  {
    echo "index,suite,engine,model_label,model,threads,ctx_size,parallel_requests,bench_concurrency,max_tokens,cache_type_k,cache_type_v"
    for idx in "${!CONFIG_SUITES[@]}"; do
      engine_idx="${CONFIG_ENGINE_IDXS[$idx]}"
      echo "$idx,${CONFIG_SUITES[$idx]},${ENGINE_LABELS[$engine_idx]},${CONFIG_MODEL_LABELS[$idx]},${CONFIG_MODEL_PATHS[$idx]},${CONFIG_THREADS[$idx]},${CONFIG_CTX_SIZES[$idx]},${CONFIG_PARALLEL_REQUESTS[$idx]},${CONFIG_BENCH_CONCURRENCIES[$idx]},${CONFIG_MAX_TOKENS[$idx]},${CONFIG_CACHE_TYPE_K[$idx]},${CONFIG_CACHE_TYPE_V[$idx]}"
    done
  } > "$path"
}

run_config_index() {
  local idx="$1"

  if [[ "$idx" -lt 0 || "$idx" -ge "${#CONFIG_SUITES[@]}" ]]; then
    echo "[INFO] Array task index $idx is outside config range 0-$(( ${#CONFIG_SUITES[@]} - 1 )); nothing to run."
    return
  fi

  run_config_if_supported \
    "${CONFIG_ENGINE_IDXS[$idx]}" \
    "${CONFIG_SUITES[$idx]}" \
    "${CONFIG_MODEL_LABELS[$idx]}" \
    "${CONFIG_MODEL_PATHS[$idx]}" \
    "${CONFIG_THREADS[$idx]}" \
    "${CONFIG_CTX_SIZES[$idx]}" \
    "${CONFIG_PARALLEL_REQUESTS[$idx]}" \
    "${CONFIG_BENCH_CONCURRENCIES[$idx]}" \
    "${CONFIG_MAX_TOKENS[$idx]}" \
    "${CONFIG_CACHE_TYPE_K[$idx]}" \
    "${CONFIG_CACHE_TYPE_V[$idx]}"
}

next_port="$BASE_PORT"
mandatory_label="${MODEL_LABELS[$MANDATORY_INDEX]}"
mandatory_path="${MODEL_PATHS[$MANDATORY_INDEX]}"

submit_array_if_requested() {
  local last_index="$(( ${#CONFIG_SUITES[@]} - 1 ))"
  local submit_output
  local submit_status
  local array_job_id
  local summary_output_dir

  if [[ "$SUBMIT_ARRAY" != "1" && "$SUBMIT_ARRAY" != "true" ]]; then
    return
  fi

  if [[ "${#CONFIG_SUITES[@]}" -lt 1 ]]; then
    echo "No configs to submit after applying EXCLUDE_CONFIGS=$EXCLUDE_CONFIGS" >&2
    exit 1
  fi
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "SUBMIT_ARRAY=1 requires sbatch." >&2
    exit 2
  fi

  echo "[INFO] Submitting array range 0-$last_index with concurrency $ARRAY_CONCURRENCY"
  set +e
  submit_output="$(sbatch --array="0-$last_index%$ARRAY_CONCURRENCY" --export=ALL,SUBMIT_ARRAY=0 "$A1_SERVER_SWEEP_SUBMIT_SCRIPT" "$@" 2>&1)"
  submit_status="$?"
  set -e
  printf '%s\n' "$submit_output"
  if [[ "$submit_status" -ne 0 ]]; then
    exit "$submit_status"
  fi

  array_job_id="$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4; exit}')"
  if [[ -n "$array_job_id" ]]; then
    if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
      summary_output_dir="$OUTPUT_DIR_OVERRIDE"
    else
      summary_output_dir="$SUBMIT_DIR/measurements/${array_job_id}-${OUTPUT_SUFFIX}"
    fi
    echo "[INFO] Array job id: $array_job_id"
    echo "[INFO] After it finishes, submit the summary job with:"
    echo "OUTPUT_DIR=$summary_output_dir SUMMARY_ONLY=1 sbatch --dependency=afterany:$array_job_id --export=ALL $A1_SERVER_SWEEP_SUBMIT_SCRIPT"
  fi

  exit 0
}

write_final_outputs() {
  python3 "$SUMMARY_SCRIPT" "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/a1_server_summary.csv" \
    --scaling-out "$OUTPUT_DIR/a1_server_scaling_summary.csv"

  python3 "$QUALITY_SCRIPT" "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/mandatory_answer_quality.csv" \
    --summary-out "$OUTPUT_DIR/mandatory_answer_quality_summary.json"

  cat > "$OUTPUT_DIR/run_manifest.txt" <<EOF
track=A1
engines=${ENGINE_LABELS[*]}
engine_cache_sweeps=$ENGINE_CACHE_SWEEPS
engine_bin_specs=${ENGINE_BIN_SPECS:-}
engine_preload_specs=${ENGINE_PRELOAD_SPECS:-}
prompt_scope=limit_per_category_${LIMIT_PER_CATEGORY:-all}
mandatory_only=$MANDATORY_ONLY
engine_count=${#ENGINE_LABELS[@]}
model_count=${#MODEL_LABELS[@]}
config_count=${#CONFIG_SUITES[@]}
thread_sweep_count=$(count_words "$THREADS_LIST")
concurrency_sweep_count=$(count_words "$CONCURRENCY_LIST")
context_sweep_count=$(count_words "$CTX_SIZE_LIST")
decode_sweep_count=$(count_words "$MAX_TOKENS_LIST")
cache_sweep_count=$(count_words "$CACHE_SWEEP")
excluded_config_count=$EXCLUDED_CONFIG_COUNT
exclude_configs=$EXCLUDE_CONFIGS
summary=$OUTPUT_DIR/a1_server_summary.csv
scaling_summary=$OUTPUT_DIR/a1_server_scaling_summary.csv
quality=$OUTPUT_DIR/mandatory_answer_quality.csv
quality_summary=$OUTPUT_DIR/mandatory_answer_quality_summary.json
EOF
}

build_config_matrix

echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Engines: ${ENGINE_LABELS[*]}"
echo "[INFO] Mandatory-only prompts: $MANDATORY_ONLY"
echo "[INFO] Limit per category: ${LIMIT_PER_CATEGORY:-<unset>}"
echo "[INFO] Trials: $TRIALS measured, $WARMUP_TRIALS warmup"
echo "[INFO] Models: ${MODEL_LABELS[*]}"
echo "[INFO] Baseline model: $mandatory_label"
echo "[INFO] Engine cache sweeps: $ENGINE_CACHE_SWEEPS"
echo "[INFO] Thread sweep: $THREADS_LIST"
echo "[INFO] Concurrency sweep: $CONCURRENCY_LIST"
echo "[INFO] Context sweep: $CTX_SIZE_LIST"
echo "[INFO] Decode sweep: $MAX_TOKENS_LIST"
echo "[INFO] Excluded configs: $EXCLUDED_CONFIG_COUNT ($EXCLUDE_CONFIGS)"
echo "[INFO] Config count: ${#CONFIG_SUITES[@]}"

submit_array_if_requested "$@"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID:-}" == "0" ]]; then
  write_inventories
  write_config_list "$OUTPUT_DIR/config_matrix.csv"
fi

if [[ "$LIST_CONFIGS" == "1" || "$LIST_CONFIGS" == "true" ]]; then
  echo "CONFIG_COUNT=${#CONFIG_SUITES[@]}"
  echo "CONFIG_LAST_INDEX=$(( ${#CONFIG_SUITES[@]} - 1 ))"
  echo "CONFIG_MATRIX=$OUTPUT_DIR/config_matrix.csv"
  exit 0
fi

if [[ "$SUMMARY_ONLY" == "1" || "$SUMMARY_ONLY" == "true" ]]; then
  write_final_outputs
  echo "[OK] Summary complete: $OUTPUT_DIR"
  echo "[OK] Timing summary: $OUTPUT_DIR/a1_server_summary.csv"
  echo "[OK] Quality summary: $OUTPUT_DIR/mandatory_answer_quality_summary.json"
  exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_config_index "$SLURM_ARRAY_TASK_ID"
  echo "[OK] Array task $SLURM_ARRAY_TASK_ID complete: $OUTPUT_DIR"
  exit 0
fi

for idx in "${!CONFIG_SUITES[@]}"; do
  run_config_index "$idx"
done

write_final_outputs

echo "[OK] Track A1 server sweep complete: $OUTPUT_DIR"
echo "[OK] Timing summary: $OUTPUT_DIR/a1_server_summary.csv"
echo "[OK] Quality summary: $OUTPUT_DIR/mandatory_answer_quality_summary.json"
