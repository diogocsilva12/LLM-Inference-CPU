#!/usr/bin/env bash
#SBATCH --job-name=a1-gptoss-node-scaling
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=46
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${PROJECT_DIR:-}" ]]; then
  for candidate in "$SUBMIT_DIR" "$SUBMIT_DIR/.." "$SCRIPT_DIR/../.."; do
    if [[ -f "$candidate/prompts/track_a_prompts.json" ]]; then
      PROJECT_DIR="$(cd -- "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "Unable to locate project directory; set PROJECT_DIR=/path/to/aded-project-llamacpp." >&2
  exit 2
fi

BASE_SWEEP_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-node-scaling.sh"
WRAPPER_SCRIPT="$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-gptoss-node-scaling.sh"
if [[ ! -f "$BASE_SWEEP_SCRIPT" ]]; then
  echo "Missing base node-scaling script: $BASE_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
  echo "Missing GPT-OSS wrapper script: $WRAPPER_SCRIPT" >&2
  exit 2
fi

# GPT-OSS 120B Q8_0 is intentionally larger than one 32 GiB ARM node. This
# wrapper drives the existing llama.cpp RPC node-scaling harness with safer
# first-pass defaults: low context, low concurrency, short decode, and f16 KV.
# Override any exported setting at submission time for follow-up sweeps.
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-gptoss-120b-node-scaling}"
export MODEL_PATH="${MODEL_PATH:-/share/chatbot/models/gpt-oss-120b-Q8_0.gguf}"
export MODEL_LABEL="${MODEL_LABEL:-gpt-oss-120b-q8_0}"
export ENGINE_LABEL="${ENGINE_LABEL:-llamacpp-tq-rpc}"

# Default to the likely-fit range for GPT-OSS 120B Q8_0 on 32 GiB ARM nodes.
# 32+ node RPC groups have previously hit llama.cpp backend-count limits in
# this build, so keep the default sweep at <=24 nodes; override NODE_COUNTS if
# you intentionally want to probe larger groups after rebuilding/tuning RPC.
export NODE_COUNTS="${NODE_COUNTS:-6 8 12 16 24}"
export SUBMIT_NODE_SWEEP="${SUBMIT_NODE_SWEEP:-1}"
export SUBMIT_SUMMARY_JOB="${SUBMIT_SUMMARY_JOB:-1}"
export NODE_SWEEP_DEPENDENCY_TYPE="${NODE_SWEEP_DEPENDENCY_TYPE:-afterany}"

export THREADS="${THREADS:-46}"
export THREADS_BATCH="${THREADS_BATCH:-46}"
export CTX_SIZE="${CTX_SIZE:-2048}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"
export BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export TRIALS="${TRIALS:-1}"
export WARMUP_TRIALS="${WARMUP_TRIALS:-0}"
export TIMEOUT="${TIMEOUT:-7200}"
export SERVER_STARTUP_WAIT_SECONDS="${SERVER_STARTUP_WAIT_SECONDS:-3600}"
export MANDATORY_ONLY="${MANDATORY_ONLY:-1}"
export LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-1}"

# Child node-count jobs are submitted through this wrapper, so the wrapper's
# #SBATCH time applies unless the base submitter overrides it. GPT-OSS 120B can
# spend many minutes just loading tensors over RPC, so use explicit longer node
# job time and a shorter summary time by default.
export NODE_JOB_TIME="${NODE_JOB_TIME:-08:00:00}"
export SUMMARY_JOB_TIME="${SUMMARY_JOB_TIME:-00:30:00}"

# Use conservative KV cache defaults for first load. Once f16 works, compare
# CACHE_TYPE_K/V=turbo3 or turbo4 by overriding these at submission time.
export CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
export CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
export FLASH_ATTN="${FLASH_ATTN:-on}"
export FIT_PARAMS="${FIT_PARAMS:-off}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

# Keep the RPC-enabled build separate from non-RPC builds. AUTO_BUILD defaults
# to 0 so benchmark jobs do not accidentally spend allocation time compiling;
# set AUTO_BUILD=1 only when intentionally building build-slurm-rpc.
export ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm-rpc}"
export BUILD_TARGETS="${BUILD_TARGETS:-llama-server rpc-server}"
export AUTO_BUILD="${AUTO_BUILD:-0}"
export RPC_MAX_SERVERS="${RPC_MAX_SERVERS:-255}"
export RPC_USE_CACHE="${RPC_USE_CACHE:-1}"

# A64FX affinity defaults. Override or unset if they hurt a specific build.
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-0-45}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"

export PROJECT_DIR
export A1_NODE_SCALING_SUBMIT_SCRIPT="${A1_NODE_SCALING_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"

cat <<EOF
[GPTOSS_NODE_SCALING]
project_dir=$PROJECT_DIR
base_script=$BASE_SWEEP_SCRIPT
model_path=$MODEL_PATH
model_label=$MODEL_LABEL
node_counts=$NODE_COUNTS
threads=$THREADS
ctx_size=$CTX_SIZE
parallel_requests=$PARALLEL_REQUESTS
bench_concurrency=$BENCH_CONCURRENCY
max_tokens=$MAX_TOKENS
server_startup_wait_seconds=$SERVER_STARTUP_WAIT_SECONDS
node_job_time=$NODE_JOB_TIME
summary_job_time=$SUMMARY_JOB_TIME
cache_type_k=$CACHE_TYPE_K
cache_type_v=$CACHE_TYPE_V
submit_node_sweep=$SUBMIT_NODE_SWEEP
output_suffix=$OUTPUT_SUFFIX
EOF

exec "$BASE_SWEEP_SCRIPT" "$@"
