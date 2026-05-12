#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  slurm/submit-llama.sh build     <profile> [repo_dir]
  slurm/submit-llama.sh run       <profile> [model_path]
  slurm/submit-llama.sh benchmark <profile> [model_path]
  slurm/submit-llama.sh prebuild [profiles...]

Profiles:
  arm-cpu     ARM CPU nodes (normal-arm)
  x86-cpu     x86 CPU nodes (override defaults if needed)
  a100-cuda   A100 GPU nodes (normal-a100-40)

Examples:
  slurm/submit-llama.sh build arm-cpu
  slurm/submit-llama.sh build a100-cuda
  slurm/submit-llama.sh run a100-cuda /path/model.gguf
  slurm/submit-llama.sh benchmark a100-cuda /path/model.gguf
  slurm/submit-llama.sh prebuild arm-cpu a100-cuda

Env overrides:
  ARM_ACCOUNT, ARM_PARTITION, ARM_CPUS
  X86_ACCOUNT, X86_PARTITION, X86_CPUS
  A100_ACCOUNT, A100_PARTITION, A100_CPUS, A100_GPUS
  BUILD_TIME, RUN_TIME, BENCH_TIME, REPO_DIR, BUILD_DIR, MODEL_PATH, PORT,
  HOST, N_GPU_LAYERS, TARGET, LOCAL_TUNNEL_PORT, CACHE_SWEEP,
  PROMPT_TOKENS, GEN_TOKENS, REPETITIONS, OUTPUT_DIR
EOF
}

ACTION="${1:-}"
PROFILE="${2:-}"
EXTRA="${3:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build-llama-matrix.sh"
RUN_SCRIPT="$SCRIPT_DIR/run-llama-matrix.sh"
BENCH_SCRIPT="$SCRIPT_DIR/benchmark-llama-matrix.sh"

if [[ "$ACTION" == "" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$ACTION" == "prebuild" ]]; then
  shift || true
  if [[ "$#" -eq 0 ]]; then
    set -- arm-cpu a100-cuda
  fi
  for p in "$@"; do
    "$0" build "$p"
  done
  exit 0
fi

if [[ "$ACTION" != "build" && "$ACTION" != "run" && "$ACTION" != "benchmark" ]]; then
  echo "Unknown action: $ACTION" >&2
  usage
  exit 1
fi

if [[ -z "$PROFILE" ]]; then
  echo "Missing profile" >&2
  usage
  exit 1
fi

TARGET="$PROFILE"
NODES=1
NTASKS=1
GPUS=0

case "$PROFILE" in
  arm-cpu)
    ACCOUNT="${ARM_ACCOUNT:-f202500001hpcvlabepicurea}"
    PARTITION="${ARM_PARTITION:-normal-arm}"
    CPUS="${ARM_CPUS:-48}"
    ;;
  x86-cpu)
    ACCOUNT="${X86_ACCOUNT:-f202500001hpcvlabepicurex}"
    PARTITION="${X86_PARTITION:-normal-x86}"
    CPUS="${X86_CPUS:-32}"
    ;;
  a100-cuda)
    ACCOUNT="${A100_ACCOUNT:-f202500010hpcvlabuminhog}"
    PARTITION="${A100_PARTITION:-normal-a100-40}"
    CPUS="${A100_CPUS:-32}"
    GPUS="${A100_GPUS:-1}"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

if [[ "$ACTION" == "build" ]]; then
  TIME="${BUILD_TIME:-04:00:00}"
  SCRIPT="$BUILD_SCRIPT"
elif [[ "$ACTION" == "run" ]]; then
  TIME="${RUN_TIME:-48:00:00}"
  SCRIPT="$RUN_SCRIPT"
else
  TIME="${BENCH_TIME:-04:00:00}"
  SCRIPT="$BENCH_SCRIPT"
fi

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --nodes="$NODES"
  --ntasks="$NTASKS"
  --cpus-per-task="$CPUS"
  --time="$TIME"
  --output="%x_%j.out"
  --error="%x_%j.err"
)
if [[ "$GPUS" != "0" ]]; then
  SBATCH_ARGS+=( --gpus="$GPUS" )
fi

EXPORTS=("TARGET=$TARGET")
if [[ -n "${REPO_DIR:-}" ]]; then
  EXPORTS+=("REPO_DIR=$REPO_DIR")
fi
if [[ -n "${BUILD_DIR:-}" ]]; then
  EXPORTS+=("BUILD_DIR=$BUILD_DIR")
fi
if [[ -n "${PORT:-}" ]]; then
  EXPORTS+=("PORT=$PORT")
fi
if [[ -n "${HOST:-}" ]]; then
  EXPORTS+=("HOST=$HOST")
fi
if [[ -n "${LOCAL_TUNNEL_PORT:-}" ]]; then
  EXPORTS+=("LOCAL_TUNNEL_PORT=$LOCAL_TUNNEL_PORT")
fi
if [[ -n "${N_GPU_LAYERS:-}" ]]; then
  EXPORTS+=("N_GPU_LAYERS=$N_GPU_LAYERS")
fi
if [[ "$ACTION" == "build" && -n "$EXTRA" ]]; then
  EXPORTS+=("REPO_DIR=$EXTRA")
fi
if [[ "$ACTION" == "run" || "$ACTION" == "benchmark" ]]; then
  if [[ -n "${MODEL_PATH:-}" ]]; then
    EXPORTS+=("MODEL_PATH=$MODEL_PATH")
  fi
  if [[ -n "$EXTRA" ]]; then
    EXPORTS+=("MODEL_PATH=$EXTRA")
  fi
fi
for name in CACHE_SWEEP PROMPT_TOKENS GEN_TOKENS REPETITIONS OUTPUT_DIR BENCH_BIN OUTPUT_FORMAT BATCH_SIZE UBATCH_SIZE FLASH_ATTN THREADS; do
  if [[ -n "${!name:-}" ]]; then
    EXPORTS+=("$name=${!name}")
  fi
done

EXPORT_STRING="ALL"
for e in "${EXPORTS[@]}"; do
  EXPORT_STRING+=",$e"
done

echo "[INFO] Submitting $ACTION for profile=$PROFILE"
echo "[INFO] Partition=$PARTITION Account=$ACCOUNT CPUs=$CPUS GPUs=$GPUS Time=$TIME"
JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" --export="$EXPORT_STRING" "$SCRIPT")
echo "[OK] Submitted job: $JOB_ID"
