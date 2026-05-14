#!/usr/bin/env bash
#SBATCH --job-name=a1-submit-sweeps
#SBATCH --account=f202500001hpcvlabepicurea
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=FAIL

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

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to submit the sweep arrays." >&2
  exit 2
fi

ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
SUBMIT_ARRAY="${SUBMIT_ARRAY:-1}"
AUTO_BUILD="${AUTO_BUILD:-0}"

MODEL_DIR="$PROJECT_DIR/llamacpp-tq/models"
MODEL_FILES=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  "SmolLM2-360M-Instruct.Q4_K_M.gguf"
  "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"
  "gpt-oss-20b-Q4_K_M.gguf"
)
MODEL_LABELS=(
  "model-1-mandatory"
  "model-2"
  "model-3"
  "model-4"
)

MODEL_SPECS=""
for idx in "${!MODEL_FILES[@]}"; do
  model_path="$MODEL_DIR/${MODEL_FILES[$idx]}"
  if [[ ! -f "$model_path" ]]; then
    echo "Missing model file for ${MODEL_LABELS[$idx]}: $model_path" >&2
    exit 1
  fi

  if [[ -n "$MODEL_SPECS" ]]; then
    MODEL_SPECS+=";"
  fi
  MODEL_SPECS+="${MODEL_LABELS[$idx]}=$model_path"
done

SWEEP_SCRIPTS=(
  "$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep.sh"
  "$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh"
  "$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh"
)

for sweep_script in "${SWEEP_SCRIPTS[@]}"; do
  if [[ ! -f "$sweep_script" ]]; then
    echo "Missing sweep script: $sweep_script" >&2
    exit 1
  fi
done

EXPECTED_BINARIES=(
  "$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant/build-slurm/bin/llama-server"
  "$PROJECT_DIR/llamacpp-vanilla/llama.cpp/build-slurm/bin/llama-server"
  "$PROJECT_DIR/llamacpp-vanilla/llama.cpp/build-slurm-x86/bin/llama-server"
  "$PROJECT_DIR/llamacpp-vanilla/llama.cpp/build-blas/openblas-fujitsu/bin/llama-server"
  "$PROJECT_DIR/llamacpp-vanilla/llama.cpp/build-blas/blis/bin/llama-server"
)

for binary in "${EXPECTED_BINARIES[@]}"; do
  if [[ ! -x "$binary" ]]; then
    echo "Missing executable benchmark binary: $binary" >&2
    echo "Set AUTO_BUILD=1 to let non-BLAS sweeps build missing default binaries, or build it first." >&2
    exit 1
  fi
done

echo "[INFO] Project dir: $PROJECT_DIR"
echo "[INFO] Array concurrency per sweep: $ARRAY_CONCURRENCY"
echo "[INFO] Auto-build during sweeps: $AUTO_BUILD"
echo "[INFO] Models:"
for idx in "${!MODEL_FILES[@]}"; do
  echo "  ${MODEL_LABELS[$idx]}=$MODEL_DIR/${MODEL_FILES[$idx]}"
done

for sweep_script in "${SWEEP_SCRIPTS[@]}"; do
  echo "[SUBMIT] $sweep_script"
  sbatch \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",MODEL_SPECS="$MODEL_SPECS",SUBMIT_ARRAY="$SUBMIT_ARRAY",ARRAY_CONCURRENCY="$ARRAY_CONCURRENCY",AUTO_BUILD="$AUTO_BUILD",SUMMARY_ONLY=0,LIST_CONFIGS=0,OUTPUT_DIR= \
    "$sweep_script"
done
