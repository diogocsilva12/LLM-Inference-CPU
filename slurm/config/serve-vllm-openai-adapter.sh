#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
serve-vllm-openai-adapter.sh - llama-server flag adapter for vLLM OpenAI server

Accepted llama-style options:
  -m <model>
  --host <host>
  --port <port>
  --threads <n>
  --threads-batch <n>
  -c <ctx_size>
  -np <parallel_requests>
  --cache-type-k <type>
  --cache-type-v <type>
  -h, --help

Cache support (for sweep compatibility): f16
EOF
}

model=""
host="127.0.0.1"
port="8000"
threads=""
ctx_size=""
parallel_requests=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -m)
      model="${2:-}"
      shift 2
      ;;
    --host)
      host="${2:-}"
      shift 2
      ;;
    --port)
      port="${2:-}"
      shift 2
      ;;
    --threads)
      threads="${2:-}"
      shift 2
      ;;
    --threads-batch|-ngl|-c|-np|--cache-type-k|--cache-type-v|--flash-attn|-fit)
      if [[ "$1" == "-c" ]]; then
        ctx_size="${2:-}"
      elif [[ "$1" == "-np" ]]; then
        parallel_requests="${2:-}"
      fi
      shift 2
      ;;
    *)
      # Keep unknown extra args compatible with existing sweep behavior.
      shift
      ;;
  esac
done

if [[ -z "$model" ]]; then
  echo "Missing model path. Expected: -m /path/to/model" >&2
  exit 2
fi

python_bin="${VLLM_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi

if ! "$python_bin" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not importable with $python_bin. Set VLLM_PYTHON to your vLLM venv python." >&2
  exit 2
fi

if [[ -n "$threads" ]]; then
  export OMP_NUM_THREADS="$threads"
fi
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}"
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-auto}"

extra_args=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$VLLM_EXTRA_ARGS"
fi

args=(
  -m vllm.entrypoints.openai.api_server
  --model "$model"
  --host "$host"
  --port "$port"
)
if [[ -n "$ctx_size" ]]; then
  args+=(--max-model-len "$ctx_size")
fi
if [[ -n "$parallel_requests" ]]; then
  args+=(--max-num-seqs "$parallel_requests")
fi

exec "$python_bin" "${args[@]}" "${extra_args[@]}"
