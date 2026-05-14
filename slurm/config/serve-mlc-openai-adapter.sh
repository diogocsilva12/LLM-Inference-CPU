#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
serve-mlc-openai-adapter.sh - llama-server flag adapter for MLC OpenAI server

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
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$model" ]]; then
  echo "Missing model path. Expected: -m /path/to/model" >&2
  exit 2
fi

python_bin="${MLC_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi

if ! "$python_bin" -c "import mlc_llm" >/dev/null 2>&1; then
  echo "MLC-LLM is not importable with $python_bin. Set MLC_PYTHON to your MLC venv python." >&2
  exit 2
fi

if [[ -n "$threads" ]]; then
  export OMP_NUM_THREADS="$threads"
fi

# The default serves in OpenAI-compatible mode on CPU. Additional tuning flags
# can be passed via MLC_EXTRA_ARGS if your local MLC build supports them.
extra_args=()
if [[ -n "${MLC_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$MLC_EXTRA_ARGS"
fi

exec "$python_bin" -m mlc_llm serve "$model" \
  --host "$host" \
  --port "$port" \
  --device cpu \
  "${extra_args[@]}"
