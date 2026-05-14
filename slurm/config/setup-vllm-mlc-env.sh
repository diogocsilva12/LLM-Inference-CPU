#!/usr/bin/env bash
# Optional vLLM/MLC Python environment setup helpers for Track A1 sweeps.
# Source this file from a SLURM wrapper after PROJECT_DIR is known.

ensure_python_venv() {
  local venv_dir="$1"
  local python_cmd="${2:-python3}"

  if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "[INFO] Creating Python venv: $venv_dir"
    mkdir -p "$(dirname "$venv_dir")"
    "$python_cmd" -m venv "$venv_dir"
  fi
}

python_import_ok() {
  local python_bin="$1"
  local module_name="$2"
  [[ -x "$python_bin" ]] && "$python_bin" -c "import ${module_name}" >/dev/null 2>&1
}

setup_vllm_env() {
  if [[ "${VLLM_BUILD:-0}" != "1" && "${VLLM_BUILD:-0}" != "true" ]]; then
    return 0
  fi

  local venv_dir="${VLLM_VENV_DIR:-$HOME/venvs/vllm-cpu}"
  local python_cmd="${VLLM_BOOTSTRAP_PYTHON:-python3}"
  local mode="${VLLM_BUILD_MODE:-pip}"
  local package="${VLLM_PACKAGE:-vllm}"
  local source_dir="${VLLM_SOURCE_DIR:-$HOME/src/vllm}"
  local repo_url="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
  local python_bin="$venv_dir/bin/python"

  ensure_python_venv "$venv_dir" "$python_cmd"
  export VLLM_PYTHON="$python_bin"

  if [[ "${VLLM_FORCE_REBUILD:-0}" != "1" ]] && python_import_ok "$python_bin" vllm; then
    echo "[INFO] vLLM already importable with $python_bin; skipping setup. Set VLLM_FORCE_REBUILD=1 to reinstall."
    return 0
  fi

  echo "[INFO] Installing/updating vLLM environment at $venv_dir with VLLM_BUILD_MODE=$mode"
  "$python_bin" -m pip install --upgrade pip wheel setuptools

  case "$mode" in
    pip)
      "$python_bin" -m pip install ${VLLM_PIP_FLAGS:-} "$package"
      ;;
    source)
      if [[ ! -d "$source_dir/.git" ]]; then
        mkdir -p "$(dirname "$source_dir")"
        git clone "$repo_url" "$source_dir"
      fi
      if [[ -n "${VLLM_GIT_REF:-}" ]]; then
        git -C "$source_dir" fetch --all --tags
        git -C "$source_dir" checkout "$VLLM_GIT_REF"
      fi
      if [[ -f "$source_dir/requirements/cpu.txt" ]]; then
        "$python_bin" -m pip install -r "$source_dir/requirements/cpu.txt"
      fi
      VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}" "$python_bin" -m pip install ${VLLM_PIP_FLAGS:-} -e "$source_dir"
      ;;
    custom)
      if [[ -z "${VLLM_BUILD_CMD:-}" ]]; then
        echo "VLLM_BUILD_MODE=custom requires VLLM_BUILD_CMD." >&2
        return 2
      fi
      VLLM_PYTHON="$python_bin" VLLM_VENV_DIR="$venv_dir" bash -lc "$VLLM_BUILD_CMD"
      ;;
    *)
      echo "Unknown VLLM_BUILD_MODE=$mode. Use pip, source, or custom." >&2
      return 2
      ;;
  esac

  "$python_bin" -c "import vllm; print('vllm import ok')"
}

setup_mlc_env() {
  if [[ "${MLC_BUILD:-0}" != "1" && "${MLC_BUILD:-0}" != "true" ]]; then
    return 0
  fi

  local venv_dir="${MLC_VENV_DIR:-$HOME/venvs/mlc-cpu}"
  local python_cmd="${MLC_BOOTSTRAP_PYTHON:-python3}"
  local mode="${MLC_BUILD_MODE:-pip}"
  local package="${MLC_PACKAGE:-mlc-llm}"
  local python_bin="$venv_dir/bin/python"

  ensure_python_venv "$venv_dir" "$python_cmd"
  export MLC_PYTHON="$python_bin"

  if [[ "${MLC_FORCE_REBUILD:-0}" != "1" ]] && python_import_ok "$python_bin" mlc_llm; then
    echo "[INFO] MLC-LLM already importable with $python_bin; skipping setup. Set MLC_FORCE_REBUILD=1 to reinstall."
    return 0
  fi

  echo "[INFO] Installing/updating MLC environment at $venv_dir with MLC_BUILD_MODE=$mode"
  "$python_bin" -m pip install --upgrade pip wheel setuptools

  case "$mode" in
    pip)
      "$python_bin" -m pip install ${MLC_PIP_FLAGS:-} "$package"
      ;;
    custom|source)
      if [[ -z "${MLC_BUILD_CMD:-}" ]]; then
        echo "MLC_BUILD_MODE=$mode requires MLC_BUILD_CMD. Use it for your TVM Unity + MLC source-build commands." >&2
        return 2
      fi
      MLC_PYTHON="$python_bin" MLC_VENV_DIR="$venv_dir" bash -lc "$MLC_BUILD_CMD"
      ;;
    *)
      echo "Unknown MLC_BUILD_MODE=$mode. Use pip, source, or custom." >&2
      return 2
      ;;
  esac

  "$python_bin" -c "import mlc_llm; print('mlc_llm import ok')"
}
