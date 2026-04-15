#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_PARENT="${TARGET_PARENT:-$ROOT_DIR/llamacpp-vanilla}"
TARGET_DIR="${TARGET_DIR:-$TARGET_PARENT/llama.cpp}"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
BRANCH="${BRANCH:-master}"
SHALLOW="${SHALLOW:-0}"

if ! command -v git >/dev/null 2>&1; then
  echo "git command not found" >&2
  exit 1
fi

mkdir -p "$TARGET_PARENT"

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "Repository already exists at: $TARGET_DIR"
  echo "No changes made."
  exit 0
fi

echo "Cloning $REPO_URL (branch: $BRANCH) into $TARGET_DIR"
if [[ "$SHALLOW" == "1" ]]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
fi

echo "Clone successful: $TARGET_DIR"
