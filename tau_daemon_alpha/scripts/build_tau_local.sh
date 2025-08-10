#!/usr/bin/env bash
set -euo pipefail

# Build a local Docker image of IDNI tau-lang for internal testing only.
# Source: https://github.com/IDNI/tau-lang (licensed per Tau Language License)
# Do not distribute built artifacts.

IMAGE_NAME=${IMAGE_NAME:-tau-local}
REPO_URL=${REPO_URL:-https://github.com/IDNI/tau-lang}
CHECKOUT_REF=${CHECKOUT_REF:-main}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR="${ROOT_DIR}/.tau_lang_cache"
SRC_DIR="${CACHE_DIR}/tau-lang"

mkdir -p "$CACHE_DIR"

if [[ -d "$SRC_DIR/.git" ]]; then
  echo "Updating existing tau-lang clone..."
  git -C "$SRC_DIR" fetch --all --tags
else
  echo "Cloning tau-lang..."
  git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

echo "Checking out $CHECKOUT_REF..."
git -C "$SRC_DIR" checkout -q "$CHECKOUT_REF"

echo "Building Docker image $IMAGE_NAME from $SRC_DIR..."
docker build -t "$IMAGE_NAME" "$SRC_DIR"

echo "\n✓ Built image: $IMAGE_NAME"


