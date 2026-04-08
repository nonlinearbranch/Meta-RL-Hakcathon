#!/usr/bin/env bash

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
IMAGE_NAME="${3:-heatshield_env-env:latest}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <hf_space_url> [repo_dir] [image_name]"
  exit 1
fi

PING_URL="${PING_URL%/}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

echo "== HeatShield submission validator =="
echo "Repo:  $REPO_DIR"
echo "Space: $PING_URL"
echo

echo "[1/4] Checking Space reset endpoint"
RESET_OUT="$(mktemp)"
trap 'rm -f "$RESET_OUT"' EXIT
HTTP_CODE="$(curl -s -o "$RESET_OUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || printf "000")"

if [ "$HTTP_CODE" != "200" ]; then
  echo "Reset endpoint failed with HTTP $HTTP_CODE"
  exit 1
fi
echo "OK"
echo

echo "[2/4] Building Docker image"
docker build -t "$IMAGE_NAME" -f "$REPO_DIR/server/Dockerfile" "$REPO_DIR"
echo "OK"
echo

echo "[3/4] Running OpenEnv local validation"
(cd "$REPO_DIR" && openenv validate .)
echo "OK"
echo

echo "[4/4] Running baseline inference"
(
  cd "$REPO_DIR"
  export LOCAL_IMAGE_NAME="$IMAGE_NAME"
  python inference.py
)
echo "OK"
