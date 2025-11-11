#!/bin/bash
set -e

echo "[start.sh] Starting CSM streaming container..."

if [ -n "${HUGGINGFACE_TOKEN}" ]; then
  echo "[start.sh] Logging into Hugging Face with token env..."
  huggingface-cli login --token "${HUGGINGFACE_TOKEN}" --add-to-git-credential <<< "n"
else
  echo "[start.sh] WARNING: HUGGINGFACE_TOKEN not set. Model downloads will fail."
fi

echo "[start.sh] Running setup.py to download models (idempotent)..."
python3 setup.py || echo "[start.sh] setup.py error (may already be downloaded). Continuing."

echo "[start.sh] Launching main.py (serves UI on port 8000)..."
python3 main.py
