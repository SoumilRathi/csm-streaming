#!/bin/bash
set -e

echo "[start.sh] Starting CSM TTS server..."

if [ -n "${HUGGINGFACE_TOKEN}" ]; then
  echo "[start.sh] Logging into Hugging Face..."
  huggingface-cli login --token "${HUGGINGFACE_TOKEN}" --add-to-git-credential <<< "n"
else
  echo "[start.sh] WARNING: HUGGINGFACE_TOKEN not set."
fi

echo "[start.sh] Running setup.py to download models..."
python3 setup.py || echo "[start.sh] setup.py failed, continuing if cache is present."

echo "[start.sh] Launching tts_server.py on port 8000..."
python3 tts_server.py