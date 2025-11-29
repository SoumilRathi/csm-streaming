#!/bin/bash
set -e

echo "[start.sh] Checking /workspace..."

# 1. THE BOOTSTRAP: Move code to persistent storage if needed
# If generator.py is missing from /workspace, we assume it's an empty volume.
if [ ! -f "/workspace/generator.py" ]; then
    echo "[start.sh] Persistent volume is empty. Copying app code from /app to /workspace..."
    
    # Copy everything from /app to /workspace
    # -a: archive mode (preserves permissions)
    # -v: verbose
    # --exclude: don't overwrite the .venv or huge git history if you don't want to
    rsync -av /app/ /workspace/
    
    echo "[start.sh] Copy complete."
else
    echo "[start.sh] Code already exists in /workspace. Skipping copy."
fi

# 2. SWITCH EXECUTION TO PERSISTENT STORAGE
# Now we go to the persistent folder. Any changes you make here will be saved.
cd /workspace
echo "[start.sh] Current directory: $(pwd)"

# 3. RUN APPLICATION
if [ -n "${HUGGINGFACE_TOKEN}" ]; then
  echo "[start.sh] Logging into Hugging Face..."
  huggingface-cli login --token "${HUGGINGFACE_TOKEN}" --add-to-git-credential <<< "n"
fi

if [ ! -d "models" ]; then
    echo "[start.sh] Running setup.py to download models..."
    python3 setup.py || echo "[start.sh] setup.py failed (maybe models exist?), continuing."
fi

echo "[start.sh] Launching tts_server.py..."
# This is now running the file located at /workspace/tts_server.py
python3 tts_server.py