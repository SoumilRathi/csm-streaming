# CHANGE: Use CUDA 12.4.1 Base Image (Matches PyTorch 2.5.1 stability)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-pip python3-venv \
      git ffmpeg libportaudio2 build-essential \
    && rm -rf /var/lib/apt/lists/*

# CORRECTED: Set working directory to the persistent RunPod path
WORKDIR /workspace 

# 1. Install PyTorch STABLE and Flash Attention in a separate layer
# This ensures PyTorch uses CUDA 12.4 binaries and Flash Attention links correctly.
RUN python3 -m pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install packaging psutil && \
    pip install flash-attn --no-build-isolation

# 2. Install the rest of the requirements
# We copy requirements.txt to the WORKDIR (/workspace)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Final Code Copy (Most frequently changed layer, maximizing cache use)
# This copies all your source code (including start.sh) into /workspace
COPY . .

# CORRECTED: Start script is now in /workspace, so chmod it there.
RUN chmod +x /workspace/start.sh
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# CORRECTED: Execute the start script from its new location in /workspace
CMD ["/workspace/start.sh"]