# CHANGE: Use CUDA 12.4.1 Base Image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-pip python3-venv \
      git ffmpeg libportaudio2 build-essential \
      rsync \
    && rm -rf /var/lib/apt/lists/*

# SAFE BUILD LOCATION: We build here so files aren't hidden by the volume mount
WORKDIR /app 

# 1. Install PyTorch STABLE and Flash Attention
RUN python3 -m pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install packaging psutil && \
    pip install flash-attn --no-build-isolation

# 2. Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Copy source code to /app (Safe Zone)
COPY . .

# 4. Make start script executable
RUN chmod +x /app/start.sh

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# 5. Launch the script from the safe zone
CMD ["/app/start.sh"]