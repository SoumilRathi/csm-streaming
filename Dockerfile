FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-pip python3-venv \
      git \
      ffmpeg \
      libportaudio2 libportaudio-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install huggingface_hub

RUN chmod +x /app/start.sh

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]
