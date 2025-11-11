# tts_server.py
import io
from typing import Generator as PyGenerator

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generator import load_csm_1b

app = FastAPI()
generator = None


class TTSRequest(BaseModel):
    text: str
    speaker: int = 0


@app.on_event("startup")
def _load_model():
    global generator
    print("[tts_server] Loading CSM-1B...")
    generator = load_csm_1b("cuda")
    print("[tts_server] CSM-1B ready.")


# ---- old blocking endpoint (keep it) ----
def _synthesize_to_wav_bytes(text: str, speaker: int) -> bytes:
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        stream=True,  # internal streaming, but we wait for full result
    )
    if audio is None or audio.numel() == 0:
        return b""

    buf = io.BytesIO()
    torchaudio.save(
        buf,
        audio.unsqueeze(0).cpu(),
        generator.sample_rate,
        format="wav",
    )
    buf.seek(0)
    return buf.read()


@app.post("/tts/wav")
def tts_wav(req: TTSRequest):
    wav_bytes = _synthesize_to_wav_bytes(req.text, req.speaker)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


# ---- NEW: true streaming endpoint ----
def _pcm_chunk_stream(text: str, speaker: int) -> PyGenerator[bytes, None, None]:
    """
    Yields raw float32 PCM bytes in small chunks as CSM generates them.
    Client must know sample_rate (we send it via header).
    """
    for chunk in generator.generate_stream(
        text=text,
        speaker=speaker,
        context=[],
    ):
        if chunk is None or chunk.numel() == 0:
            continue

        # Ensure CPU float32
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().to(torch.float32).cpu()

        # Convert to bytes (float32 little-endian)
        yield chunk.numpy().tobytes()


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    headers = {"X-Sample-Rate": str(generator.sample_rate)}
    return StreamingResponse(
        _pcm_chunk_stream(req.text, req.speaker),
        media_type="application/octet-stream",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000)