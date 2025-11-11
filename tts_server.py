# tts_server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import torch
import torchaudio

from generator import load_csm_1b  # or load_csm_1b_local if that's what you're using

app = FastAPI()
generator = None


class TTSRequest(BaseModel):
    text: str
    speaker: int = 0


@app.on_event("startup")
def _load_model():
    global generator
    # one-time load on GPU
    generator = load_csm_1b("cuda")
    print("CSM-1B loaded on CUDA")


def _synthesize_to_wav_bytes(text: str, speaker: int) -> bytes:
    # full generation (you can flip stream=True + generate_stream later)
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        stream=True,           # keep model’s internal streaming optimizations
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000)
