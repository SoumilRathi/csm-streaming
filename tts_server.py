# tts_server.py
import io
import threading
import queue

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generator import load_csm_1b

app = FastAPI()

# single worker thread + job queue for all inference
_job_queue: "queue.Queue[tuple[str, int, queue.Queue]]" = queue.Queue()
_generator = None


class TTSRequest(BaseModel):
    text: str
    speaker: int = 0


def _worker():
    global _generator
    print("[worker] Loading CSM-1B...")
    _generator = load_csm_1b("cuda")
    print("[worker] CSM-1B ready.")

    while True:
        text, speaker, result_q = _job_queue.get()
        try:
            audio = _generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                stream=True,
            )
            result_q.put(audio)
        except Exception as e:
            result_q.put(e)


@app.on_event("startup")
def _startup():
    # start background worker thread once
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _synthesize_to_wav_bytes(text: str, speaker: int) -> bytes:
    result_q: "queue.Queue" = queue.Queue()
    _job_queue.put((text, speaker, result_q))
    result = result_q.get()

    if isinstance(result, Exception):
        raise result

    audio = result  # tensor [T]
    if audio is None or audio.numel() == 0:
        return b""

    buf = io.BytesIO()
    torchaudio.save(
        buf,
        audio.unsqueeze(0).cpu(),
        _generator.sample_rate,
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
