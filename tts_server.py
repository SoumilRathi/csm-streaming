# tts_server.py
import io
import threading
import asyncio
import queue
from typing import Generator as PyGenerator, Tuple, Any
import time
import torch
import torchaudio
from hashlib import sha1
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generator import load_csm_1b, Segment

app = FastAPI()

# job_type: "wav" or "stream"
# job: (job_type, text, speaker, result_queue, enqueue_time)
_job_queue: "queue.Queue[Tuple[str, str, int, queue.Queue, float]]" = queue.Queue()
_generator = None

def load_audio(path, generator):
    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(
        wav.squeeze(0), sr, generator.sample_rate
    )
    return wav

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0


def _worker():
    global _generator
    print("[worker] Loading CSM-1B (with compile)…")
    _generator = load_csm_1b("cuda")
    # earlier first chunk
    _generator._stream_buffer_size = 5
    print("[worker] CSM-1B ready.")
    orig_encode = _generator._audio_tokenizer.encode
    import time as _t
    def _enc_wrap(wav):
        t0 = _t.time()
        out = orig_encode(wav)
        print(f"[enc] mimi encode took {(_t.time()-t0):.3f}s")
        return out
    _generator._audio_tokenizer.encode = _enc_wrap
    # ---------- Pre-encode reference audio once and reuse Mimi codes ----------
    def _load_wav(path: str):
        wav, sr = torchaudio.load(path)
        wav = torchaudio.functional.resample(wav.squeeze(0), sr, _generator.sample_rate)
        return wav.contiguous().cpu()

    _enc_cache = {}

    def _encode_once(wav_cpu_f32: torch.Tensor):
        key = sha1(wav_cpu_f32.numpy().tobytes()).hexdigest()
        if key not in _enc_cache:
            _enc_cache[key] = _generator._audio_tokenizer.encode(wav_cpu_f32)
        return _enc_cache[key]

    raw_refs = [
        (
            "You've got about 20 unread Slack messages. Want a quick digest?",
            "refs/ref_0.wav",
        ),
        (
            "That summary I made yesterday is still in drafts. Should I post it?",
            "refs/ref_1.wav",
        ),
        (
            "I just ran diagnostics; your GPU temperature's stable.",
            "refs/untitled #120.wav",
        ),
    ]

    encoded_context = []
    for text, p in raw_refs:
        try:
            wav = _load_wav(p)
            codes = _encode_once(wav)
            encoded_context.append(Segment(text=text, speaker=0, audio=codes))
        except Exception as e:
            print(f"[worker] Failed to prepare ref '{p}': {e}")

    # NOTE: Previous waveform-based context (re-encoded per request) kept for reference.
    # context_segments = [
    #     Segment(
    #         text="You've got about 20 unread Slack messages. Want a quick digest?",
    #         speaker=0,
    #         audio=load_audio("refs/ref_0.wav", _generator),
    #     ),
    #     # Segment(
    #     #     text="That summary I made yesterday is still in drafts. Should I post it?",
    #     #     speaker=0,
    #     #     audio=load_audio("refs/ref_1.wav", _generator),
    #     # ),
    #     Segment(
    #         text="I just ran diagnostics; your GPU temperature's stable.",
    #         speaker=0,
    #         audio=load_audio("refs/untitled #120.wav", _generator),
    #     ),
    #     # Segment(
    #     #     text="Wow, your CPU temperature just spiked. Either you're training a model or launching a rocket.",
    #     #     speaker=0,
    #     #     audio=load_audio("refs/ref_3.wav", _generator),
    #     # ),
    #     # Segment(
    #     #     text="You're on a roll today. Keep that streak going.",
    #     #     speaker=0,
    #     #     audio=load_audio("refs/ref_4.wav", _generator),
    #     # ),
    # ]

    while True:

        job_type, text, speaker, result_q, enqueue_time = _job_queue.get()
        worker_start_time = time.time()
        print(f"[worker] queue wait: {worker_start_time - enqueue_time:.3f}s")
        try:
            if job_type == "wav":
                # full generation
                audio = _generator.generate(
                    text=text,
                    speaker=speaker,
                    context=encoded_context,
                    stream=True,  # internal streaming, but we return full tensor
                )
                result_q.put(audio)

            elif job_type == "stream":
                # streaming generation: push chunks into result_q
                start = time.time()
                first_chunk = True # used for latency measurement
                for chunk in _generator.generate_stream(
                    text=text,
                    speaker=0,
                    context=encoded_context,
                    # context=[]
                ):
                    if first_chunk:
                        time_to_first_chunk = time.time()
                        print(f"[worker] time to first chunk: {time_to_first_chunk - start:.3f}s")
                        first_chunk = False
                    if chunk is None or chunk.numel() == 0:
                        continue
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.detach().to(torch.float32).cpu()
                    result_q.put(chunk)
                # sentinel for end-of-stream
                result_q.put(None)

            else:
                result_q.put(RuntimeError(f"Unknown job_type: {job_type}"))

        except Exception as e:
            # propagate error to caller; for stream also send sentinel
            result_q.put(e)
            if job_type == "stream":
                result_q.put(None)


@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ---------- blocking WAV endpoint ----------

def _synthesize_to_wav_bytes(text: str, speaker: int) -> bytes:
    result_q: "queue.Queue[Any]" = queue.Queue()
    enqueue_time = time.time()
    _job_queue.put(("wav", text, speaker, result_q, enqueue_time))
    result = result_q.get()
    dequeue_time = time.time()
    print(f"[worker] time to dequeue: {dequeue_time - enqueue_time:.3f}s")

    if isinstance(result, Exception):
        raise result

    audio = result  # torch.Tensor [T]
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

@app.get("/ping/stream")
async def ping_stream(req: Request):
    server_t0 = time.time()
    client_host = req.client.host if req.client else "unknown"
    print(f"[ping] handler entered from {client_host} at t0")

    async def gen():
        first = True
        for _ in range(5):
            if first:
                print(f"[ping] first bytes generated at +{time.time() - server_t0:.3f}s")
                first = False
            # 4096 bytes of junk
            yield b"\0" * 4096
            await asyncio.sleep(0.01)

    return StreamingResponse(gen(), media_type="application/octet-stream")

@app.post("/tts/wav")
def tts_wav(req: TTSRequest):
    wav_bytes = _synthesize_to_wav_bytes(req.text, req.speaker)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


# ---------- true streaming endpoint ----------

def _pcm_chunk_stream(text: str, speaker: int) -> PyGenerator[bytes, None, None]:
    """
    Yields raw float32 PCM bytes as CSM generates them, via the worker thread.
    """
    result_q: "queue.Queue[Any]" = queue.Queue()
    enqueue_time = time.time()
    # _job_queue.put(("stream", text, speaker, result_q, enqueue_time))
    _job_queue.put(("stream", text, speaker, result_q, enqueue_time))


    while True:
        item = result_q.get()
        if item is None:
            # end-of-stream sentinel
            break
        if isinstance(item, Exception):
            raise item

        chunk = item  # torch.Tensor on CPU, float32 [T]
        yield chunk.numpy().tobytes()


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    # sample_rate is stable once model is loaded
    headers = {"X-Sample-Rate": str(_generator.sample_rate)}
    return StreamingResponse(
        _pcm_chunk_stream(req.text, req.speaker),
        media_type="application/octet-stream",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000)
