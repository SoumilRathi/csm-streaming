# tts_server.py
import io
import threading
import asyncio
import queue
from typing import Generator as PyGenerator, Tuple, Any, Optional, Dict
from dataclasses import dataclass, field
import time
import torch
import torchaudio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generator import load_csm_1b, Segment

app = FastAPI()

# =============================================================================
# Session State Management for Rolling Context
# =============================================================================

@dataclass
class SessionState:
    """Tracks state for a single TTS session to enable prosody continuity."""
    last_audio_tail: Optional[torch.Tensor] = None  # Last ~50ms for crossfade
    last_text: str = ""                              # For logging/debugging
    last_generation_time: float = 0.0                # Timestamp of last generation
    
    def is_expired(self, timeout_seconds: float = 300.0) -> bool:
        """Check if session has expired (default 5 minutes)."""
        if self.last_generation_time == 0.0:
            return False
        return (time.time() - self.last_generation_time) > timeout_seconds


class SessionManager:
    """
    Manages TTS sessions for prosody continuity across multiple generations.
    Thread-safe for use with the worker thread.
    """
    def __init__(self, session_timeout: float = 300.0):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._session_timeout = session_timeout
    
    def get(self, session_id: str) -> SessionState:
        """Get or create a session state."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState()
            return self._sessions[session_id]
    
    def update(self, session_id: str, audio_tail: Optional[torch.Tensor], text: str):
        """Update session state after generation."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState()
            session = self._sessions[session_id]
            session.last_audio_tail = audio_tail
            session.last_text = text
            session.last_generation_time = time.time()
    
    def clear(self, session_id: str):
        """Clear a specific session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    def cleanup_expired(self):
        """Remove expired sessions to prevent memory leaks."""
        with self._lock:
            expired = [
                sid for sid, state in self._sessions.items() 
                if state.is_expired(self._session_timeout)
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                print(f"[session] Cleaned up {len(expired)} expired sessions")
    
    def get_active_count(self) -> int:
        """Return number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Global session manager instance
_session_manager = SessionManager(session_timeout=300.0)

# =============================================================================
# Audio Processing Utilities
# =============================================================================

def crossfade(prev_tail: torch.Tensor, next_audio: torch.Tensor, fade_samples: int = 1200) -> torch.Tensor:
    """
    Apply overlap-add crossfade between the tail of the previous audio and the start of new audio.
    
    Args:
        prev_tail: Last ~50ms (1200 samples at 24kHz) from previous generation
        next_audio: New audio chunk to blend with
        fade_samples: Number of samples for the crossfade region (default 1200 = 50ms at 24kHz)
    
    Returns:
        Crossfaded audio tensor
    """
    if prev_tail is None or prev_tail.numel() == 0:
        return next_audio
    
    if next_audio.numel() < fade_samples:
        # If next audio is shorter than fade, just return it
        return next_audio
    
    # Ensure prev_tail is the right size
    if prev_tail.numel() > fade_samples:
        prev_tail = prev_tail[-fade_samples:]
    elif prev_tail.numel() < fade_samples:
        # Pad prev_tail if needed
        pad_size = fade_samples - prev_tail.numel()
        prev_tail = torch.cat([torch.zeros(pad_size), prev_tail])
    
    # Create fade curves
    device = next_audio.device
    fade_out = torch.linspace(1.0, 0.0, fade_samples, device=device)
    fade_in = torch.linspace(0.0, 1.0, fade_samples, device=device)
    
    # Ensure prev_tail is on the same device
    prev_tail = prev_tail.to(device)
    
    # Apply crossfade
    crossfaded = prev_tail * fade_out + next_audio[:fade_samples] * fade_in
    
    # Concatenate crossfaded region with rest of audio
    return torch.cat([crossfaded, next_audio[fade_samples:]])


def chunk_audio(audio: torch.Tensor, chunk_size: int = 4800) -> PyGenerator[torch.Tensor, None, None]:
    """
    Split audio tensor into chunks for streaming.
    
    Args:
        audio: Full audio tensor
        chunk_size: Samples per chunk (default 4800 = 200ms at 24kHz)
    
    Yields:
        Audio chunks
    """
    for i in range(0, audio.numel(), chunk_size):
        yield audio[i:i + chunk_size]


# =============================================================================
# Request Models
# =============================================================================

class TTSRequest(BaseModel):
    """Basic TTS request without session support."""
    text: str
    speaker: int = 0


class TTSStreamSessionRequest(BaseModel):
    """
    TTS request with session support for prosody continuity.
    
    Attributes:
        text: The text to synthesize
        speaker: Speaker ID (default 0)
        session_id: Unique session identifier for tracking state across requests
        has_more: Whether more sentences will follow in this turn
        lookahead_text: First ~3-5 words of the next sentence for prosodic continuity
    """
    text: str
    speaker: int = 0
    session_id: str = "default"
    has_more: bool = False
    lookahead_text: str = ""


# =============================================================================
# Global State
# =============================================================================

# job_type: "wav", "stream", or "stream_session"
# job formats vary by type - see _worker() for details
_job_queue: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
_generator = None

def load_audio(path, generator):
    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(
        wav.squeeze(0), sr, generator.sample_rate
    )
    return wav

def _worker():
    global _generator
    print("[worker] Loading CSM-1B (with compile)…")
    _generator = load_csm_1b("cuda")
    print("[worker] CSM-1B ready.")
    orig_encode = _generator._audio_tokenizer.encode
    import time as _t
    def _enc_wrap(wav):
        t0 = _t.time()
        out = orig_encode(wav)
        print(f"[enc] mimi encode took {(_t.time()-t0):.3f}s")
        return out
    _generator._audio_tokenizer.encode = _enc_wrap

    context_segments = [
        Segment(
            text="You've got about 20 unread Slack messages. Want a quick digest?",
            speaker=0,
            audio=load_audio("refs/ref_0.wav", _generator),
        ),
        # Segment(
        #     text="That summary I made yesterday is still in drafts. Should I post it?",
        #     speaker=0,
        #     audio=load_audio("refs/ref_1.wav", _generator),
        # ),
        Segment(
            text="I just ran diagnostics; your GPU temperature's stable.",
            speaker=0,
            audio=load_audio("refs/untitled #120.wav", _generator),
        ),
        # Segment(
        #     text="Wow, your CPU temperature just spiked. Either you're training a model or launching a rocket.",
        #     speaker=0,
        #     audio=load_audio("refs/ref_3.wav", _generator),
        # ),
        # Segment(
        #     text="You're on a roll today. Keep that streak going.",
        #     speaker=0,
        #     audio=load_audio("refs/ref_4.wav", _generator),
        # ),
    ]
    
    # Session cleanup counter - run cleanup every N jobs
    job_counter = 0
    CLEANUP_INTERVAL = 50

    while True:
        job = _job_queue.get()
        job_type = job[0]
        worker_start_time = time.time()
        
        # Periodic session cleanup
        job_counter += 1
        if job_counter >= CLEANUP_INTERVAL:
            _session_manager.cleanup_expired()
            job_counter = 0
        
        try:
            if job_type == "wav":
                # Format: (job_type, text, speaker, result_q, enqueue_time)
                _, text, speaker, result_q, enqueue_time = job
                print(f"[worker] queue wait: {worker_start_time - enqueue_time:.3f}s")
                
                # full generation
                audio = _generator.generate(
                    text=text,
                    speaker=speaker,
                    context=context_segments,
                    stream=True,  # internal streaming, but we return full tensor
                )
                result_q.put(audio)

            elif job_type == "stream":
                # Format: (job_type, text, speaker, result_q, enqueue_time)
                _, text, speaker, result_q, enqueue_time = job
                print(f"[worker] queue wait: {worker_start_time - enqueue_time:.3f}s")
                
                # streaming generation: push chunks into result_q
                start = time.time()
                first_chunk = True # used for latency measurement
                for chunk in _generator.generate_stream(
                    text=text,
                    speaker=speaker,
                    context=context_segments,
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

            elif job_type == "stream_session":
                # Format: (job_type, session_id, text, speaker, has_more, lookahead, result_q, enqueue_time)
                _, session_id, text, speaker, has_more, lookahead, result_q, enqueue_time = job
                print(f"[worker] queue wait: {worker_start_time - enqueue_time:.3f}s")
                print(f"[worker] stream_session: session={session_id}, has_more={has_more}, lookahead={repr(lookahead[:30] + '...' if len(lookahead) > 30 else lookahead)}")
                
                start = time.time()
                
                # Get session state for crossfading
                session = _session_manager.get(session_id)
                
                # Build generation text with lookahead for prosodic continuity
                gen_text = text
                if lookahead:
                    gen_text = f"{text} {lookahead}"
                    print(f"[worker] generating with lookahead: '{gen_text[:60]}...'")
                
                # Generate full audio with lookahead context
                audio_chunks = []
                first_chunk = True
                for chunk in _generator.generate_stream(
                    text=gen_text,
                    speaker=speaker,
                    context=context_segments,
                ):
                    if first_chunk:
                        time_to_first_chunk = time.time()
                        print(f"[worker] time to first chunk: {time_to_first_chunk - start:.3f}s")
                        first_chunk = False
                    if chunk is None or chunk.numel() == 0:
                        continue
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.detach().to(torch.float32).cpu()
                    audio_chunks.append(chunk)
                
                if not audio_chunks:
                    print("[worker] Warning: no audio generated")
                    result_q.put(None)
                    continue
                
                # Concatenate all chunks
                full_audio = torch.cat(audio_chunks)
                print(f"[worker] full audio: {full_audio.numel()} samples ({full_audio.numel() / _generator.sample_rate:.2f}s)")
                
                # Truncate to actual text boundary (estimate based on token ratio)
                if lookahead and len(lookahead) > 0:
                    try:
                        main_tokens = len(_generator._text_tokenizer.encode(f"[{speaker}]{text}"))
                        total_tokens = len(_generator._text_tokenizer.encode(f"[{speaker}]{gen_text}"))
                        ratio = main_tokens / total_tokens if total_tokens > 0 else 1.0
                        
                        # Apply ratio with a small buffer to avoid cutting mid-phoneme
                        # Add ~100ms (2400 samples) buffer
                        truncate_at = min(
                            int(full_audio.numel() * ratio) + 2400,
                            full_audio.numel()
                        )
                        
                        original_length = full_audio.numel()
                        full_audio = full_audio[:truncate_at]
                        print(f"[worker] truncated from {original_length} to {truncate_at} samples (ratio={ratio:.2f})")
                    except Exception as e:
                        print(f"[worker] Warning: truncation failed: {e}, using full audio")
                
                # Apply crossfade with previous session tail
                if session.last_audio_tail is not None:
                    original_length = full_audio.numel()
                    full_audio = crossfade(session.last_audio_tail, full_audio, fade_samples=1200)
                    print(f"[worker] crossfaded with previous tail, audio now {full_audio.numel()} samples")
                
                # Store tail for next chunk if more is coming
                if has_more:
                    # Store last 50ms (1200 samples at 24kHz)
                    tail_samples = min(1200, full_audio.numel())
                    _session_manager.update(
                        session_id,
                        audio_tail=full_audio[-tail_samples:].clone(),
                        text=text
                    )
                    print(f"[worker] stored {tail_samples} samples for next crossfade")
                else:
                    # Clear session tail - this turn is complete
                    _session_manager.update(session_id, audio_tail=None, text=text)
                    print(f"[worker] cleared session tail (turn complete)")
                
                # Stream chunks to client
                chunk_size = 4800  # 200ms at 24kHz
                for audio_chunk in chunk_audio(full_audio, chunk_size):
                    result_q.put(audio_chunk)
                
                # sentinel for end-of-stream
                result_q.put(None)
                
                total_time = time.time() - start
                print(f"[worker] stream_session complete: {full_audio.numel() / _generator.sample_rate:.2f}s audio in {total_time:.2f}s")

            else:
                result_q = job[3] if len(job) > 3 else None
                if result_q:
                    result_q.put(RuntimeError(f"Unknown job_type: {job_type}"))

        except Exception as e:
            import traceback
            print(f"[worker] Error: {e}")
            traceback.print_exc()
            # propagate error to caller; for stream types also send sentinel
            result_q = job[3] if job_type == "wav" else (job[6] if job_type == "stream_session" else job[3])
            if result_q:
            result_q.put(e)
                if job_type in ("stream", "stream_session"):
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


# ---------- session-aware streaming endpoint ----------

def _pcm_chunk_stream_session(
    session_id: str,
    text: str, 
    speaker: int,
    has_more: bool,
    lookahead_text: str
) -> PyGenerator[bytes, None, None]:
    """
    Yields raw float32 PCM bytes with session-aware prosody continuity.
    
    This endpoint supports:
    - Lookahead text for avoiding end-of-sentence prosodic artifacts
    - Crossfading between consecutive generations in the same session
    - Automatic session state management
    """
    result_q: "queue.Queue[Any]" = queue.Queue()
    enqueue_time = time.time()
    
    # Job format: (job_type, session_id, text, speaker, has_more, lookahead, result_q, enqueue_time)
    _job_queue.put((
        "stream_session", 
        session_id, 
        text, 
        speaker, 
        has_more, 
        lookahead_text, 
        result_q, 
        enqueue_time
    ))

    while True:
        item = result_q.get()
        if item is None:
            # end-of-stream sentinel
            break
        if isinstance(item, Exception):
            raise item

        chunk = item  # torch.Tensor on CPU, float32 [T]
        yield chunk.numpy().tobytes()


@app.post("/tts/stream/session")
def tts_stream_session(req: TTSStreamSessionRequest):
    """
    Session-aware streaming TTS endpoint with prosody continuity.
    
    This endpoint addresses the "sentence splitting artifact" where splitting
    text into sentences causes prosodic discontinuities (pitch declination,
    final lengthening) at chunk boundaries.
    
    Usage:
    1. For each sentence, include 3-5 words of lookahead from the next sentence
    2. Set has_more=True if more sentences will follow
    3. Use consistent session_id across a conversation turn
    
    Example flow for "Hello world. How are you?":
    - Request 1: text="Hello world.", lookahead_text="How are", has_more=True
    - Request 2: text="How are you?", lookahead_text="", has_more=False
    
    The model generates "world." knowing "How are" follows, avoiding the
    end-of-utterance prosodic pattern.
    """
    headers = {
        "X-Sample-Rate": str(_generator.sample_rate),
        "X-Session-Id": req.session_id,
    }
    return StreamingResponse(
        _pcm_chunk_stream_session(
            session_id=req.session_id,
            text=req.text,
            speaker=req.speaker,
            has_more=req.has_more,
            lookahead_text=req.lookahead_text,
        ),
        media_type="application/octet-stream",
        headers=headers,
    )


# ---------- session management endpoints ----------

@app.delete("/tts/session/{session_id}")
def delete_session(session_id: str):
    """Clear a specific session's state."""
    _session_manager.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/tts/sessions/stats")
def get_session_stats():
    """Get statistics about active sessions."""
    return {
        "active_sessions": _session_manager.get_active_count(),
    }


@app.post("/tts/sessions/cleanup")
def cleanup_sessions():
    """Manually trigger cleanup of expired sessions."""
    _session_manager.cleanup_expired()
    return {
        "status": "cleanup_complete",
        "remaining_sessions": _session_manager.get_active_count(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000)
