import argparse
import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import uvicorn
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger("qwen_asr_streaming_service")
app = FastAPI(title="Qwen3-ASR Streaming Service")

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2


@dataclass
class StreamSession:
    state: Any
    last_text: str = ""
    speech_started: bool = False
    voiced_ms: float = 0.0
    silence_ms: float = 0.0
    audio_duration: float = 0.0
    finalize_task: asyncio.Task[None] | None = None
    pre_speech_buffer: deque[tuple[bytes, float]] = field(default_factory=deque)
    pre_speech_buffer_ms: float = 0.0
    vad_remainder: bytes = b""


class StreamingASRService:
    def __init__(
        self,
        *,
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_new_tokens: int,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
        chunk_size_sec: float,
        energy_threshold: float,
        min_speech_ms: float,
        silence_timeout_ms: float,
        vad_mode: int,
        vad_frame_ms: int,
        pre_speech_pad_ms: float,
    ) -> None:
        self._asr = Qwen3ASRModel.LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
        )
        self._unfixed_chunk_num = unfixed_chunk_num
        self._unfixed_token_num = unfixed_token_num
        self._chunk_size_sec = chunk_size_sec
        self._energy_threshold = energy_threshold
        self._min_speech_ms = min_speech_ms
        self._silence_timeout_ms = silence_timeout_ms
        self._vad = webrtcvad.Vad(vad_mode)
        self._vad_frame_ms = vad_frame_ms
        self._pre_speech_pad_ms = pre_speech_pad_ms
        self._vad_frame_bytes = int(
            (SAMPLE_RATE * vad_frame_ms / 1000.0) * SAMPLE_WIDTH_BYTES
        )

    def _new_state(self) -> Any:
        return self._asr.init_streaming_state(
            unfixed_chunk_num=self._unfixed_chunk_num,
            unfixed_token_num=self._unfixed_token_num,
            chunk_size_sec=self._chunk_size_sec,
        )

    def new_session(self) -> StreamSession:
        return StreamSession(state=self._new_state())

    def reset_session_state(self, session: StreamSession) -> None:
        session.state = self._new_state()
        session.last_text = ""
        session.speech_started = False
        session.voiced_ms = 0.0
        session.silence_ms = 0.0
        session.audio_duration = 0.0
        session.finalize_task = None
        session.pre_speech_buffer.clear()
        session.pre_speech_buffer_ms = 0.0
        session.vad_remainder = b""

    def _segment_ms(self, seg: np.ndarray) -> float:
        return (len(seg) / SAMPLE_RATE) * 1000.0

    def _segment_rms(self, seg: np.ndarray) -> float:
        if seg.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(seg), dtype=np.float32)))

    def _pcm16_duration_s(self, pcm16: bytes) -> float:
        return len(pcm16) / SAMPLE_WIDTH_BYTES / float(SAMPLE_RATE)

    def _buffer_pre_speech(
        self, session: StreamSession, pcm16: bytes, chunk_ms: float
    ) -> None:
        if not pcm16:
            return

        session.pre_speech_buffer.append((pcm16, chunk_ms))
        session.pre_speech_buffer_ms += chunk_ms
        while (
            session.pre_speech_buffer
            and session.pre_speech_buffer_ms > self._pre_speech_pad_ms
        ):
            _, dropped_ms = session.pre_speech_buffer.popleft()
            session.pre_speech_buffer_ms -= dropped_ms

    def _flush_pre_speech(self, session: StreamSession) -> bytes:
        if not session.pre_speech_buffer:
            return b""

        pcm16 = b"".join(chunk for chunk, _ in session.pre_speech_buffer)
        session.pre_speech_buffer.clear()
        session.pre_speech_buffer_ms = 0.0
        return pcm16

    def _vad_voiced_ms(self, session: StreamSession, pcm16: bytes) -> float:
        if not pcm16:
            return 0.0

        buf = session.vad_remainder + pcm16
        voiced_ms = 0.0
        offset = 0
        while offset + self._vad_frame_bytes <= len(buf):
            frame = buf[offset : offset + self._vad_frame_bytes]
            if self._vad.is_speech(frame, SAMPLE_RATE):
                voiced_ms += self._vad_frame_ms
            offset += self._vad_frame_bytes

        session.vad_remainder = buf[offset:]
        return voiced_ms

    def _transcribe_bytes(
        self, session: StreamSession, pcm16: bytes
    ) -> dict[str, Any] | None:
        if not pcm16:
            return None

        seg = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._asr.streaming_transcribe(seg, session.state)
        session.audio_duration += self._pcm16_duration_s(pcm16)

        text = str(getattr(session.state, "text", "") or "")
        language = getattr(session.state, "language", None)
        if text and text != session.last_text:
            session.last_text = text
            return {
                "type": "interim",
                "text": text,
                "language": language,
            }
        return None

    def push_audio(self, session: StreamSession, pcm16: bytes) -> dict[str, Any] | None:
        if not pcm16:
            return None

        seg = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_ms = self._segment_ms(seg)
        rms = self._segment_rms(seg)
        vad_voiced_ms = self._vad_voiced_ms(session, pcm16)
        is_voiced = rms >= self._energy_threshold and vad_voiced_ms > 0.0

        if not session.speech_started:
            self._buffer_pre_speech(session, pcm16, chunk_ms)

        if is_voiced:
            session.voiced_ms += chunk_ms
            session.silence_ms = 0.0
            if session.finalize_task is not None:
                session.finalize_task.cancel()
                session.finalize_task = None
            if not session.speech_started and session.voiced_ms >= self._min_speech_ms:
                session.speech_started = True
                logger.info(
                    "speech started",
                    extra={
                        "voiced_ms": round(session.voiced_ms, 1),
                        "vad_voiced_ms": round(vad_voiced_ms, 1),
                        "rms": round(rms, 5),
                    },
                )
                return self._transcribe_bytes(session, self._flush_pre_speech(session))
        elif session.speech_started:
            session.silence_ms += chunk_ms

        if session.speech_started:
            return self._transcribe_bytes(session, pcm16)

        return None

    def should_finalize(self, session: StreamSession) -> bool:
        return session.speech_started and session.silence_ms >= self._silence_timeout_ms

    def finish(self, session: StreamSession) -> dict[str, Any]:
        self._asr.finish_streaming_transcribe(session.state)
        text = str(getattr(session.state, "text", "") or "")
        language = getattr(session.state, "language", None)
        session.last_text = text
        return {
            "type": "final",
            "text": text,
            "language": language,
        }


service: StreamingASRService | None = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    global service
    if service is None:
        await websocket.close(code=1011, reason="service not initialized")
        return

    await websocket.accept()
    session: StreamSession | None = None

    async def _finalize_current_session(*, reason: str, close_after: bool) -> None:
        nonlocal session
        if session is None:
            return

        current = session
        current.finalize_task = None
        payload = service.finish(current)
        logger.info(
            "speech finalized",
            extra={
                "reason": reason,
                "text_len": len(payload.get("text", "")),
                "audio_duration_s": round(current.audio_duration, 3),
            },
        )
        await websocket.send_json(payload)

        if close_after:
            return

        service.reset_session_state(current)
        session = current

    async def _schedule_finalize() -> None:
        nonlocal session
        if session is None:
            return
        try:
            if session is not None and service.should_finalize(session):
                await _finalize_current_session(
                    reason="silence_timeout", close_after=False
                )
        except asyncio.CancelledError:
            return

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")

                if msg_type == "start":
                    session = service.new_session()
                    logger.info("stream session started")
                    await websocket.send_json({"type": "started"})
                    continue

                if msg_type == "finish":
                    if session is None:
                        await websocket.send_json(
                            {"type": "error", "message": "session not started"}
                        )
                        continue
                    if session.finalize_task is not None:
                        session.finalize_task.cancel()
                        session.finalize_task = None
                    await _finalize_current_session(
                        reason="explicit_finish", close_after=True
                    )
                    break

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                await websocket.send_json(
                    {"type": "error", "message": f"unsupported message type: {msg_type}"}
                )
                continue

            if "bytes" in message and message["bytes"] is not None:
                if session is None:
                    await websocket.send_json(
                        {"type": "error", "message": "send start before audio"}
                    )
                    continue

                event = service.push_audio(session, message["bytes"])
                if event is not None:
                    await websocket.send_json(event)
                if service.should_finalize(session) and session.finalize_task is None:
                    session.finalize_task = asyncio.create_task(_schedule_finalize())
                continue

    except WebSocketDisconnect:
        logger.info("websocket client disconnected")
    except Exception:
        logger.exception("streaming session failed")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json(
                {"type": "error", "message": "internal server error"}
            )
    finally:
        if session is not None and session.finalize_task is not None:
            session.finalize_task.cancel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR streaming websocket service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--energy-threshold", type=float, default=0.015)
    parser.add_argument("--min-speech-ms", type=float, default=180.0)
    parser.add_argument("--silence-timeout-ms", type=float, default=280.0)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--vad-frame-ms", type=int, choices=[10, 20, 30], default=30)
    parser.add_argument("--pre-speech-pad-ms", type=float, default=240.0)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    global service

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("loading qwen3-asr streaming model")
    service = StreamingASRService(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
        energy_threshold=args.energy_threshold,
        min_speech_ms=args.min_speech_ms,
        silence_timeout_ms=args.silence_timeout_ms,
        vad_mode=args.vad_mode,
        vad_frame_ms=args.vad_frame_ms,
        pre_speech_pad_ms=args.pre_speech_pad_ms,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
