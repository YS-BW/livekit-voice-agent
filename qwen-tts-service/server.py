import argparse
import asyncio
import io
import logging
from typing import Any

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("qwen_tts_service")
app = FastAPI(title="Qwen3-TTS Service")


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: str | None = None
    language: str | None = None
    instruct: str | None = None
    max_new_tokens: int | None = None
    top_p: float | None = None
    temperature: float | None = None


class QwenTTSService:
    def __init__(
        self,
        *,
        model_path: str,
        device_map: str,
        dtype: str,
        attn_implementation: str | None,
        default_speaker: str | None,
        default_language: str | None,
        default_instruct: str | None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": _resolve_dtype(dtype),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        logger.info("loading qwen3-tts model", extra={"model_path": model_path})
        self._model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        self._default_speaker = default_speaker
        self._default_language = default_language
        self._default_instruct = default_instruct
        self._lock = asyncio.Lock()

    def supported_speakers(self) -> list[str]:
        try:
            speakers = self._model.get_supported_speakers()
        except Exception:
            logger.exception("failed to read supported speakers")
            return []
        return list(speakers or [])

    def supported_languages(self) -> list[str]:
        try:
            languages = self._model.get_supported_languages()
        except Exception:
            logger.exception("failed to read supported languages")
            return []
        return list(languages or [])

    def _synthesize(self, request: TTSRequest) -> tuple[bytes, int]:
        generation_kwargs: dict[str, Any] = {}
        if request.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = request.max_new_tokens
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature

        speaker = request.speaker or self._default_speaker
        language = request.language or self._default_language
        instruct = request.instruct if request.instruct is not None else self._default_instruct
        if not speaker:
            raise ValueError("speaker is required")

        wavs, sample_rate = self._model.generate_custom_voice(
            text=request.text,
            language=language or "Auto",
            speaker=speaker,
            instruct=instruct or "",
            **generation_kwargs,
        )
        if not wavs:
            raise RuntimeError("model returned empty audio")

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sample_rate, format="WAV")
        return buf.getvalue(), sample_rate

    async def synthesize(self, request: TTSRequest) -> tuple[bytes, int]:
        async with self._lock:
            return await run_in_threadpool(self._synthesize, request)


service: QwenTTSService | None = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/voices")
async def voices() -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return {
        "speakers": service.supported_speakers(),
        "languages": service.supported_languages(),
    }


@app.post("/tts")
async def tts(request: TTSRequest) -> Response:
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")

    try:
        wav_bytes, sample_rate = await service.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("tts generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {"X-Audio-Sample-Rate": str(sample_rate)}
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--default-speaker", default="Vivian")
    parser.add_argument("--default-language", default="Chinese")
    parser.add_argument("--default-instruct", default="")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    global service

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    service = QwenTTSService(
        model_path=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        default_speaker=args.default_speaker,
        default_language=args.default_language,
        default_instruct=args.default_instruct,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
