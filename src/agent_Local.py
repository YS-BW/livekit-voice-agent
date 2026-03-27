import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import minimax, openai

ROOT_DIR = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT_DIR / "qwen-livekit-stt"
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from qwen_livekit_stt import QwenStreamingSTT
from assistant import Assistant
from metrics_hooks import (
    MetricsState,
    register_session_metrics_hooks,
    start_network_probe_task,
)
from metrics_logger import MetricsLogger

logger = logging.getLogger("agent")

load_dotenv(".env.local")

server = AgentServer()
METRICS_LOG_DIR = ROOT_DIR / "log"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext) -> None:
    '''Build the voice pipeline, join the room, and collect latency metrics.'''
    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_log_path = (
        METRICS_LOG_DIR / f"{session_stamp}_metrics_{ctx.job.id}_{ctx.job.room.sid}.jsonl"
    )
    metrics_logger = MetricsLogger(
        metrics_log_path,
        room=ctx.room.name if ctx.room else None,
        room_id=ctx.job.room.sid if ctx.job and ctx.job.room else None,
        job_id=ctx.job.id if ctx.job else None,
        agent_name="my-agent",
    )

    qwen_streaming_ws_url = os.getenv(
        "QWEN_STREAMING_STT_WS_URL", "ws://8.141.21.41:8001/ws"
    )
    logger.warning(
        "stt backend fixed",
        extra={"backend": "qwen_streaming_ws", "ws_url": qwen_streaming_ws_url},
    )
    llm_model = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://8.141.21.41:8000/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "fake-key")
    allow_interruptions = _env_bool("ALLOW_INTERRUPTIONS", False)
    preemptive_generation = _env_bool("PREEMPTIVE_GENERATION", False)
    resume_false_interruption = _env_bool("RESUME_FALSE_INTERRUPTION", False)
    min_interruption_duration = _env_float("MIN_INTERRUPTION_DURATION", 0.45)
    min_endpointing_delay = _env_float("MIN_ENDPOINTING_DELAY", 0.0)
    max_endpointing_delay = _env_float("MAX_ENDPOINTING_DELAY", 0.05)

    session = AgentSession(
        stt=QwenStreamingSTT(
            ws_url=qwen_streaming_ws_url,
            model=os.getenv("QWEN_STREAMING_STT_MODEL", "qwen3-asr-streaming"),
        ),
        llm=openai.LLM(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        ),
        # llm=openai.LLM(
        #     # ollama
        #     model="qwen2.5:0.5b",
        #     base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        #     api_key=llm_api_key,
        # ),
        tts=minimax.TTS(
            model="speech-02-turbo",
            voice="socialmedia_female_1_v1",
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat",
            speed=1.05,
        ),
        allow_interruptions=allow_interruptions,
        resume_false_interruption=resume_false_interruption,
        preemptive_generation=preemptive_generation,
        min_interruption_duration=min_interruption_duration,
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay,
        turn_detection="stt",
    )

    await ctx.connect()

    metrics_state = MetricsState()
    register_session_metrics_hooks(session, logger, metrics_logger, metrics_state)

    probe_task = start_network_probe_task(
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        state=metrics_state,
        logger=logger,
        metrics_logger=metrics_logger,
    )
    try:
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(sample_rate=16000),
            ),
        )
    finally:
        if probe_task is not None:
            probe_task.cancel()
            try:
                await probe_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    cli.run_app(server)
