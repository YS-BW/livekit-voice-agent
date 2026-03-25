import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import minimax, openai, silero

ROOT_DIR = Path(__file__).resolve().parents[2]
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
METRICS_LOG_DIR = Path(__file__).resolve().parents[2] / "log"


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

    session = AgentSession(
        stt=QwenStreamingSTT(
            ws_url=qwen_streaming_ws_url,
            model=os.getenv("QWEN_STREAMING_STT_MODEL", "qwen3-asr-streaming"),
        ),
        llm=openai.LLM(
            model="qwen-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        ),
        tts=minimax.TTS(
            model="speech-02-turbo",
            voice="socialmedia_female_1_v1",
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat",
            speed=1.05,
        ),
        preemptive_generation=True,
        min_interruption_duration=0.2,
        min_endpointing_delay=0.0,
        max_endpointing_delay=0.05,
        turn_detection="stt",
        vad=silero.VAD.load(
            min_speech_duration=0.04,
            min_silence_duration=0.06,
            activation_threshold=0.55,
            deactivation_threshold=0.45,
            prefix_padding_duration=0.12,
            max_buffered_speech=20.0,
            sample_rate=16000,
        ),
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
