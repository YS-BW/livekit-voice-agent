# Qwen3-TTS Service

Minimal HTTP service that keeps a `Qwen3TTSModel` loaded in memory and exposes:

- `GET /healthz`
- `GET /voices`
- `POST /tts`

## Install

```bash
cd qwen-tts-service
uv sync
```

## Run

```bash
uv run server.py \
  --model /root/my-vllm-py312-cu128/qwen/tts \
  --host 0.0.0.0 \
  --port 8091 \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --default-speaker Vivian \
  --default-language Chinese
```

If `flash-attn` is not installed, omit `--attn-implementation`.

## Test

```bash
curl http://127.0.0.1:8091/healthz
```

```bash
curl http://127.0.0.1:8091/voices
```

```bash
curl -X POST http://127.0.0.1:8091/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "language": "Chinese",
    "speaker": "Vivian",
    "instruct": "自然、亲切、中文客服风格。"
  }' \
  --output output.wav
```
