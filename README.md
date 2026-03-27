# 🎙️ AI Voice Sales

> 基于 LiveKit 的实时语音 Agent 工程。当前推荐的自部署模型组合是：
>
> - **ASR**: `qwen-asr`
> - **LLM**: `qwen2.5-7b-instruct`
> - **TTS**: `Qwen3-TTS-12Hz-1.7B-CustomVoice`

---

## 1️⃣ 项目定位

这个仓库主要解决一件事：把 **实时语音通话**、**ASR**、**LLM**、**TTS** 串成一条可以落地部署的语音销售 Agent 链路。

当前推荐的整体部署形态是：

- **普通服务器** 运行 `LiveKit Server`、`Agent`
- **GPU 服务器** 运行 `Qwen ASR`、`Qwen2.5-7B-Instruct`、`Qwen3-TTS-CustomVoice`

也就是说，**在线部署优先**，本地版本主要用于联调和开发。

---

## 2️⃣ Online / Local 版本说明

### 🌐 Online 版

这是推荐优先启动的版本，适合正式环境、联调环境、云服务器部署。

- `LiveKit Server` 放在普通服务器
- `Agent` 放在普通服务器
- `ASR / LLM / TTS` 放在 GPU 服务器
- Agent 通过 HTTP / WebSocket 去调用远端推理服务

这也是当前文档里默认推荐的启动方式。

### 💻 Local 版

这是开发机本地调试版，适合本机联调、排查问题、快速试跑。

- `Agent` 在本地启动
- `LiveKit Server` 可以连线上，也可以连你自己的测试环境
- `ASR / LLM / TTS` 可以继续走远端 GPU 服务
- 不建议把 `Qwen ASR / Qwen2.5-7B / Qwen3-TTS-1.7B` 全放本地 CPU 跑正式实时链路

---

## 3️⃣ 当前推荐模型组合

### 🧠 自部署推荐栈

| 组件 | 模型 | 推荐部署位置 | 说明 |
| :--- | :--- | :--- | :--- |
| ASR | `qwen-asr` | GPU 服务器 | 实时流式识别，供 LiveKit 自定义 STT 插件通过 WebSocket 调用 |
| LLM | `qwen2.5-7b-instruct` | GPU 服务器 | 通过 vLLM 提供 OpenAI 兼容接口 |
| TTS | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | GPU 服务器 | 当前推荐走原生 `qwen-tts` Python 推理服务，不推荐现在用 `vllm-omni` 直接接 |

### 📌 当前代码现状

- [`src/agent_Local.py`](./src/agent_Local.py) 已经接到 `Qwen ASR + qwen2.5-7b-instruct`
- [`src/agent_Local.py`](./src/agent_Local.py) 里的 **TTS 目前默认还是 MiniMax**
- [`qwen-tts-service/server.py`](./qwen-tts-service/server.py) 已经准备好，可单独启动 `Qwen3-TTS-CustomVoice` 服务
- 也就是说：**Qwen TTS 服务端已经有了，但 agent 侧还没有正式切到这个 HTTP TTS 服务**

---

## 4️⃣ 哪些必须跑在 GPU，哪些可以跑在普通服务器

### 🖥️ GPU 服务器

下面这些建议放在 GPU 服务器：

- `qwen-asr-streaming-service`
- `qwen2.5-7b-instruct` + `vLLM`
- `qwen-tts-service` + `Qwen3-TTS-12Hz-1.7B-CustomVoice`

原因很直接：

- `Qwen ASR` 需要持续流式推理
- `Qwen2.5-7B-Instruct` 实时对话对吞吐和首 token 延迟有要求
- `Qwen3-TTS-1.7B-CustomVoice` 虽然原生 Python API 可跑，但正式链路仍建议上 GPU

### ☁️ 普通服务器

下面这些可以放在普通服务器：

- `LiveKit Server`
- `Agent` 进程
- `Nginx / 反向代理`
- 业务接口、日志、监控

普通服务器主要负责编排和转发，不承担大模型推理。

### 🧪 本地 CPU

本地 CPU 可以做这些事：

- 跑 `Agent`
- 跑开发联调
- 验证服务接口是否通

但不建议把下面这些作为正式实时方案放 CPU：

- `qwen-asr`
- `qwen2.5-7b-instruct`
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`

---

## 5️⃣ 推荐启动顺序

### ✅ Online 版推荐顺序

1. 启动 `LiveKit Server`
2. 在 GPU 服务器启动 `qwen2.5-7b-instruct` 的 vLLM 服务
3. 在 GPU 服务器启动 `qwen-asr-streaming-service`
4. 在 GPU 服务器启动 `qwen-tts-service`
5. 在普通服务器启动 Agent

如果只是联调，至少也建议先把 **LLM / ASR** 起好，再启动 Agent。

---

## 6️⃣ GPU 服务器启动方式

### 6.1 启动 Qwen2.5-7B-Instruct

建议在你的 vLLM 环境里启动：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --tensor-parallel-size 1
```

Agent 默认按 OpenAI 兼容接口接入：

- `LLM_MODEL=qwen2.5-7b-instruct`
- `LLM_BASE_URL=http://<gpu-server>:8000/v1`
- `LLM_API_KEY=fake-key`

### 6.2 启动 Qwen ASR 流式服务

先安装依赖：

```bash
uv sync --group asr_service
```

再启动：

```bash
uv run qwen-asr-streaming-service/server.py \
  --model /path/to/qwen-asr \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.8 \
  --max-new-tokens 32
```

默认 WebSocket 地址：

```bash
ws://<gpu-server>:8001/ws
```

### 6.3 启动 Qwen3-TTS-CustomVoice 服务

先安装依赖：

```bash
uv sync --group tts_service
```

再启动：

```bash
uv run qwen-tts-service/server.py \
  --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --host 0.0.0.0 \
  --port 8091 \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --default-speaker Vivian \
  --default-language Chinese
```

说明：

- 当前验证通过的是 **原生 `qwen-tts` Python API 方案**
- `flash-attn` 没装时只是慢一些，不是必需
- `sox` 缺失会有 warning，但不一定阻塞推理
- 现阶段 **不建议把 Qwen3-TTS 主要接入方案写成 `vllm-omni`**

---

## 7️⃣ 普通服务器启动方式

### 7.1 启动 LiveKit Server

参考：

- [`docs/LiveKit-Server部署.md`](./docs/LiveKit-Server部署.md)

### 7.2 启动 Agent

#### Online 部署形态

如果是线上部署，推荐把 Agent 放在普通服务器，连接远端 GPU 推理服务。

当前仓库里需要区分两件事：

- [`src/agent_Online.py`](./src/agent_Online.py) 是历史线上入口，仍保留旧版云厂商配置
- [`src/agent_Local.py`](./src/agent_Local.py) 才是当前已经切到 `Qwen ASR + qwen2.5-7b-instruct` 的入口

所以如果你要跑当前推荐的 **Qwen 自部署链路**，实际应该优先参考：

```bash
uv run src/agent_Local.py dev
```

然后把 `.env.local` 配成远端 GPU 服务地址。

#### Local 开发形态

本地开发直接启动：

```bash
uv run src/agent_Local.py dev
```

---

## 8️⃣ 推荐环境变量

可以参考 `.env.local` / `.env.example`：

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

QWEN_STREAMING_STT_WS_URL=ws://127.0.0.1:8001/ws
QWEN_STREAMING_STT_MODEL=qwen3-asr-streaming

LLM_MODEL=qwen2.5-7b-instruct
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=fake-key

MINIMAX_API_KEY=
```

说明：

- 当前 Agent 代码默认 TTS 还是 `MiniMax`
- 如果要正式切成 `Qwen3-TTS-CustomVoice`，还需要把 agent 侧 TTS 插件改成调用 [`qwen-tts-service/server.py`](./qwen-tts-service/server.py) 的 HTTP 接口

---

## 9️⃣ 统一依赖管理

现在仓库统一使用根目录一个 `pyproject.toml`，不再拆多个 TOML。

### 📦 基础依赖

```bash
uv sync
```

### 📦 ASR 服务依赖

```bash
uv sync --group asr_service
```

### 📦 TTS 服务依赖

```bash
uv sync --group tts_service
```

### 📦 一次装全

```bash
uv sync --group asr_service --group tts_service
```

---

## 🔟 目录结构

```text
my-agent/
├─ src/                          # Agent 主逻辑
├─ qwen-livekit-stt/             # LiveKit 自定义 Qwen STT 插件
├─ qwen-asr-streaming-service/   # Qwen ASR 流式 WebSocket 服务
├─ qwen-tts-service/             # Qwen3-TTS HTTP 服务
├─ docs/                         # 部署文档
├─ pyproject.toml                # 唯一依赖配置文件
└─ README.md
```

---

## 1️⃣1️⃣ 相关文档

- [`docs/系统架构与数据流.md`](./docs/系统架构与数据流.md)
- [`docs/LLM部署.md`](./docs/LLM部署.md)
- [`docs/LiveKit-Server部署.md`](./docs/LiveKit-Server部署.md)
- [`qwen-asr-streaming-service/README.md`](./qwen-asr-streaming-service/README.md)
- [`qwen-tts-service/README.md`](./qwen-tts-service/README.md)
