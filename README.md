# 🎙️ AI Voice Sales

> 基于 LiveKit 的实时语音 Agent 工程。
>
> 当前 README 只保留 **Online 版本** 的启动说明。

---

## 1️⃣ 最重要的前提

在启动 Agent 之前，必须先有一个可用的 `LiveKit Server`。

没有 `LiveKit Server`：

- Agent 不能加入房间
- 前端不能发起测试会话
- 整条语音链路无法工作

`LiveKit Server` 部署文档：

- [`docs/LiveKit-Server部署.md`](./docs/LiveKit-Server部署.md)

---

## 2️⃣ 当前推荐方式

当前仓库推荐直接使用 **Online 版本**。

原因：

- 更适合从零开始
- 更容易先跑通整条链路
- 配置完成后可以直接启动

当前 Online 版默认链路：

- STT: 火山 `BigModelSTT`
- LLM: DashScope 兼容接口
- TTS: MiniMax

入口文件：

- [`src/agent_Online.py`](./src/agent_Online.py)

---

## 3️⃣ Online 版从零启动

### 3.1 先准备 LiveKit Server

先部署并确认 `LiveKit Server` 可用。

需要准备：

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

参考文档：

- [`docs/LiveKit-Server部署.md`](./docs/LiveKit-Server部署.md)

### 3.2 再准备前端

如果你使用 `uv run src/agent_Online.py dev` 方式启动 Agent，还需要一个前端页面来发起测试会话。

当前测试环境使用：

- [livekit-agent-ui](https://livekit.com/blog/design-voice-ai-interfaces-with-agents-ui)

说明：

- 前端需要你自行部署
- 部署完成后，需要把前端访问地址填到你的测试环境说明里

示例：

```text
测试前端地址：
https://your-livekit-agent-ui.example.com
```

### 3.3 配置环境变量

Online 版实际需要这些变量：

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

VOLCENGINE_STT_APP_ID=
VOLCENGINE_STT_ACCESS_TOKEN=
VOLCENGINE_BIGMODEL_STT_MODEL=bigmodel

DASHSCOPE_API_KEY=
MINIMAX_API_KEY=
```

环境变量模板见：

- [`.env.example`](./.env.example)

### 3.4 安装依赖

```bash
uv sync
```

### 3.5 启动 Agent

```bash
uv run src/agent_Online.py dev
```

### 3.6 最小检查清单

1. `LiveKit Server` 可用
2. 前端 `livekit-agent-ui` 可访问
3. Agent 已启动
4. 火山 STT 密钥正确
5. DashScope 密钥正确
6. MiniMax 密钥正确

---

## 4️⃣ 统一依赖管理

仓库统一使用根目录一个 `pyproject.toml`。

安装基础依赖：

```bash
uv sync
```

---

## 5️⃣ 目录结构

```text
my-agent/
├─ src/                   # Agent 主逻辑
├─ docs/                  # 部署文档
├─ pyproject.toml         # 依赖配置文件
└─ README.md
```

---

## 6️⃣ 相关文档

- [`docs/LiveKit-Server部署.md`](./docs/LiveKit-Server部署.md)
- [`docs/LLM部署.md`](./docs/LLM部署.md)
- [`docs/系统架构与数据流.md`](./docs/系统架构与数据流.md)
