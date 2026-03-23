# 🚀 Qwen2.5-7B-Instruct + vLLM + LiveKit 部署指南

> **💡 项目概览**: 在 AutoDL 云端 GPU 实例上，利用 `uv` 极速构建环境，部署 **Qwen2.5-7B** 大模型，并通过 **vLLM** 提供高性能推理服务，最终接入 **LiveKit Agent** 实现智能语音交互。

---

## 1️⃣ 🖥️ 测试服务器配置

| 配置项 | 详细信息 |
| :--- | :--- |
| **☁️ 实例平台** | AutoDL 容器实例 |
| **🎮 显卡规格** | **RTX PRO 6000** × 1 卡 (24GB VRAM) |
| **📍 地区节点** | 重庆 A 区 |
| **📦 环境管理** | **UV** (`uv add` / `uv run`) - *极速包管理器* |
| **🤖 模型来源** | ModelScope (`Qwen/Qwen2.5-7B-Instruct`) |
| **⚡ 推理引擎** | **vLLM** (OpenAI 兼容接口) |

---

## 2️⃣ 🛠️ 环境初始化与依赖安装

使用 `uv` 初始化项目并秒级安装依赖：

```bash
# 1. 🌱 初始化 uv 项目 (如未初始化)
uv init
uv sync

# 2. 📦 添加核心依赖 (vLLM, modelscope)
uv add vllm modelscope

# 3. 🔌 激活虚拟环境
source .venv/bin/activate
```
> ✨ **提示**: `uv add` 会自动创建 `.venv` 虚拟环境并锁定依赖版本，比 pip 快 10-100 倍！

---

## 3️⃣ 📥 模型下载 (ModelScope)

将模型权重下载至本地 `./qwen` 目录：

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./qwen
```
✅ **检查清单**: 下载完成后，请确认 `./qwen` 目录下包含 `config.json` 和 `model.safetensors` 等关键文件。

---

## 4️⃣ 🔥 启动推理服务

使用以下命令启动 vLLM 服务，开启高性能推理：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1
```

**🎉 启动成功标志**:
终端日志最后显示：
`INFO: Uvicorn running on http://0.0.0.0:8000`

---

## 5️⃣ 🌉 本地连接 (SSH 端口转发)

在**本地电脑**的终端执行以下命令，建立安全隧道，将云端端口映射到本地：

```bash
# 🔄 请将 <实例端口> 和 <实例地址> 替换为你的 AutoDL 实际连接信息
ssh -p <实例端口> -L 8000:localhost:8000 root@<实例地址>
```
> ⚠️ **重要**: 此终端窗口必须**保持开启**状态！关闭窗口会导致本地无法访问模型服务。

---

## 6️⃣ 🤖 LiveKit 代码配置

在你的 LiveKit Agent 项目中，配置 LLM 插件指向本地映射的 vLLM 服务：

```python
from livekit.plugins import openai

# 🧠 初始化 LLM 实例
llm_instance = openai.LLM(
    model="./qwen",                # 📂 对应启动命令中的 --model 路径
    base_url="http://127.0.0.1:8000/v1", # 🌐 本地转发地址 (必须带 /v1 后缀)
    api_key="fake-key"             # 🔑 vLLM 无需真实密钥，填任意字符串即可
)
```

### 🔑 关键参数说明

| 参数 | 说明 |
| :--- | :--- |
| **`model`** | 必须与启动命令中的 `--model` 参数完全一致（此处为 `./qwen`）。*(若报错可尝试改为模型名称 `Qwen2.5-7B-Instruct`)* |
| **`base_url`** | 固定为 `http://127.0.0.1:8000/v1`，指向本地映射端口。 |
| **`api_key`** | 占位符即可，vLLM 默认不校验密钥真实性。 |

---

## 🎯 下一步

现在你的本地开发环境已经通过 SSH 隧道连接到了云端的 Qwen2.5 大模型！
你可以开始编写 LiveKit Agent 逻辑，让 AI 拥有听觉和语音能力了！ 🎙️🔊✨