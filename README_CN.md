# reachy-mini-agora-python-sdk 项目档案

## 1. 项目概述
本项目是一个 **Reachy Mini + Agora Conversational AI** 对话应用，保留 Python SDK 模式：

- 音视频收发由本地应用进程中的 Agora Python SDK 完成。
- Reachy Mini 的麦克风、摄像头、扬声器用于频道音视频 I/O。
- ConvoAI 加入同一频道，并通过 datastream 下发本地动作指令。

## 2. 方案架构图
![Solution Architecture](presentation/ppt_scheme_from_sketch.svg)

## 3. 目录结构（核心）
```text
reachy-mini-agora-python-sdk/
├── .env
├── agent_config.json
├── prompt.txt
├── README_CN.md / README.md
├── src/reachy_mini_agora_python_sdk/
│   ├── main.py
│   ├── agora_conversational_ai.py
│   ├── agora_bridge.py
│   ├── agent_manager.py
│   ├── moves.py
│   ├── audio/
│   ├── tools/
│   └── vision/
└── docs/
```

## 4. 配置说明

### 4.1 .env必填环境变量
- `AGORA_APP_ID`
- `AGORA_APP_CERTIFICATE`
- `AGORA_API_KEY`
- `AGORA_API_SECRET`
- `AGORA_CHANNEL_NAME`
- `AGORA_Reachy_mini_USER_ID`
- `PLAYBACK_VOLUME`

`.env` 有效路径：
- 项目根目录下的 `.env`
- 可先复制 `.env.example` 作为起始模板。

### 4.2 `agent_config.json` 填写说明
- 文件路径：项目根目录下的 `agent_config.json`
- 内容要求：填写符合 Agora ConvoAI `/join` 规范的 Start Body JSON。
- 建议检查项：
  - `properties.agent_rtc_uid` 不能与本地用户 UID 冲突。
  - `properties.remote_rtc_uids` 应包含 `AGORA_Reachy_mini_USER_ID`。
  - 如使用外部 prompt 文件，占位可写 `"{{prompt.txt}}"`。
- 参考文档：
  - https://docs.agora.io/en/conversational-ai/rest-api/agent/join
  - https://docs.agora.io/en/server-gateway/get-started/integrate-sdk?platform=python

## 5. 运行步骤（Python SDK 模式）

1. 先准备 Python 运行环境
运行本项目之前，请先按 Reachy Mini 官方安装文档完成环境准备。macOS / Linux 最小步骤如下：

注意：在 macOS 上安装 Python 时，要确认 Python 架构与机器芯片一致。可先执行：
```bash
python3 -c "import platform; print(platform.machine())"
```

输出可能是：
- `arm64`：Apple Silicon
- `x86_64`：Intel 或 Rosetta

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 --default
```

请先确保系统里已经安装 Git 和 Git LFS，然后继续执行：

```bash
git lfs install
uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate
uv pip install "reachy-mini"
```

2. 将项目代码 clone 到你的 apps 目录（例如 `<apps-dir>`）
```bash
cd <apps-dir>
git clone <repo-url> reachy-mini-agora-python-sdk
```

3. 启动 daemon（终端 A）
```bash
source /path/to/venv/bin/activate
reachy-mini-daemon
```

4. 启动应用（终端 B）
```bash
source /path/to/venv/bin/activate
cd /path/to/reachy-mini-agora-python-sdk
pip install -e .
python -m reachy_mini_agora_python_sdk.main
```

5. 应用会加入频道并启动配置好的 ConvoAI agent 链路

## 6. Python SDK 模式运行时行为
- 应用在本地采集 Reachy 麦克风音频和摄像头画面。
- Agora Python SDK 将本地音视频发布到频道。
- ConvoAI 返回的远端 TTS 音频会播放到 Reachy 扬声器。
- datastream 在本地解析后分发到动作工具链路。
- `message.state` 与语音音频共同驱动说话律动和空闲呼吸动作。

## 7. 停止与退出
- 在应用终端按 `Ctrl+C`：
  - 停止本地 RTC 会话。
  - 停止本服务启动的 agent（若在运行）。
  - 释放本地媒体与动作资源。

## 8. 官方文档参考
- Agora ConvoAI REST `/join`：
  - https://docs.agora.io/en/conversational-ai/rest-api/agent/join
- Agora Python SDK 集成：
  - https://docs.agora.io/en/server-gateway/get-started/integrate-sdk?platform=python
- Agora 账号与鉴权：
  - https://docs.agora.io/en/conversational-ai/get-started/manage-agora-account
  - https://docs.agora.io/en/conversational-ai/rest-api/restful-authentication
- Reachy Mini SDK 安装：
  - https://huggingface.co/docs/reachy_mini/SDK/installation
