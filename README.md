# reachy-mini-agora-python-sdk

## 1. Project Overview
This project is a **Reachy Mini + Agora Conversational AI** conversation app that keeps the Python SDK mode:

- Audio/video transmission is handled by the Agora Python SDK in the local app process.
- Reachy Mini microphone, camera, and speaker are used for channel I/O.
- ConvoAI joins the same channel and sends datastream messages for local actions.

## 2. Solution Architecture
![Solution Architecture](presentation/ppt_scheme_from_sketch.svg)

## 3. Directory Structure (Core)
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

## 4. Configuration

### 4.1 Required environment variables in `.env`
- `AGORA_APP_ID`
- `AGORA_APP_CERTIFICATE`
- `AGORA_API_KEY`
- `AGORA_API_SECRET`
- `AGORA_CHANNEL_NAME`
- `AGORA_Reachy_mini_USER_ID`
- `PLAYBACK_VOLUME`

Valid `.env` path:
- `.env` in the project root
- You can start from `.env.example`.

### 4.2 `agent_config.json` setup
- File path: `agent_config.json` in the project root
- Content requirement: provide a Start Body JSON that matches Agora ConvoAI `/join` schema.
- Recommended checks:
  - `properties.agent_rtc_uid` must not conflict with the local user UID.
  - `properties.remote_rtc_uids` should include `AGORA_Reachy_mini_USER_ID`.
  - If using an external prompt file, you can set `"{{prompt.txt}}"` as a placeholder.
- References:
  - https://docs.agora.io/en/conversational-ai/rest-api/agent/join
  - https://docs.agora.io/en/server-gateway/get-started/integrate-sdk?platform=python

## 5. Run Steps (Python SDK mode)

1. Prepare the Python environment first
Follow the official Reachy Mini installation guide before running this project. A minimal setup on macOS/Linux is:

On macOS, make sure the Python build matches your machine architecture. Check it with:
```bash
python3 -c "import platform; print(platform.machine())"
```

Possible outputs:
- `arm64` for Apple Silicon
- `x86_64` for Intel or Rosetta

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 --default
```

Make sure Git and Git LFS are installed on your OS first, then run:

```bash
git lfs install
uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate
uv pip install "reachy-mini"
```

2. Clone the project into your apps directory (for example `<apps-dir>`)
```bash
cd <apps-dir>
git clone <repo-url> reachy-mini-agora-python-sdk
```

3. Start daemon (Terminal A)
```bash
source /path/to/venv/bin/activate
reachy-mini-daemon
```

4. Start app (Terminal B)
```bash
source /path/to/venv/bin/activate
cd /path/to/reachy-mini-agora-python-sdk
pip install -e .
python -m reachy_mini_agora_python_sdk.main
```

5. The app joins the channel and starts the configured ConvoAI agent flow

## 6. Runtime Behavior in Python SDK Mode
- The app captures Reachy microphone audio and camera frames locally.
- The Agora Python SDK publishes local audio/video to the channel.
- Remote TTS audio from ConvoAI is played on the Reachy speaker.
- Datastream messages are parsed locally and dispatched to motion tools.
- `message.state` and speech audio drive speaking wobble and idle breathing behavior.

## 7. Stop and Exit
- Press `Ctrl+C` in the app terminal:
  - Stops the local RTC session.
  - Stops the agent started by this service if it is running.
  - Releases local media and motion resources.

## 8. Official References
- Reachy Mini SDK installation:
  - https://huggingface.co/docs/reachy_mini/SDK/installation
- Agora ConvoAI REST `/join`:
  - https://docs.agora.io/en/conversational-ai/rest-api/agent/join
- Agora Python SDK integration:
  - https://docs.agora.io/en/server-gateway/get-started/integrate-sdk?platform=python
- Agora account and authentication:
  - https://docs.agora.io/en/conversational-ai/get-started/manage-agora-account
  - https://docs.agora.io/en/conversational-ai/rest-api/restful-authentication
