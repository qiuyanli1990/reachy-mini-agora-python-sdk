"""Agora Conversational AI handler compatible with LocalStream/Gradio wiring.

This handler keeps the same high-level async interface expected by LocalStream:
- ``start_up``: bootstrap remote session
- ``receive``: ingest local microphone frames
- ``emit``: provide remote audio / text outputs
- ``shutdown``: graceful teardown

It reuses the existing Agora bridge + agent manager modules and dispatches
tool-like device actions from ConvoAI datastream ``message.user`` payloads.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_agora_python_sdk.agent_manager import AgentManager
from reachy_mini_agora_python_sdk.agora_bridge import AgoraBridge
from reachy_mini_agora_python_sdk.token_builder import TokenGenerator
from reachy_mini_agora_python_sdk.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
)
from reachy_mini_agora_python_sdk.video_converter import VideoConverter


logger = logging.getLogger(__name__)


DEFAULT_AGORA_APP_ID = ""
DEFAULT_AGORA_APP_CERTIFICATE = ""
DEFAULT_AGORA_API_KEY = ""
DEFAULT_AGORA_API_SECRET = ""
DEFAULT_AGORA_CHANNEL_NAME = "reachy_conversation"
DEFAULT_REACHY_USER_ID = 12345
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 15


class AgoraConversationalAIHandler(AsyncStreamHandler):
    """Async stream handler for Agora Conversational AI."""

    requires_agora_config = True

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        self._audio_sample_rate = self._env_int("AGORA_AUDIO_SAMPLE_RATE", DEFAULT_AUDIO_SAMPLE_RATE)
        self._audio_channels = self._env_int("AGORA_AUDIO_CHANNELS", DEFAULT_AUDIO_CHANNELS)
        self._video_width = self._env_int("AGORA_VIDEO_WIDTH", DEFAULT_VIDEO_WIDTH)
        self._video_height = self._env_int("AGORA_VIDEO_HEIGHT", DEFAULT_VIDEO_HEIGHT)
        self._video_fps = self._env_int("AGORA_VIDEO_FPS", DEFAULT_VIDEO_FPS)
        self._channel_name = self._env("AGORA_CHANNEL_NAME", DEFAULT_AGORA_CHANNEL_NAME)
        super().__init__(
            expected_layout="mono",
            output_sample_rate=self._audio_sample_rate,
            input_sample_rate=self._audio_sample_rate,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: "asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = (
            asyncio.Queue()
        )
        self._shutdown_requested = False
        self._connected_event: asyncio.Event = asyncio.Event()

        self._agora: Optional[AgoraBridge] = None
        self._agent_manager: Optional[AgentManager] = None
        self._token_generator: Optional[TokenGenerator] = None
        self._agent_started = False

        self._datastream_task: Optional[asyncio.Task[Any]] = None
        self._video_task: Optional[asyncio.Task[Any]] = None

        self._last_video_sent_ts = 0.0
        self._video_period_s = 1.0 / max(1, self._video_fps)
        self._last_uplink_diag_ts = 0.0
        self._seen_datastream_ids: set[str] = set()
        self._tool_trace_stats = {
            "message_user_received": 0,
            "action_payload_parsed": 0,
            "tool_dispatch_attempted": 0,
            "tool_dispatch_succeeded": 0,
            "tool_dispatch_failed": 0,
        }
        self._datastream_obj_stats: dict[str, int] = {}
        self._datastream_last_stats_log_ts = 0.0
        self._conversation_state = "idle"
        self._assistant_speaking = False

    def copy(self) -> "AgoraConversationalAIHandler":
        return AgoraConversationalAIHandler(
            self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
        )

    async def apply_personality(self, profile: str | None) -> str:
        # Personality logic is currently managed by ConvoAI agent configuration.
        return f"Agora backend does not hot-switch personality yet (requested: {profile or 'default'})."

    async def get_available_voices(self) -> list[str]:
        # Voice is controlled by ConvoAI agent config (tts section).
        return ["agent-managed-voice"]

    async def start_up(self) -> None:
        self._shutdown_requested = False

        required = self._required_agora_settings()
        missing = [k for k, v in required.items() if not str(v).strip()]
        if missing:
            msg = f"Missing required Agora settings: {', '.join(missing)}"
            logger.error(msg)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))
            return

        agent_cfg = self._resolve_agent_config_path()
        if not agent_cfg.exists():
            msg = f"agent_config.json not found at: {agent_cfg}"
            logger.error(msg)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))
            return

        self._agora = AgoraBridge(
            app_id=required["AGORA_APP_ID"],
            channel_name=self._channel_name,
            user_id=self._resolve_reachy_user_id(),
            audio_sample_rate=self._audio_sample_rate,
            audio_channels=self._audio_channels,
            video_width=self._video_width,
            video_height=self._video_height,
            video_fps=self._video_fps,
        )

        if not self._agora.initialize():
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[error] Failed to initialize Agora service"})
            )
            return

        reachy_uid = self._resolve_reachy_user_id()
        reachy_token = ""
        agent_token = ""
        self._token_generator = TokenGenerator(
            app_id=required["AGORA_APP_ID"],
            app_certificate=self._env("AGORA_APP_CERTIFICATE", DEFAULT_AGORA_APP_CERTIFICATE),
        )
        if self._token_generator.is_certificate_enabled():
            agent_uid_for_token = self._resolve_agent_uid(agent_cfg)
            logger.info("Generating local RTC token for reachy_uid=%s", reachy_uid)
            reachy_token = self._token_generator.generate_token_for_user(
                channel_name=self._channel_name,
                uid=reachy_uid,
                expire_time=3600,
            )
            logger.info("Generating agent RTC token for agent_rtc_uid=%s", agent_uid_for_token)
            agent_token = self._token_generator.generate_token_for_agent(
                channel_name=self._channel_name,
                agent_uid=agent_uid_for_token,
                expire_time=3600,
            )

        if not self._agora.join_channel(reachy_token):
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[error] Failed to join Agora channel"})
            )
            return

        self._agent_manager = AgentManager(
            app_id=required["AGORA_APP_ID"],
            api_key=required["AGORA_API_KEY"],
            api_secret=required["AGORA_API_SECRET"],
            config_file=str(agent_cfg),
        )
        self._agent_started = self._agent_manager.start_agent_from_config(
            channel_name=self._channel_name,
            user_uid=reachy_uid,
            token=agent_token,
        )
        if not self._agent_started:
            logger.warning("Agent failed to start; audio/video bridge remains active.")

        self._connected_event.set()
        self._datastream_task = asyncio.create_task(self._datastream_loop(), name="agora-datastream-loop")
        self._video_task = asyncio.create_task(self._video_loop(), name="agora-video-loop")

        logger.info("Agora conversational handler started")
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    async def _datastream_loop(self) -> None:
        assert self._agora is not None
        while not self._shutdown_requested:
            message = self._agora.get_received_stream_message()
            if not message:
                await asyncio.sleep(0.02)
                continue

            payload_json = message.get("json")
            if not isinstance(payload_json, dict):
                continue

            if self._is_duplicate_datastream(payload_json):
                continue

            obj = payload_json.get("object")
            obj_s = str(obj or "unknown")
            self._log_datastream_stats(obj_s)

            if obj == "message.state":
                state = str(payload_json.get("state", "")).strip().lower()
                self._handle_message_state(state)
                continue

            if obj == "assistant.transcription":
                self._handle_assistant_transcription(payload_json)
                continue

            content = payload_json.get("content")
            action_payload = self._coerce_action_payload(content)
            await self._handle_action_message(obj_s, payload_json, action_payload)

    def _is_duplicate_datastream(self, payload_json: dict[str, Any]) -> bool:
        """Return True if message_id has been seen and should be skipped."""
        msg_id = str(payload_json.get("message_id", "")).strip()
        if not msg_id:
            return False
        if msg_id in self._seen_datastream_ids:
            return True
        self._seen_datastream_ids.add(msg_id)
        if len(self._seen_datastream_ids) > 1000:
            self._seen_datastream_ids.clear()
        return False

    def _log_datastream_stats(self, obj_s: str) -> None:
        """Aggregate and periodically log datastream object/tool counters."""
        self._datastream_obj_stats[obj_s] = self._datastream_obj_stats.get(obj_s, 0) + 1
        now = time.time()
        if now - self._datastream_last_stats_log_ts < 5.0:
            return
        self._datastream_last_stats_log_ts = now
        logger.info(
            "DATASTREAM_TRACE objects=%s tool_trace=%s",
            json.dumps(self._datastream_obj_stats, ensure_ascii=False),
            json.dumps(self._tool_trace_stats, ensure_ascii=False),
        )

    def _handle_assistant_transcription(self, payload_json: dict[str, Any]) -> None:
        """Keep assistant transcription logging concise."""
        text = str(payload_json.get("text", "")).strip()
        source = str(payload_json.get("source", "")).strip()
        turn_id = payload_json.get("turn_id")
        if not text:
            return
        low = text.lower()
        if "couldn't process" in low or "could not process" in low:
            logger.warning(
                "assistant.transcription indicates ASR/NLU failure (turn_id=%s): %s",
                turn_id,
                text,
            )
        logger.debug(
            "assistant.transcription(source=%s, turn_id=%s): %s",
            source,
            turn_id,
            text,
        )

    async def _handle_action_message(
        self,
        obj_s: str,
        payload_json: dict[str, Any],
        action_payload: Optional[dict[str, Any]],
    ) -> None:
        """Trace and dispatch datastream action payloads."""
        if action_payload is None:
            return
        turn_id = payload_json.get("turn_id")
        self._tool_trace_stats["message_user_received"] += 1
        logger.info(
            "TOOL_TRACE action candidate received: object=%s turn_id=%s payload=%s",
            obj_s,
            turn_id,
            json.dumps(action_payload, ensure_ascii=False),
        )
        await self._dispatch_action_payload(action_payload)

    def _handle_message_state(self, state: str) -> None:
        """Update local motion gates from ConvoAI state heartbeat."""
        if not state:
            return

        previous_state = self._conversation_state
        self._conversation_state = state

        # Mirror official behavior: listening mode should suppress idle breathing
        # and freeze antennas while user is speaking.
        try:
            self.deps.movement_manager.set_listening(state == "listening")
        except Exception as e:
            logger.debug("set_listening failed for state=%s: %s", state, e)

        is_speaking = state == "speaking"
        if is_speaking != self._assistant_speaking:
            self._assistant_speaking = is_speaking
            if not is_speaking and self.deps.head_wobbler is not None:
                # Stop residual wobble immediately when TTS ends.
                self.deps.head_wobbler.reset()
            logger.info(
                "MOTION_GATE state transition: %s -> %s (assistant_speaking=%s)",
                previous_state,
                state,
                self._assistant_speaking,
            )

    def _coerce_action_payload(self, content: Any) -> Optional[dict[str, Any]]:
        """Parse message.user content into action payload.

        Supports:
        - direct JSON dict: {"action_type": ...}
        - JSON string of dict
        - wrapped form: {"content": "{\"action_type\": ...}"}
        """
        payload: Any = content
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                logger.debug("message.user content is not JSON: %s", payload)
                return None

        if not isinstance(payload, dict):
            return None

        # Unwrap publish_message envelope if present.
        wrapped_content = payload.get("content")
        if "action_type" not in payload and wrapped_content is not None:
            if isinstance(wrapped_content, str):
                try:
                    unwrapped = json.loads(wrapped_content)
                    if isinstance(unwrapped, dict):
                        payload = unwrapped
                except json.JSONDecodeError:
                    return None
            elif isinstance(wrapped_content, dict):
                payload = wrapped_content

        if not isinstance(payload, dict):
            return None
        if not str(payload.get("action_type", "")).strip():
            return None
        return payload

    async def _dispatch_action_payload(self, payload: dict[str, Any]) -> bool:
        self._tool_trace_stats["action_payload_parsed"] += 1
        action_type = str(payload.get("action_type", "")).strip()
        logger.info(
            "TOOL_TRACE parsed action payload: action_type=%s payload=%s",
            action_type,
            json.dumps(payload, ensure_ascii=False),
        )

        tool_name, tool_args = self._map_action_to_tool(action_type, payload)

        if tool_name is None:
            logger.info("Unsupported action_type from datastream: %s", action_type)
            self._tool_trace_stats["tool_dispatch_failed"] += 1
            return False

        self._tool_trace_stats["tool_dispatch_attempted"] += 1
        logger.info(
            "Dispatching local tool from datastream: tool=%s args=%s",
            tool_name,
            json.dumps(tool_args, ensure_ascii=False),
        )
        result = await dispatch_tool_call(tool_name, json.dumps(tool_args), self.deps)
        if isinstance(result, dict) and result.get("error"):
            self._tool_trace_stats["tool_dispatch_failed"] += 1
            logger.warning(
                "TOOL_TRACE tool execution failed: tool=%s error=%s stats=%s",
                tool_name,
                result.get("error"),
                json.dumps(self._tool_trace_stats, ensure_ascii=False),
            )
        else:
            self._tool_trace_stats["tool_dispatch_succeeded"] += 1
            logger.info(
                "TOOL_TRACE tool execution succeeded: tool=%s result=%s stats=%s",
                tool_name,
                json.dumps(result, ensure_ascii=False),
                json.dumps(self._tool_trace_stats, ensure_ascii=False),
            )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(result, ensure_ascii=False),
                    "metadata": {"title": f"Agora action -> tool: {tool_name}", "status": "done"},
                }
            )
        )
        return True

    def _map_action_to_tool(self, action_type: str, payload: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
        """Map datastream action payload to local tool call."""
        if action_type in {"display_emotion", "play_emotion", "emotion"}:
            emotion = payload.get("emotion_type") or payload.get("emotion")
            if emotion:
                return "play_emotion", {"emotion": str(emotion)}
            return None, {}

        if action_type == "move_head":
            direction = payload.get("direction")
            if direction:
                return "move_head", {"direction": str(direction)}
            return None, {}

        if action_type == "dance":
            return "dance", {
                "move": payload.get("move", "random"),
                "repeat": int(payload.get("repeat", 1)),
            }

        if action_type == "stop_dance":
            return "stop_dance", {}

        if action_type == "stop_emotion":
            return "stop_emotion", {"dummy": True}

        if action_type == "head_tracking":
            return "head_tracking", {"enabled": bool(payload.get("enabled", True))}

        return None, {}

    async def _video_loop(self) -> None:
        assert self._agora is not None
        while not self._shutdown_requested:
            now = time.monotonic()
            if now - self._last_video_sent_ts < self._video_period_s:
                await asyncio.sleep(0.002)
                continue

            frame_bgr = None
            if self.deps.camera_worker is not None:
                try:
                    frame_bgr = self.deps.camera_worker.get_latest_frame()
                except Exception:
                    frame_bgr = None

            if frame_bgr is None:
                await asyncio.sleep(0.01)
                continue

            try:
                yuv = VideoConverter.bgr_to_yuv_i420(frame_bgr)
                self._agora.send_video_frame(yuv, frame_bgr.shape[1], frame_bgr.shape[0])
                self._last_video_sent_ts = now
            except Exception as e:
                logger.debug("Video send skipped: %s", e)
                await asyncio.sleep(0.02)

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        if not self._agora:
            return

        input_sample_rate, audio_frame = frame
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sample_rate),
            )

        # Send raw captured PCM to uplink without software gain to avoid clipping.
        pcm_np = audio_to_int16(audio_frame).astype(np.int16, copy=False)

        now = time.monotonic()
        if now - self._last_uplink_diag_ts >= 2.0:
            rms = float(
                np.sqrt(np.mean((pcm_np.astype(np.float32) / 32767.0) ** 2))
            ) if pcm_np.size else 0.0
            peak = float(np.max(np.abs(pcm_np.astype(np.float32))) / 32767.0) if pcm_np.size else 0.0
            logger.info(
                "Uplink mic level: rms=%.4f peak=%.4f gain=1.00(raw) sample_rate=%s",
                rms,
                peak,
                input_sample_rate,
            )
            self._last_uplink_diag_ts = now

        pcm = pcm_np.tobytes()
        self._agora.send_audio_frame(pcm)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        if self._agora is None:
            await asyncio.sleep(0.02)
            return None

        # Audio must be prioritized, otherwise frequent state messages can
        # block audible playback.
        pcm = self._agora.get_received_audio_blocking(timeout_s=0.02)
        if pcm:
            audio_np = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
            if (
                self._assistant_speaking
                and self.deps.head_wobbler is not None
                and audio_np.size > 0
            ):
                # Reuse existing lip-sync wobble path by feeding a small synthetic delta.
                rms = float(np.sqrt(np.mean((audio_np.astype(np.float32) / 32767.0) ** 2)))
                if rms > 0.002:
                    fake_pcm = (np.clip(audio_np, -32768, 32767).astype(np.int16)).tobytes()
                    self.deps.head_wobbler.feed(self._pcm_to_b64(fake_pcm))
            return (self.output_sample_rate, audio_np)

        # If no audio frame available, emit pending textual/tool messages.
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        return None

    def _pcm_to_b64(self, pcm: bytes) -> str:
        import base64

        return base64.b64encode(pcm).decode("utf-8")

    def _resolve_agent_config_path(self) -> Path:
        """Resolve the single supported config path: app root agent_config.json."""
        return Path(__file__).resolve().parents[2] / "agent_config.json"

    def _resolve_agent_uid(self, agent_cfg: Path) -> int:
        """Resolve agent_rtc_uid for token generation.

        Priority:
        1) Environment AGORA_AGENT_UID
        2) agent_config.json properties.agent_rtc_uid
        3) Fallback 1000
        """
        env_uid = self._env("AGORA_AGENT_UID", "")
        if env_uid:
            try:
                return int(env_uid)
            except ValueError:
                logger.warning("Invalid AGORA_AGENT_UID=%s, fallback to config", env_uid)

        try:
            data = json.loads(agent_cfg.read_text(encoding="utf-8"))
            props = data.get("properties", {}) if isinstance(data, dict) else {}
            cfg_uid = str(props.get("agent_rtc_uid", "")).strip()
            if cfg_uid:
                return int(cfg_uid)
        except Exception as e:
            logger.warning("Failed to read agent_rtc_uid from %s: %s", agent_cfg, e)

        return 1000

    def _env(self, key: str, fallback: str) -> str:
        """Read runtime env value with fallback."""
        return str((os.getenv(key) or fallback or "")).strip()

    def _required_agora_settings(self) -> dict[str, str]:
        """Read required Agora settings from environment."""
        return {
            "AGORA_APP_ID": self._env("AGORA_APP_ID", DEFAULT_AGORA_APP_ID),
            "AGORA_API_KEY": self._env("AGORA_API_KEY", DEFAULT_AGORA_API_KEY),
            "AGORA_API_SECRET": self._env("AGORA_API_SECRET", DEFAULT_AGORA_API_SECRET),
        }

    def _resolve_reachy_user_id(self) -> int:
        """Read Reachy user id from unified env variable."""
        value = self._env("AGORA_Reachy_mini_USER_ID", "")
        if value:
            try:
                return int(value)
            except Exception:
                logger.warning("Invalid AGORA_Reachy_mini_USER_ID=%r, fallback to default", value)

        return DEFAULT_REACHY_USER_ID

    def _env_int(self, key: str, fallback: int) -> int:
        raw = self._env(key, str(fallback))
        try:
            return int(raw)
        except Exception:
            logger.warning("Invalid %s=%r, fallback to %s", key, raw, fallback)
            return fallback

    async def shutdown(self) -> None:
        self._shutdown_requested = True

        if self._datastream_task is not None:
            self._datastream_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._datastream_task

        if self._video_task is not None:
            self._video_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._video_task

        if self._agent_manager and self._agent_started:
            try:
                self._agent_manager.stop_agent()
            except Exception:
                logger.exception("Failed to stop agent")

        if self._agora:
            try:
                self._agora.leave_channel()
            except Exception:
                logger.exception("Failed to leave Agora channel")
            self._agora = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
