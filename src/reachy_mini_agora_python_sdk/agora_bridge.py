"""Agora RTC Bridge for Reachy Mini.

This module provides a bridge between Reachy Mini and Agora RTC,
handling audio/video streaming and communication.
"""

import logging
import threading
import queue
import time
import json
import base64
import binascii
from typing import Optional, Callable, Any

import numpy as np

try:
    from agora.rtc.agora_service import AgoraService
    from agora.rtc.agora_base import (
        AgoraServiceConfig,
        RTCConnConfig,
        RtcConnectionPublishConfig,
        ClientRoleType,
        ChannelProfileType,
        AudioScenarioType,
        AudioProfileType,
        AudioPublishType,
        VideoPublishType,
        AudioParams,
        VideoDimensions,
        VideoEncoderConfiguration,
        VideoCodecType,
        ExternalVideoFrame,
    )
    from agora.rtc.audio_frame_observer import IAudioFrameObserver
    from agora.rtc.rtc_connection import RTCConnection
    from agora.rtc.local_user_observer import IRTCLocalUserObserver
    from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
except ImportError as e:
    raise ImportError(
        "Agora Python SDK not found. Please install it with: "
        "pip install agora-python-server-sdk"
    ) from e


class _AgoraLocalUserObserver(IRTCLocalUserObserver):
    """Local user observer to receive Agora datastream messages."""

    def __init__(self, bridge: "AgoraBridge"):
        self._bridge = bridge

    def on_stream_message(self, agora_local_user, user_id, stream_id, data, length):
        self._bridge._handle_stream_message(user_id, stream_id, data, length)

    def on_first_remote_audio_frame(self, agora_local_user, user_id, elapsed):
        self._bridge.logger.info(
            "on_first_remote_audio_frame: user=%s elapsed=%s", user_id, elapsed
        )

    def on_first_remote_audio_decoded(self, agora_local_user, user_id, elapsed):
        self._bridge.logger.info(
            "on_first_remote_audio_decoded: user=%s elapsed=%s", user_id, elapsed
        )

    def on_audio_subscribe_state_changed(
        self,
        agora_local_user,
        channel,
        user_id,
        old_state,
        new_state,
        elapse_since_last_state,
    ):
        self._bridge.logger.info(
            "on_audio_subscribe_state_changed: channel=%s user=%s %s->%s elapsed=%s",
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        )


class _AgoraAudioFrameObserver(IAudioFrameObserver):
    """Audio frame observer to receive remote playback audio."""

    def __init__(self, bridge: "AgoraBridge"):
        self._bridge = bridge
        self._first_playback_logged = False

    def on_playback_audio_frame(self, agora_local_user, channelId, frame):
        if not self._first_playback_logged:
            self._bridge.logger.info(
                "First playback audio frame received (channel=%s)", channelId
            )
            self._first_playback_logged = True
        self._bridge._handle_received_audio_frame(frame)
        return 1

    def on_playback_audio_frame_before_mixing(
        self, agora_local_user, channelId, userId, frame
    ):
        # Compatibility fallback: some SDK builds only trigger before-mixing callback.
        if not self._first_playback_logged:
            self._bridge.logger.info(
                "First playback(before_mixing) audio frame received "
                "(channel=%s, user=%s)",
                channelId,
                userId,
            )
            self._first_playback_logged = True
        self._bridge._handle_received_audio_frame(frame)
        return 1

    def on_get_audio_frame_position(self, agora_local_user):
        # Only request mixed playback frames.
        return 1

    def on_get_playback_audio_frame_param(self, agora_local_user):
        # Some SDK versions require this callback to activate playback frame delivery.
        return AudioParams(
            sample_rate=self._bridge.audio_sample_rate,
            channels=self._bridge.audio_channels,
            mode=0,  # RAW_AUDIO_FRAME_OP_MODE_READ_ONLY
            samples_per_call=960,  # 60ms @ 16kHz
        )


class _AgoraConnectionObserver(IRTCConnectionObserver):
    """Connection observer for user join/leave callbacks and diagnostics."""

    def __init__(self, bridge: "AgoraBridge"):
        self._bridge = bridge

    def on_user_joined(self, agora_rtc_conn, user_id):
        self._bridge.logger.info("on_user_joined: user=%s", user_id)
        try:
            # Be explicit: subscribe in case auto_subscribe does not take effect.
            ret_sub = self._bridge.connection.local_user.subscribe_audio(str(user_id))
            if ret_sub != 0:
                self._bridge.logger.warning(
                    "subscribe_audio(%s) returned: %s", user_id, ret_sub
                )
            else:
                self._bridge.logger.info("Subscribed remote audio for user %s", user_id)
        except Exception:
            self._bridge.logger.exception("subscribe_audio on user_joined failed")

        if self._bridge.on_user_joined:
            try:
                self._bridge.on_user_joined(int(user_id))
            except Exception:
                self._bridge.logger.exception("on_user_joined callback failed")

    def on_user_left(self, agora_rtc_conn, user_id, reason):
        if self._bridge.on_user_left:
            try:
                self._bridge.on_user_left(int(user_id))
            except Exception:
                self._bridge.logger.exception("on_user_left callback failed")

    def on_stream_message_error(self, agora_rtc_conn, user_id_str, stream_id, code, missed, cached):
        self._bridge.logger.warning(
            "Datastream error: user=%s stream_id=%s code=%s missed=%s cached=%s",
            user_id_str,
            stream_id,
            code,
            missed,
            cached,
        )


class AgoraBridge:
    """Bridge between Reachy Mini and Agora RTC.

    This class manages the connection to Agora RTC channel and handles
    bidirectional audio/video streaming.

    Attributes:
        app_id: Agora application ID
        channel_name: Name of the channel to join
        user_id: User ID for this client
        audio_sample_rate: Audio sample rate (default: 16000 Hz)
        audio_channels: Number of audio channels (default: 1 for mono)
    """

    def __init__(
        self,
        app_id: str,
        channel_name: str,
        user_id: int,
        audio_sample_rate: int = 16000,
        audio_channels: int = 1,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: int = 15,
        video_bitrate: int = 1500,
    ):
        """Initialize the Agora bridge.

        Args:
            app_id: Agora application ID
            channel_name: Channel name to join
            user_id: User ID for this connection
            audio_sample_rate: Audio sample rate in Hz
            audio_channels: Number of audio channels (1 or 2)
        """
        self.app_id = app_id
        self.channel_name = channel_name
        self.user_id = user_id
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.video_fps = int(video_fps)
        self.video_bitrate = int(video_bitrate)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Agora components
        self.agora_service: Optional[AgoraService] = None
        self.connection: Optional[RTCConnection] = None

        # Queues for audio/video data
        self.audio_send_queue = queue.Queue(maxsize=100)
        self.audio_recv_queue = queue.Queue(maxsize=100)
        self.video_send_queue = queue.Queue(maxsize=30)
        self.stream_recv_queue = queue.Queue(maxsize=200)

        # Thread control
        self.running = False
        self.threads = []
        self._leave_lock = threading.Lock()
        self._left_channel = False
        self._video_enabled = False
        self._encoder_width: Optional[int] = None
        self._encoder_height: Optional[int] = None
        self._stream_messages_received = 0
        self._local_user_observer: Optional[_AgoraLocalUserObserver] = None
        self._connection_observer: Optional[_AgoraConnectionObserver] = None
        self._audio_frame_observer: Optional[_AgoraAudioFrameObserver] = None
        self._stream_fragments: dict[str, dict[str, Any]] = {}
        self._first_remote_audio_enqueued = False

        # Callbacks
        self.on_user_joined: Optional[Callable[[int], None]] = None
        self.on_user_left: Optional[Callable[[int], None]] = None
        self.on_stream_message_callback: Optional[Callable[[dict[str, Any]], None]] = None

        self.logger.info(
            f"AgoraBridge initialized: channel={channel_name}, "
            f"user_id={user_id}, sample_rate={audio_sample_rate}, "
            f"video={self.video_width}x{self.video_height}@{self.video_fps}"
        )

    def initialize(self) -> bool:
        """Initialize Agora service and configure audio/video settings.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Agora service...")

            # Create Agora service
            self.agora_service = AgoraService()

            # Configure service
            config = AgoraServiceConfig()
            config.appid = self.app_id
            # Align with official Python Server SDK examples for AI server pipelines.
            config.audio_scenario = AudioScenarioType.AUDIO_SCENARIO_AI_SERVER
            config.enable_audio_processor = True
            config.enable_audio_device = False  # We handle audio ourselves
            config.enable_video = True

            # Initialize service
            ret = self.agora_service.initialize(config)
            if ret != 0:
                self.logger.error(f"Failed to initialize Agora service: {ret}")
                return False

            self.logger.info("Agora service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Agora service: {e}")
            return False

    def join_channel(self, token: str) -> bool:
        """Join the Agora RTC channel.

        Args:
            token: RTC token for authentication

        Returns:
            True if joined successfully, False otherwise
        """
        try:
            self.logger.info(f"Joining channel: {self.channel_name}")

            # Create connection configuration
            conn_config = RTCConnConfig(
                auto_subscribe_audio=1,
                auto_subscribe_video=1,
                enable_audio_recording_or_playout=0,
                client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
                channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
            )

            # Create publish configuration
            publish_config = RtcConnectionPublishConfig(
                audio_profile=AudioProfileType.AUDIO_PROFILE_DEFAULT,
                audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
                is_publish_audio=True,
                audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
                is_publish_video=True,
                video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_YUV,
            )

            # Create RTC connection
            self.connection = self.agora_service.create_rtc_connection(conn_config, publish_config)
            if not self.connection:
                self.logger.error("Failed to create RTC connection")
                return False
            self._left_channel = False

            # Register connection observer
            self._connection_observer = _AgoraConnectionObserver(self)
            ret_conn_observer = self.connection.register_observer(self._connection_observer)
            if ret_conn_observer != 0:
                self.logger.warning("register_observer returned: %s", ret_conn_observer)

            # Register local user observer for datastream callback (on_stream_message)
            self._local_user_observer = _AgoraLocalUserObserver(self)
            ret_local_observer = self.connection.local_user._register_local_user_observer(
                self._local_user_observer
            )
            if ret_local_observer != 0:
                self.logger.warning(
                    "local_user._register_local_user_observer returned: %s",
                    ret_local_observer,
                )
            else:
                self.logger.info("Datastream observer registered")

            # Connect to channel
            # Configure pull parameters before connect to maximize callback compatibility.
            ret_playback_param = self.connection.local_user.set_playback_audio_frame_parameters(
                self.audio_channels,
                self.audio_sample_rate,
                0,  # RAW_AUDIO_FRAME_OP_MODE_READ_ONLY
                960,  # 60 ms @16k
            )
            if ret_playback_param != 0:
                self.logger.warning(
                    "set_playback_audio_frame_parameters returned: %s",
                    ret_playback_param,
                )

            # Register audio frame observer before connect (official sequence).
            self._audio_frame_observer = _AgoraAudioFrameObserver(self)
            ret_audio_observer = self.connection.register_audio_frame_observer(
                self._audio_frame_observer,
                0,
                None,
            )
            if ret_audio_observer != 0:
                self.logger.warning(
                    "register_audio_frame_observer returned: %s",
                    ret_audio_observer,
                )
            else:
                self.logger.info("Audio frame observer registered")

            # Use positional args for broader compatibility across SDK versions.
            # Official signature: connect(token, chan_id, user_id)
            ret = self.connection.connect(
                token,
                self.channel_name,
                str(self.user_id),
            )

            if ret != 0:
                self.logger.error(f"Failed to connect to channel: {ret}")
                return False

            # Defensive subscribe-all after connect.
            try:
                ret_sub_all = self.connection.local_user.subscribe_all_audio()
                if ret_sub_all != 0:
                    self.logger.warning("subscribe_all_audio returned: %s", ret_sub_all)
                else:
                    self.logger.info("Subscribed all remote audio tracks")
            except Exception:
                self.logger.exception("subscribe_all_audio failed")

            # Publish local tracks for external PCM/YUV pushing
            if not self._setup_audio_sender():
                return False

            # Start worker threads
            self._start_threads()

            self.logger.info(f"Successfully joined channel: {self.channel_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error joining channel: {e}")
            return False

    def _setup_audio_sender(self) -> bool:
        """Prepare audio/video publishing on the connection."""
        try:
            if not self.connection:
                self.logger.error("RTC connection is not ready")
                return False

            # The current Agora Python server SDK creates senders internally.
            # We only need to publish tracks, then push PCM/YUV via RTCConnection.
            video_config = VideoEncoderConfiguration(
                frame_rate=self.video_fps,
                codec_type=VideoCodecType.VIDEO_CODEC_H264,
                dimensions=VideoDimensions(
                    width=self.video_width,
                    height=self.video_height,
                ),
                bitrate=self.video_bitrate,
                min_bitrate=max(1, self.video_bitrate // 3),
                encode_alpha=0,
            )
            ret_cfg = self.connection.set_video_encoder_configuration(video_config)
            if ret_cfg != 0:
                self.logger.warning(
                    "set_video_encoder_configuration returned: %s "
                    "(%sx%s@%s, bitrate=%s)",
                    ret_cfg,
                    self.video_width,
                    self.video_height,
                    self.video_fps,
                    self.video_bitrate,
                )
            else:
                self._encoder_width = self.video_width
                self._encoder_height = self.video_height

            ret_audio = self.connection.publish_audio()
            if ret_audio != 0:
                self.logger.error(f"Failed to publish audio track: {ret_audio}")
                return False

            ret_video = self.connection.publish_video()
            self._video_enabled = ret_video == 0
            if not self._video_enabled:
                self.logger.warning(
                    f"Failed to publish video track: {ret_video}. "
                    "Video sending will be disabled."
                )

            self.logger.info("Audio/video publish tracks prepared")
            return True

        except Exception as e:
            self.logger.error(f"Error setting up audio sender: {e}")
            return False
    def _start_threads(self):
        """Start worker threads for audio/video processing."""
        self.running = True
        
        # Audio sending thread
        audio_send_thread = threading.Thread(
            target=self._audio_send_loop,
            name="AudioSendThread",
            daemon=True
        )
        audio_send_thread.start()
        self.threads.append(audio_send_thread)
        
        # Video sending thread
        video_send_thread = threading.Thread(
            target=self._video_send_loop,
            name="VideoSendThread",
            daemon=True
        )
        video_send_thread.start()
        self.threads.append(video_send_thread)
        
        self.logger.info("Worker threads started")

    def send_audio_frame(self, pcm_data: bytes) -> bool:
        """Send audio frame to Agora channel.
        
        Args:
            pcm_data: PCM audio data in bytes (int16 format)
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            if not self.running:
                return False
                
            self.audio_send_queue.put(pcm_data, block=False)
            return True
            
        except queue.Full:
            self.logger.warning("Audio send queue is full, dropping frame")
            return False

    def send_video_frame(self, yuv_data: bytes, width: int, height: int) -> bool:
        """Send video frame to Agora channel.
        
        Args:
            yuv_data: YUV I420 video data in bytes
            width: Frame width
            height: Frame height
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            if not self.running:
                return False
            if not self._video_enabled:
                return False

            expected_size = (int(width) * int(height) * 3) // 2
            if len(yuv_data) != expected_size:
                self.logger.warning(
                    "Invalid I420 frame size: got=%s expected=%s (w=%s h=%s)",
                    len(yuv_data),
                    expected_size,
                    width,
                    height,
                )
                return False

            # Keep encoder dimensions aligned with the real frame source.
            frame_w = int(width)
            frame_h = int(height)
            if (
                self.connection
                and (self._encoder_width != frame_w or self._encoder_height != frame_h)
            ):
                ret_cfg = self.connection.set_video_encoder_configuration(
                    VideoEncoderConfiguration(
                        frame_rate=self.video_fps,
                        codec_type=VideoCodecType.VIDEO_CODEC_H264,
                        dimensions=VideoDimensions(width=frame_w, height=frame_h),
                        bitrate=self.video_bitrate,
                        min_bitrate=max(1, self.video_bitrate // 3),
                        encode_alpha=0,
                    )
                )
                if ret_cfg == 0:
                    self._encoder_width = frame_w
                    self._encoder_height = frame_h
                    self.logger.info(
                        "Video encoder config updated to %sx%s",
                        frame_w,
                        frame_h,
                    )
                else:
                    self.logger.warning(
                        "Dynamic set_video_encoder_configuration failed: %s (%sx%s)",
                        ret_cfg,
                        frame_w,
                        frame_h,
                    )
                    return False
                
            frame_info = {
                'data': yuv_data,
                'width': frame_w,
                'height': frame_h,
                'timestamp': int(time.time() * 1000)
            }
            self.video_send_queue.put(frame_info, block=False)
            return True
            
        except queue.Full:
            self.logger.warning("Video send queue is full, dropping frame")
            return False

    def _video_send_loop(self):
        """Video sending loop (runs in separate thread)."""
        self.logger.info("Video send loop started")

        while self.running:
            try:
                frame_info = self.video_send_queue.get(timeout=0.1)

                if self.connection and self._video_enabled and frame_info:
                    width = int(frame_info["width"])
                    height = int(frame_info["height"])
                    # Agora external video frame expects writable buffer
                    raw_yuv = frame_info["data"]
                    writable_yuv = (
                        raw_yuv if isinstance(raw_yuv, bytearray)
                        else bytearray(raw_yuv)
                    )

                    frame = ExternalVideoFrame(
                        type=1,  # raw data
                        format=1,  # I420
                        buffer=writable_yuv,
                        stride=width,
                        height=height,
                        rotation=0,
                        timestamp=int(frame_info["timestamp"]),
                    )
                    ret = self.connection.push_video_frame(frame)
                    if ret != 0:
                        self.logger.warning(f"push_video_frame failed: {ret}")

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in video send loop: {e}")
                time.sleep(0.01)

        self.logger.info("Video send loop stopped")

    def get_received_audio(self) -> Optional[bytes]:
        """Get received audio frame from Agora channel.
        
        Returns:
            PCM audio data in bytes, or None if no data available
        """
        try:
            return self.audio_recv_queue.get(block=False)
        except queue.Empty:
            return None

    def get_received_audio_blocking(self, timeout_s: float = 0.1) -> Optional[bytes]:
        """Get received audio with timeout to reduce polling jitter."""
        try:
            return self.audio_recv_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def get_received_stream_message(self) -> Optional[dict[str, Any]]:
        """Get one received Agora datastream message (if available)."""
        try:
            return self.stream_recv_queue.get(block=False)
        except queue.Empty:
            return None

    def _handle_stream_message(self, user_id: str, stream_id: int, data: bytes, length: int):
        """Handle incoming datastream payload from Agora SDK callback."""
        raw = bytes(data[:length]) if data else b""
        text = ""
        parsed_json: Optional[Any] = None

        if raw:
            try:
                text = raw.decode("utf-8")
                try:
                    parsed_json = json.loads(text)
                except json.JSONDecodeError:
                    parsed_json = None
            except UnicodeDecodeError:
                text = ""

        # BK-style datastream framing: msg_id|index|total|base64_payload
        if parsed_json is None and text:
            decoded_text, decoded_json = self._try_decode_framed_stream_text(text)
            if decoded_text is None and decoded_json is None:
                # Incomplete fragment; wait for next packets
                return
            if decoded_text is not None:
                text = decoded_text
            if decoded_json is not None:
                parsed_json = decoded_json

        message = {
            "user_id": user_id,
            "stream_id": int(stream_id),
            "length": int(length),
            "raw": raw,
            "text": text,
            "json": parsed_json,
            "timestamp_ms": int(time.time() * 1000),
        }

        self._stream_messages_received += 1

        try:
            self.stream_recv_queue.put_nowait(message)
        except queue.Full:
            # Drop oldest message to keep newest data flowing.
            try:
                _ = self.stream_recv_queue.get_nowait()
                self.stream_recv_queue.put_nowait(message)
            except queue.Empty:
                pass

        if self.on_stream_message_callback:
            try:
                self.on_stream_message_callback(message)
            except Exception:
                self.logger.exception("on_stream_message_callback failed")

    def _try_decode_framed_stream_text(self, text: str) -> tuple[Optional[str], Optional[Any]]:
        """Decode BK-style framed datastream text into JSON string/object.

        Returns:
            (decoded_text, decoded_json)
            - decoded_text/json are None,None when fragment is incomplete.
        """
        # Already plain JSON text.
        try:
            parsed = json.loads(text)
            return text, parsed
        except json.JSONDecodeError:
            pass

        parts = text.split("|", 3)
        if len(parts) != 4:
            return text, None

        msg_id, index_str, total_str, payload_part = parts
        try:
            index = int(index_str)
            total = int(total_str)
        except ValueError:
            return text, None

        if not msg_id or total <= 0 or index <= 0 or index > total:
            return text, None

        now = time.time()
        self._cleanup_stream_fragments(now)

        entry = self._stream_fragments.get(msg_id)
        if entry is None or entry.get("total") != total:
            entry = {"total": total, "parts": {}, "ts": now}
            self._stream_fragments[msg_id] = entry

        entry["parts"][index] = payload_part
        entry["ts"] = now

        if len(entry["parts"]) < total:
            return None, None

        ordered = [entry["parts"].get(i) for i in range(1, total + 1)]
        if any(part is None for part in ordered):
            return None, None

        b64_payload = "".join(ordered)
        self._stream_fragments.pop(msg_id, None)

        try:
            decoded_bytes = base64.b64decode(b64_payload, validate=False)
        except (binascii.Error, ValueError):
            self.logger.warning("Datastream base64 decode failed for msg_id=%s", msg_id)
            return text, None

        try:
            decoded_text = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            self.logger.warning("Datastream utf-8 decode failed for msg_id=%s", msg_id)
            return text, None

        try:
            decoded_json = json.loads(decoded_text)
        except json.JSONDecodeError:
            decoded_json = None

        return decoded_text, decoded_json

    def _cleanup_stream_fragments(self, now: Optional[float] = None) -> None:
        """Remove stale fragmented datastream entries."""
        current = time.time() if now is None else now
        stale_keys = [
            msg_id
            for msg_id, entry in self._stream_fragments.items()
            if current - float(entry.get("ts", 0.0)) > 10.0
        ]
        for msg_id in stale_keys:
            self._stream_fragments.pop(msg_id, None)

    def _handle_received_audio_frame(self, frame: Any) -> None:
        """Convert received Agora audio frame to int16 mono PCM bytes and enqueue."""
        try:
            if frame is None or frame.buffer is None:
                return

            raw = bytes(frame.buffer)
            if not raw:
                return

            bytes_per_sample = int(getattr(frame, "bytes_per_sample", 2))
            channels = int(getattr(frame, "channels", 1))
            if bytes_per_sample != 2:
                self.logger.debug(
                    "Skip non-int16 frame: bytes_per_sample=%s", bytes_per_sample
                )
                return

            if channels <= 1:
                pcm_mono = raw
            else:
                pcm = np.frombuffer(raw, dtype=np.int16)
                usable_len = (len(pcm) // channels) * channels
                if usable_len <= 0:
                    return
                pcm = pcm[:usable_len].reshape(-1, channels)
                mono = pcm.mean(axis=1).astype(np.int16)
                pcm_mono = mono.tobytes()

            try:
                self.audio_recv_queue.put_nowait(pcm_mono)
                if not self._first_remote_audio_enqueued:
                    self.logger.info(
                        "First remote audio frame enqueued (bytes=%s, channels=%s, sample_rate=%s)",
                        len(pcm_mono),
                        channels,
                        getattr(frame, "samples_per_sec", "unknown"),
                    )
                    self._first_remote_audio_enqueued = True
            except queue.Full:
                # Keep freshest audio flowing.
                try:
                    _ = self.audio_recv_queue.get_nowait()
                    self.audio_recv_queue.put_nowait(pcm_mono)
                except queue.Empty:
                    pass

        except Exception as e:
            self.logger.debug(f"Failed to handle received audio frame: {e}")

    def _audio_send_loop(self):
        """Audio sending loop (runs in separate thread)."""
        self.logger.info("Audio send loop started")
        
        while self.running:
            try:
                # Get audio data from queue (with timeout)
                pcm_data = self.audio_send_queue.get(timeout=0.1)
                
                if self.connection and pcm_data:
                    writable_pcm = (
                        pcm_data if isinstance(pcm_data, bytearray)
                        else bytearray(pcm_data)
                    )
                    # SDK API: push_audio_pcm_data(data, sample_rate, channels, start_pts=0)
                    ret = self.connection.push_audio_pcm_data(
                        writable_pcm,
                        self.audio_sample_rate,
                        self.audio_channels,
                    )
                    if ret != 0:
                        self.logger.warning(f"push_audio_pcm_data failed: {ret}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio send loop: {e}")
                time.sleep(0.01)
        
        self.logger.info("Audio send loop stopped")

    def leave_channel(self):
        """Leave the Agora channel and cleanup resources."""
        with self._leave_lock:
            if self._left_channel:
                return
            self._left_channel = True

        self.logger.info("Leaving channel...")
        
        # Stop threads
        self.running = False
        
        # Wait for threads to finish
        for thread in list(self.threads):
            if thread.is_alive():
                thread.join(timeout=2.0)
        self.threads.clear()

        # Unpublish tracks before disconnect/release to avoid SDK finalization aborts.
        if self.connection:
            try:
                if self._local_user_observer is not None:
                    ret = self.connection._unregister_local_user_observer()
                    if ret != 0:
                        self.logger.warning(
                            "local_user._unregister_local_user_observer returned: %s",
                            ret,
                        )
                    self._local_user_observer = None
            except Exception as e:
                self.logger.warning(f"Error unregistering local user observer: {e}")

            try:
                if self._audio_frame_observer is not None:
                    ret = self.connection._unregister_audio_frame_observer()
                    if ret != 0:
                        self.logger.warning(
                            "_unregister_audio_frame_observer returned: %s",
                            ret,
                        )
                    self._audio_frame_observer = None
            except Exception as e:
                self.logger.warning(f"Error unregistering audio frame observer: {e}")

            try:
                ret_audio = self.connection.unpublish_audio()
                if ret_audio != 0:
                    self.logger.warning(f"unpublish_audio returned: {ret_audio}")
                else:
                    self.logger.info("Audio unpublished")
            except Exception as e:
                self.logger.warning(f"Error unpublishing audio: {e}")

            try:
                ret_video = self.connection.unpublish_video()
                if ret_video != 0:
                    self.logger.warning(f"unpublish_video returned: {ret_video}")
                else:
                    self.logger.info("Video unpublished")
            except Exception as e:
                self.logger.warning(f"Error unpublishing video: {e}")

        # Disconnect from channel
        if self.connection:
            try:
                ret = self.connection.disconnect()
                if ret is not None and ret != 0:
                    self.logger.warning(f"disconnect returned: {ret}")
                else:
                    self.logger.info("Disconnected from channel")
            except Exception as e:
                self.logger.error(f"Error disconnecting: {e}")

        # Explicitly release RTCConnection before releasing AgoraService.
        if self.connection:
            try:
                ret = self.connection.release()
                if ret is not None and ret != 0:
                    self.logger.warning(f"connection.release returned: {ret}")
                else:
                    self.logger.info("RTC connection released")
            except Exception as e:
                self.logger.warning(f"Error releasing RTC connection: {e}")
            finally:
                self.connection = None

        # Cleanup Agora service
        if self.agora_service:
            try:
                ret = self.agora_service.release()
                if ret is not None and ret != 0:
                    self.logger.warning(f"agora_service.release returned: {ret}")
                else:
                    self.logger.info("Agora service released")
            except Exception as e:
                self.logger.error(f"Error releasing service: {e}")
            finally:
                self.agora_service = None

        self.logger.info("Channel left successfully")

    def is_connected(self) -> bool:
        """Check if connected to Agora channel.
        
        Returns:
            True if connected, False otherwise
        """
        return self.running and self.connection is not None
