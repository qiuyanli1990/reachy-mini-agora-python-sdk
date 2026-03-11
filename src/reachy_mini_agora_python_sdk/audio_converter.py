import numpy as np

class AudioConverter:
    """Reachy Mini 和 Agora 之间的音频格式转换"""

    @staticmethod
    def reachy_to_agora(reachy_audio: np.ndarray) -> bytes:
        """
        Reachy: float32 [-1.0, 1.0], shape (samples, 2) 立体声
        Agora: int16 PCM, mono
        """
        # 转为单声道（取平均）
        if reachy_audio.ndim == 2 and reachy_audio.shape[1] == 2:
            mono_audio = reachy_audio.mean(axis=1)
        else:
            mono_audio = reachy_audio

        # 转换为 int16
        pcm_audio = (mono_audio * 32767).astype(np.int16)

        # Agora SDK expects a writable underlying buffer in push_audio_pcm_data.
        return bytearray(pcm_audio.tobytes())

    @staticmethod
    def agora_to_reachy(
        pcm_bytes: bytes,
        num_samples: int,
        playback_gain: float = 1.0,
    ) -> np.ndarray:
        """
        Agora: int16 PCM bytes, mono
        Reachy: float32 [-1.0, 1.0], shape (samples, 2) 立体声
        """
        # 解析 int16
        pcm_audio = np.frombuffer(pcm_bytes, dtype=np.int16)

        # 转换为 float32
        float_audio = pcm_audio.astype(np.float32) / 32767.0

        # Apply playback gain and clamp to valid range.
        gain = max(0.0, min(float(playback_gain), 2.0))
        float_audio = np.clip(float_audio * gain, -1.0, 1.0)

        # 转为立体声（复制通道）
        stereo_audio = np.column_stack([float_audio, float_audio])

        return stereo_audio
