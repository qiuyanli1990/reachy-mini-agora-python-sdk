import cv2
import numpy as np

class VideoConverter:
    """Reachy Mini 和 Agora 之间的视频格式转换"""

    @staticmethod
    def bgr_to_yuv_i420(bgr_frame: np.ndarray) -> bytes:
        """
        Reachy: BGR (height, width, 3) uint8
        Agora: YUV I420 格式
        """
        # 调整大小（如果需要）
        # bgr_frame = cv2.resize(bgr_frame, (width, height))

        # 转换为 YUV I420
        yuv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)

        return yuv_frame.tobytes()
