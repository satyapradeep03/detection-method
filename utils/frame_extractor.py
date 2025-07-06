import cv2
import numpy as np
from typing import List, Union
from loguru import logger


def extract_frames(
    source: Union[str, np.ndarray],
    fps: int = 1,
    max_frames: int = 100,
    target_size: tuple = (1080, 1920),
) -> List[np.ndarray]:
    """
    Extract frames from a video file or image.
    Args:
        source: Path to video/image file or numpy array (image)
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract
        target_size: (height, width) to downscale frames
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    try:
        if isinstance(source, np.ndarray):
            # Single image
            frame = cv2.resize(source, (target_size[1], target_size[0]))
            frames.append(frame)
            return frames
        # Try to open as video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video/image: {source}")
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = int(video_fps // fps) if fps < video_fps else 1
        count = 0
        idx = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
                frames.append(frame)
                count += 1
            idx += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {source}")
    except Exception as e:
        logger.exception(f"Error extracting frames: {e}")
    return frames 