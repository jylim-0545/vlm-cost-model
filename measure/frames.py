"""Qwen-native fps-based video frame sampling (qwen_vl_utils smart_nframes logic).
Sampling frames at a fixed COUNT makes video length invisible to cost; fps-based
sampling makes longer videos -> more frames -> more vision tokens, which is the
length axis the price model needs. Defaults match Qwen2.5-VL native."""
from __future__ import annotations

import numpy as np

FPS = 2.0            # Qwen2.5-VL native target fps
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
FRAME_FACTOR = 2     # frames processed in temporal pairs (temporal_patch_size=2)


def _round_to_factor(x: float, f: int) -> int:
    return int(round(x / f) * f)


def qwen_num_frames(total_frames: int, video_fps: float, *, fps: float = FPS,
                    min_frames: int = FPS_MIN_FRAMES, max_frames: int = FPS_MAX_FRAMES,
                    factor: int = FRAME_FACTOR) -> int:
    """Native frame count: desired = duration*fps, rounded to `factor`, clamped to
    [min,max], and never exceeding what's in the file."""
    desired = total_frames / max(video_fps, 1e-6) * fps
    n = _round_to_factor(desired, factor)
    n = min(max(n, min_frames), max_frames)
    n = (n // factor) * factor
    n = max(n, factor)
    cap = (total_frames // factor) * factor or factor
    return int(min(n, cap))


def qwen_frame_indices(total_frames: int, video_fps: float, **kw) -> np.ndarray:
    n = qwen_num_frames(total_frames, video_fps, **kw)
    return np.linspace(0, total_frames - 1, num=n).round().astype(int)
