import os, cv2, time, tempfile
import numpy as np
from collections import deque
from pathlib import Path


class Snapshotter:
    def __init__(self, seconds: float, fps: float, out_dir: str):
        self.buflen = max(3, int(seconds * fps))
        self.buffers = {}  # tid -> deque of frames
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def push(self, tid: int, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return
        if tid not in self.buffers:
            self.buffers[tid] = deque(maxlen=self.buflen)
        self.buffers[tid].append(crop)

    def get_frames(self, tid: int, maxn: int = 2):
        if tid not in self.buffers or not self.buffers[tid]:
            return []
        buf = list(self.buffers[tid])[-maxn:]
        return buf

    def make_temp_clip(self, tid: int, fps: int = 8):
        if tid not in self.buffers or not self.buffers[tid]:
            return None
        frames = list(self.buffers[tid])
        h, w = frames[0].shape[:2]
        tmp_path = Path(tempfile.gettempdir()) / f"track_{tid}_{int(time.time())}.mp4"
        out = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        return str(tmp_path)
