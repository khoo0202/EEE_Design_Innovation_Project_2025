from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, weights_path: str, conf: float, iou: float, device: str):
        self.model = YOLO(weights_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = getattr(self.model, "imgsz", 640)

    def __call__(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        res = self.model.predict(
            bgr, verbose=False, conf=self.conf, iou=self.iou, device=self.device, imgsz=self.imgsz
        )[0]

        out = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()

        for b, c, s in zip(xyxy, cls, conf):
            if int(c) == 0:  # COCO person class
                x1, y1, x2, y2 = map(int, b)
                out.append((x1, y1, x2, y2, float(s)))
        return out
