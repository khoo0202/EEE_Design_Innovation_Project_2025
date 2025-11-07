import cv2, requests, tempfile, os, base64
import numpy as np


class HTTPVQA:
    def __init__(self, base_url: str, timeout=(5, 15), max_frames: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_frames = max_frames

    def _encode(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode("utf-8")

    def classify_frames(self, frames: list[np.ndarray]) -> dict:
        if not frames:
            return {"raw_label": "none", "scores": {"smoking": 0.33, "vaping": 0.33, "none": 0.34}}

        # pick a representative crop (middle)
        img = frames[len(frames)//2]

        # write temp JPG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        cv2.imwrite(tmp_path, img)

        # new server contract: {"video_path": "<abs path>", "max_frames": N}
        payload = {"video_path": os.path.abspath(tmp_path), "max_frames": max(1, self.max_frames)}

        try:
            r = requests.post(self.base_url, json=payload, timeout=self.timeout)  # self.base_url should be "http://127.0.0.1:8012/vqa"
            r.raise_for_status()
            data = r.json()
            # new server returns: answer "Yes/No", prob, description
            ans = (data or {}).get("answer", "")
            scores = {"smoking": 0.5, "vaping": 0.0, "none": 0.5}
            if ans == "Yes":  # map to your 3-way schema (lightweight)
                scores = {"smoking": float(data.get("prob") or 0.7), "vaping": 0.1, "none": 1.0 - (float(data.get("prob") or 0.7) + 0.1)}
            elif ans == "No":
                scores = {"smoking": 0.05, "vaping": 0.05, "none": 0.9}
        except Exception as e:
            print(f"[HTTPVQA] error: {e}")
            scores = {"smoking": 0.33, "vaping": 0.33, "none": 0.34}
        finally:
            try: os.remove(tmp_path)
            except: pass

        label = max(scores, key=scores.get)
        return {"raw_label": label, "scores": scores}

    def classify_clip(self, clip_path: str) -> dict:
        # Optional: call server directly with the clip path (no local sampling)
        payload = {"video_path": os.path.abspath(clip_path), "max_frames": self.max_frames}
        try:
            r = requests.post(self.base_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            ans = (data or {}).get("answer", "")
            if ans == "Yes":
                p = float(data.get("prob") or 0.7)
                scores = {"smoking": p, "vaping": 0.1, "none": max(0.0, 1.0 - p - 0.1)}
            elif ans == "No":
                scores = {"smoking": 0.05, "vaping": 0.05, "none": 0.9}
            else:
                scores = {"smoking": 0.33, "vaping": 0.33, "none": 0.34}
        except Exception as e:
            print(f"[HTTPVQA] clip error: {e}")
            scores = {"smoking": 0.33, "vaping": 0.33, "none": 0.34}
        return {"raw_label": max(scores, key=scores.get), "scores": scores}
