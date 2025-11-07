import json, joblib
import numpy as np
from pathlib import Path


class Calibrator:
    def __init__(self, path: str):
        self.path = Path(path)
        self.model = None
        if self.path.exists():
            try:
                payload = joblib.load(self.path)
                if isinstance(payload, dict) and payload.get("type") == "logreg":
                    self.model = payload["sk_pipeline"]
                    self.classes_ = list(payload.get("classes", ["smoking", "vaping", "none"]))
                    print(f"[Calibrator] Loaded logistic model from {self.path}")
                else:
                    print(f"[Calibrator] {self.path} is not a logreg payload; using simple normalize.")
            except Exception as e:
                print(f"[Calibrator] Failed to load {self.path}: {e}. Using simple normalize.")
        else:
            print(f"[Calibrator] No calibrator at {self.path}; using simple normalize.")

    def _simple(self, scores: dict):
        if not scores:
            return "none", 0.34, {"smoking": 0.33, "vaping": 0.33, "none": 0.34}
        arr = np.array([max(0.0, float(scores.get("smoking", 0.0))),
                        max(0.0, float(scores.get("vaping", 0.0))),
                        max(0.0, float(scores.get("none", 0.0)))], dtype=float)
        s = arr.sum()
        arr = arr / (s if s > 0 else 1.0)
        probs = {"smoking": float(arr[0]), "vaping": float(arr[1]), "none": float(arr[2])}
        cls = max(probs, key=probs.get)
        return cls, float(probs[cls]), probs

    def top1(self, scores: dict):
        if self.model is None:
            return self._simple(scores)

        # Build feature row in the same class order the trainer used
        feat = np.array([[float(scores.get("smoking", 0.0)),
                          float(scores.get("vaping", 0.0)),
                          float(scores.get("none", 0.0))]], dtype=np.float32)

        try:
            probs = self.model.predict_proba(feat)[0]
            # map to known classes
            out = {}
            for i, c in enumerate(self.model.classes_):
                out[c] = float(probs[i])
            # ensure all 3 keys exist
            for c in ("smoking", "vaping", "none"):
                out.setdefault(c, 0.0)
            # renormalize for safety
            s = sum(out.values()) or 1.0
            out = {k: v / s for k, v in out.items()}
            cls = max(out, key=out.get)
            return cls, float(out[cls]), out
        except Exception:
            return self._simple(scores)
