#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import json
import base64
from typing import List, Optional
from pathlib import Path
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os, threading, time
app = FastAPI()
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@app.post("/shutdown")
def shutdown():
    threading.Thread(target=lambda: (time.sleep(0.2), os._exit(0)), daemon=True).start()
    return {"ok": True}

MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5vl:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
PORT = int(os.environ.get("PORT", "8012"))
JPEG_QUALITY = int(os.environ.get("VQA_JPEG_Q", "85"))
NUM_CTX = int(os.environ.get("VQA_NUM_CTX", "4096"))
TEMPERATURE = float(os.environ.get("VQA_TEMP", "0.1"))
TOP_P = float(os.environ.get("VQA_TOP_P", "0.9"))
NUM_PREDICT = int(os.environ.get("VQA_NUM_PREDICT", "256"))
TIME_TOL = float(os.environ.get("VQA_TIME_TOL", "0.5"))
STRICTNESS = int(os.environ.get("VQA_STRICT", "2"))
LOG = os.environ.get("VQA_LOG", "1") == "1"

def _np_to_jpeg_b64(img: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS

def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS

def _open_cap(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    return cap if cap.isOpened() else None

def _probe_video(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    return float(fps), int(n)

def _safe_sample_indices(n_frames: int, k: int) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.linspace(0, max(0, n_frames - 1), num=k)
    return np.unique(np.clip(idx.round().astype(int), 0, max(0, n_frames - 1)))

def _read_frames_by_indices(path: str, indices: np.ndarray):
    cap = _open_cap(path)
    if cap is None:
        return None, "Video open failed"
    frames = []
    last_ok = None
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            if last_ok is not None and int(i) > last_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, last_ok)
                while last_ok < int(i):
                    ok2, f2 = cap.read()
                    if not ok2 or f2 is None:
                        break
                    last_ok += 1
                ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                return None, f"Decode failed at frame index {i}"
        frames.append(frame)
        last_ok = int(i)
    cap.release()
    return frames, None

def log(*a):
    if LOG: print("[VQA]", *a)

class Req(BaseModel):
    video_path: str = Field(..., description="Absolute path to local video")
    max_frames: int = Field(8, description="Uniformly sample this many frames")

class Resp(BaseModel):
    answer: str
    evidence_timestamp: Optional[float] = None
    description: Optional[str] = None
    meta: dict = {}

def sample_frames(video_path: str, n: int):
    path = os.path.abspath(video_path)
    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        frames, ts = [], []
        idx = 0
        while idx < n:
            ok, frame = cap.read()
            if not ok: break
            ts.append(len(ts) / max(1.0, fps))
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
            idx += 1
        cap.release()
        return frames, ts, {"fps": fps, "total_frames": total}
    n = max(1, min(n, total))
    idxs = [int(round(i * (total - 1) / (n - 1))) if n > 1 else 0 for i in range(n)]
    frames_b64, stamps = [], []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        ts = idx / max(1.0, fps)
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        stamps.append(ts)
    cap.release()
    log(f"samples={len(frames_b64)} timestamps={','.join(f'{t:.2f}' for t in stamps)}")
    return frames_b64, stamps, {"fps": fps, "total_frames": total}

SYSTEM_PROMPT = (
    'You are a video smoking and vaping detector. Decide if anyone is smoking.'
    'Smoking includes holding a cigarette/e-cig near the mouth, lighting it, inhaling, or exhaling visible smoke.'
    'Return a strict single-line JSON ONLY. The JSON MUST have: '
    '{"smoking":"YES|NO",'
    '"evidence_timestamp":<float>,'
    '"cues":[...],'
    '"negatives":[...],'
    '"description":"<=20 words",'
    '"scores":{"smoking":<0-1>,"vaping":<0-1>,"none":<0-1>}}'
    'Valid cues: "smoke","lighter","glowing_tip","hand_to_mouth".'
    'Valid negatives: "bottle","cup","straw","drinking","toothbrush","pen","eating".'
    'If you answer "YES", include at least one cue (prefer "smoke" or "lighter" when applicable).'
    'Pick exactly one evidence_timestamp from the allowed timestamps.'
    'The "scores" object is required. Its values should roughly reflect how likely each class is.'
)

def build_user_prompt(timestamps: List[float]) -> str:
    return (
        "Allowed timestamps (seconds): ["
        + ", ".join(f"{t:.2f}" for t in timestamps)
        + "]\nSelect exactly ONE evidence_timestamp from the list. Return JSON only, no extra text."
    )

def call_ollama(images_b64: List[str], system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt, "images": images_b64},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "seed": 42,
        },
    }
    url = f"{OLLAMA_URL}/api/chat"
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    data = r.json()
    text = (data.get("message") or {}).get("content", "") or ""
    log("raw_text:", text[:400].replace("\n", " ") + ("..." if len(text) > 400 else ""))
    return text

def parse_output(text: str):
    text2 = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
    m = re.search(r"\{[\s\S]*\}", text2)
    if not m:
        return "", None, [], [], None, {}, {"raw_text": text}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return "", None, [], [], None, {}, {"raw_text": text}

    cues = obj.get("cues") or []
    negatives = obj.get("negatives") or []
    description = obj.get("description")
    if isinstance(description, str):
        description = description.strip()
        if len(description) > 200:
            description = description[:200]

    scores_obj = obj.get("scores") or {}
    if not isinstance(scores_obj, dict):
        scores_obj = {}

    if "smoking" in obj or "evidence_timestamp" in obj:
        smoking = str(obj.get("smoking", "")).strip().lower()
        ans = "Yes" if smoking in ("yes","true","是") else ("No" if smoking in ("no","false","否") else "")
        ev = None
        if "evidence_timestamp" in obj:
            try:
                ev = float(obj["evidence_timestamp"])
            except Exception:
                ev = None
        return ans, ev, cues, negatives, description, scores_obj, {"schema":"A","obj":obj}

    ans_raw = (obj.get("answer") or "").strip().lower()
    ans = "Yes" if ans_raw in ("yes","true","是") else ("No" if ans_raw in ("no","false","否") else "")
    ev = None
    stamps = obj.get("evidence_timestamps") or []
    if isinstance(stamps, list) and stamps:
        try:
            ev = float(stamps[0])
        except Exception:
            ev = None
    return ans, ev, cues, negatives, description, scores_obj, {"schema":"B","obj":obj}

app = FastAPI(title="VQA Stricter (Adjustable Strictness + Description)")

@app.post("/vqa")
def vqa_endpoint(payload: dict):
    try:
        video_path = payload.get("video_path")
        max_frames = int(payload.get("max_frames", 8))

        if not video_path or not os.path.exists(video_path):
            return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"Path not found: {video_path}"}, status_code=200)

        frames: List[np.ndarray] = []
        stamps: List[float] = []

        if _is_image(video_path):
            img = cv2.imread(video_path)
            if img is None:
                return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"Failed to read image: {video_path}"}, status_code=200)
            frames = [img]
            stamps = [0.0]
        else:
            cap = _open_cap(video_path)
            if cap is None:
                return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"Failed to open video: {video_path}"}, status_code=200)

            fps, n = _probe_video(cap)
            if n <= 0:
                cap.release()
                return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"Zero frames: {video_path}"}, status_code=200)

            k = max(1, int(max_frames))
            idxs = _safe_sample_indices(n, k)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
                    stamps.append(float(i) / max(1.0, fps))
            cap.release()

            if not frames:
                return JSONResponse({"answer":"", "description":"", "scores":{}, "error": "Could not decode any frames"}, status_code=200)

        if len(frames) >= 3:
            sel = [0, len(frames)//2, len(frames)-1]
        elif len(frames) == 2:
            sel = [0, 1]
        else:
            sel = [0]

        images_b64 = [_np_to_jpeg_b64(frames[i]) for i in sel]
        images_b64 = [s for s in images_b64 if s]
        if not images_b64:
            return JSONResponse({"answer":"", "description":"", "scores":{}, "error":"JPEG encode failed"}, status_code=200)

        user_prompt = build_user_prompt([stamps[i] for i in sel] if stamps else [0.0]*len(images_b64))
        try:
            text = call_ollama(images_b64, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"Ollama call failed: {type(e).__name__}: {e}"}, status_code=200)

        ans, ev, cues, negatives, desc, scores_obj, meta = parse_output(text)
        if not ans:
            return {
                "answer": "",
                "evidence_timestamp": None,
                "description": text[:400],
                "scores": {},
                "meta": {"parse":"failed", **meta}
            }

        if not scores_obj:
            if ans == "Yes":
                base_smoke = 0.7 if STRICTNESS == 1 else (0.8 if STRICTNESS <= 0 else 0.6)
                scores_obj = {
                    "smoking": base_smoke,
                    "vaping":  0.0,
                    "none":    max(0.0, 1.0 - base_smoke),
                }
            else:
                base_none = 0.7 if STRICTNESS == 1 else (0.8 if STRICTNESS <= 0 else 0.6)
                scores_obj = {
                    "smoking": max(0.0, 1.0 - base_none),
                    "vaping":  0.0,
                    "none":    base_none,
                }

        return {
            "answer": ans,
            "evidence_timestamp": float(ev) if ev is not None else None,
            "description": desc or "",
            "scores": {
                "smoking": float(scores_obj.get("smoking", 0.0) or 0.0),
                "vaping":  float(scores_obj.get("vaping", 0.0) or 0.0),
                "none":    float(scores_obj.get("none", 0.0) or 0.0),
            },
            "meta": {
                "cues": cues,
                "negatives": negatives,
                **meta
            },
        }

    except Exception as e:
        return JSONResponse({"answer":"", "description":"", "scores":{}, "error": f"{type(e).__name__}: {e}"}, status_code=200)

if __name__ == "__main__":
    import uvicorn, sys
    print("[VQA] PY:", sys.executable)
    print("[VQA] MODEL:", MODEL_NAME)
    print(f"[VQA] STRICTNESS={STRICTNESS} (0=loose,1=default,2=strict)")
    print(f"[VQA] Listening on http://0.0.0.0:{PORT}")
    uvicorn.run("vqa_newest_desc_strictness:app", host="0.0.0.0", port=PORT, reload=False)
