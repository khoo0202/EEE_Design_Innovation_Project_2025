import time, random, collections, cv2, threading, queue, copy
from datetime import datetime
from typing import Dict, Optional
import config as cfg

from .verifier import BehaviorVerifier
from .detector import PersonDetector
from .tracker import StableTracker
from .snapshotter import Snapshotter
from .results import ResultsWriter
from .calibrator import Calibrator 
from .vqa_http import HTTPVQA

RECHECK_NONE_SEC = 10.0
VQA_MIN = 3.0
VQA_MAX = 5.0

_next_ok: Dict[int, float] = {}
_track_state: Dict[int, dict] = {}

NEG_DRINK = {
    "bottle", "water bottle", "sports bottle", "cap", "unscrewing", "label",
    "cup", "mug", "thermos", "flask", "straw", "sippy", "can", "soda", "juice",
    "coffee", "tea", "latte", "drink", "drinking", "sip", "sipping", "gulp"
}
POS_SMOKE = {
    "cigarette", "cigar", "lighter", "match", "ash", "ashtray", "ember",
    "smoke", "smoking", "puff", "exhale", "inhaling", "vapor", "vape", "e-cig"
}

def enlarge_box_upper(b, k, W, H, top_bias=0.7):
    try:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    except Exception:
        return (0, 0, 1, 1)

    h = max(1, y2 - y1)

    y2_upper = y1 + int(h * top_bias)
    y2_upper = min(H, max(y1 + 1, y2_upper))

    cx = (x1 + x2) / 2.0
    base_w = max(1, x2 - x1)
    base_h = max(1, y2_upper - y1)

    bw = base_w * k
    bh = base_h * k

    extra_up = 0.12 * bh

    nx1 = int(round(cx - bw / 2))
    nx2 = int(round(cx + bw / 2))
    ny1 = int(round(y1 - extra_up))
    ny2 = int(round(ny1 + bh))

    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(W, nx2); ny2 = min(H, ny2)

    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)

    return (nx1, ny1, nx2, ny2)

class SmokingVapingPipeline:
    def __init__(self, _cfg):

        self.detector = PersonDetector(cfg.YOLO_WEIGHTS,
                                       cfg.CONF_THRESH,
                                       cfg.IOU_THRESH,
                                       cfg.DEVICE)
        self.tracker  = StableTracker(
            iou_thresh   = cfg.TRACK_IOU,
            max_age      = cfg.TRACK_MAX_AGE,
            smooth_alpha = getattr(cfg, "SMOOTH_ALPHA", 0.7),
            lost_ttl_sec = getattr(cfg, "LOST_TTL_SEC", 10.0),
            hist_thresh  = getattr(cfg, "HIST_THRESH", 0.3),
        )
        self.snap     = Snapshotter(cfg.SNAPSHOT_SECS,
                                    cfg.ASSUMED_FPS,
                                    cfg.SNAP_DIR)

        self.vqa_client = HTTPVQA(
                cfg.VQA_HTTP_URL,
                timeout=(5, getattr(cfg, "VQA_TIMEOUT_S", 10)),
                max_frames=getattr(cfg, "OLLAMA_MAX_FRAMES", 2),
            )
        
        self.behavior_verifier = None
        try:
            self.behavior_verifier = BehaviorVerifier(cfg.BEHAVIOR_MODEL_PATH)
            self.behavior_verifier_thresh = float(getattr(cfg, "BEHAVIOR_MODEL_THRESH", 0.60))
        except Exception as e:
            self.behavior_verifier = None

        calib_path = getattr(cfg, "CALIBRATOR_PATH", "./out/best_model_weightage.joblib")
        self.calibrator = Calibrator(calib_path)

        now = datetime.now()
        session_id    = now.strftime("%Y%m%d_%H%M%S")
        session_label = now.strftime("%d%b%Y%H%M%S")
        source_type   = "RTSP" if str(cfg.RTSP_OR_FILE).lower().startswith(("rtsp://","rtsps://")) else "FILE"


        self.writer = ResultsWriter(
            db_path       = getattr(cfg, "DB_PATH", "./out/violations.db"),
            evidence_root = getattr(cfg, "EVIDENCE_ROOT", "./out/evidence"),
            session_id    = session_id,
            session_label = session_label,
            source_type   = source_type,
        )

        self.interval    = cfg.SEND_INTERVAL_S
        self.last_sent   = {}
        self.source_name = str(cfg.RTSP_OR_FILE)

        self.on_result = lambda d: None

        self.job_q    = queue.Queue(maxsize=getattr(cfg, "VQA_QUEUE_SIZE", 32))
        self.done_q   = queue.Queue()

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker_thread.start()

    @property
    def session_id(self) -> str:
        return self.writer.session_id

    @property
    def session_label(self) -> str:
        return self.writer.session_label
    
    def _ensure_state(self, tid: int):
        st = _track_state.get(tid)
        if st is None:
            st = {
                "samples": [],
                "confirmed": False,
                "last_none_check": 0.0,
                "violation_id": None,
                "lock_label": None,
                "lock_counter": 0,
                "recent_history": [],
            }
            _track_state[tid] = st
        else:
            if "samples" not in st:           st["samples"] = []
            if "confirmed" not in st:         st["confirmed"] = False
            if "last_none_check" not in st:   st["last_none_check"] = 0.0
            if "violation_id" not in st:      st["violation_id"] = None
            if "lock_label" not in st:        st["lock_label"] = None
            if "lock_counter" not in st:      st["lock_counter"] = 0
            if "recent_history" not in st:    st["recent_history"] = []
        return st

    def _worker_loop(self):
        while True:
            try:
                tid, frames_opt, clip_opt, bbox, frame_bgr = self.job_q.get()
            except Exception:
                continue

            try:
                if frames_opt is not None:
                    vqa_out = self.vqa_client.classify_frames(frames_opt)
                else:
                    vqa_out = self.vqa_client.classify_clip(clip_opt)

                scores = vqa_out.get("scores", {}) or {}
                desc   = vqa_out.get("description", "") or ""

                self.done_q.put({
                    "tid": tid,
                    "bbox": bbox,
                    "frame": frame_bgr,
                    "scores": scores,
                    "description": desc,
                })
            except Exception:
                pass

    def _decide_after_5(self, tid: int):
        st = self._ensure_state(tid)

        if st.get("confirmed"):
            return None

        samples = st.get("samples", [])
        if len(samples) < 5:
            return None

        votes = collections.Counter([c for (c, _, _, _, _) in samples])
        top, cnt = votes.most_common(1)[0]

        if top in ("smoking", "vaping") and cnt >= 3:
            final_label = top
        else:
            st["last_none_check"] = time.time()
            return ("none", None, None)

        cls_last, _, desc_last, frame_last, box_last = samples[-1]

        st["confirmed"] = True
        st["lock_counter"] = 1

        if self.behavior_verifier and final_label in ("smoking", "vaping"):
            # pick the latest person crop for this track
            x1, y1, x2, y2 = map(int, box_last)  # your last bbox
            crop = frame_last[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            verified_label, conf = self.behavior_verifier.predict(crop)

            # only flip when verifier is confident AND disagrees
            if conf >= self.behavior_verifier_thresh and verified_label != final_label:
                final_label = verified_label

        st["lock_label"]   = final_label
        
        info = self.writer.save_main(
            full_frame_bgr = frame_last,
            bbox           = box_last,
            track_id       = tid,
            predicted_label= final_label,
            description    = desc_last
        )
        violation_id = info["violation_id"]
        st["violation_id"] = violation_id

        for i, (c_i, _, desc_i, frame_i, box_i) in enumerate(samples[:5], start=1):
            self.writer.save_review_image(
                violation_id,
                i,
                frame_i,
                box_i
            )

        self.on_result({
            "track_id": tid,
            "class": final_label,
            "probs": {},
            "bbox": box_last,
            "img_path": info["image_path"],
            "confirmed": True,
        })

        return ("confirmed", final_label, violation_id)

    def _maybe_downgrade_after_confirm(self, tid: int):
        st = self._ensure_state(tid)
        if not st.get("confirmed"):
            return
        hist = st.get("recent_history", [])
        if len(hist) < 4:
            return
        last4 = hist[-4:]
        none_ct = sum(1 for c in last4 if c == "none")
        if none_ct >= 3:
            st["lock_label"] = "none"

    def _apply_text_guardrails(self, cls_in: str, probs_in: dict, desc_lower: str):
        NON_SMOKE_HINTS = {
            "phone", "cellphone", "cell phone", "smartphone", "talking on the phone",
            "call", "calling", "speaking on phone", "scrolling phone", "texting",
            "bottle", "water bottle", "sports bottle", "drinking", "drink", "sip",
            "sipping", "straw", "unscrewing cap", "unscrewing the cap",
            "cup", "mug", "tumbler", "thermos", "flask", "can", "soda", "juice",
            "coffee", "tea", "latte", "milk tea", "bubble tea", "boba",
            "wipe mouth", "wiping mouth", "scratch face", "scratching face",
            "touching face", "hand near mouth", "hand near face",
            "mask", "wearing a mask", "adjusting mask",
            "eating", "food", "snack", "chips", "sandwich", "bite", "biting",
            "tissue", "napkin",
        }
        SMOKE_HINTS = {
            "cigarette", "cigar", "lighter", "lighting", "ash",
            "puff", "puffing", "inhale", "inhaling", "exhale", "exhaling",
            "smoking", "smoke",
            "vape", "vaping", "e-cig", "e cigarette", "e-cigarette", "vapor cloud",
        }

        has_non_smoke = any(tok in desc_lower for tok in NON_SMOKE_HINTS)
        has_smoke     = any(tok in desc_lower for tok in SMOKE_HINTS)

        if has_non_smoke and not has_smoke:

            return "none", {"none": 0.95, "smoking": 0.03, "vaping": 0.02}

        return cls_in, probs_in

    def _handle_vqa_result(self, tid: int, vqa: dict, bbox, frame):
        st = self._ensure_state(tid)

        raw_scores = vqa.get("scores", {}) or {}
        desc_raw   = vqa.get("description", "") or ""
        desc_lower = desc_raw.lower()

        cls_cal, pmax, probs_cal = self.calibrator.top1(raw_scores)

        cls_final, probs_final = self._apply_text_guardrails(cls_cal, probs_cal, desc_lower)

        st["recent_history"].append(cls_final)
        if len(st["recent_history"]) > 8:
            st["recent_history"] = st["recent_history"][-8:]

        st["samples"].append(
            (cls_final, max(probs_final.values()), desc_raw, None if frame is None else frame.copy(), bbox)
        )
        if len(st["samples"]) > 5:
            st["samples"] = st["samples"][-5:]

        self.on_result({
            "track_id": tid,
            "class": cls_final,
            "probs": probs_final,
            "bbox": bbox,
            "img_path": "",
            "confirmed": st.get("confirmed", False),
        })

        if st["confirmed"]:
            self._maybe_downgrade_after_confirm(tid)
            return

        decision = self._decide_after_5(tid)
        if not decision:
            return
        tag, top_label, vid = decision
        if tag == "confirmed":
            return
        elif tag == "none":
            return

    def poll_async_results(self):
        while True:
            try:
                item = self.done_q.get_nowait()
            except queue.Empty:
                break

            tid        = item["tid"]
            bbox       = item["bbox"]
            frame_bgr  = item["frame"]
            scores     = item["scores"]
            desc       = item["description"]

            self._handle_vqa_result(
                tid,
                {"scores": scores, "description": desc},
                bbox,
                frame_bgr
            )

    def get_latest_crop_for_track(self, tid: int):
        st = self._ensure_state(tid)
        if not st["samples"]:
            return None, None
        cls_i, _, _, frm, bb = st["samples"][-1]
        if frm is None or bb is None:
            return None, bb
        x1, y1, x2, y2 = bb
        h_img, w_img = frm.shape[:2]
        x1 = max(0, min(int(x1), w_img-1))
        x2 = max(0, min(int(x2), w_img-1))
        y1 = max(0, min(int(y1), h_img-1))
        y2 = max(0, min(int(y2), h_img-1))
        if x2 <= x1 or y2 <= y1:
            return None, bb
        crop = frm[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None, bb
        return crop, bb

    def process_frame(self, frame):
        dets   = self.detector(frame)
        tracks = self.tracker.update(dets, frame)

        H, W = frame.shape[:2]
        for tid, bbox in tracks:
            ub = enlarge_box_upper(bbox, k=1.5, W=W, H=H, top_bias=0.7)
            self.snap.push(tid, frame, ub)

        self.poll_async_results()

        now = time.time()
        for tid, bbox in tracks:
            st = self._ensure_state(tid)

            if st.get("confirmed"):
                _next_ok.pop(tid, None)
                continue

            if now < _next_ok.get(tid, 0.0):
                continue

            if len(st.get("samples", [])) >= 5:
                last_none = st.get("last_none_check", 0.0)
                if now - last_none < RECHECK_NONE_SEC:
                    continue

            frames_opt = None
            clip_opt   = None
            if hasattr(self.vqa_client, "classify_frames"):
                frames_opt = self.snap.get_frames(
                    tid,
                    maxn=getattr(cfg, "OLLAMA_MAX_FRAMES", 2)
                )

            if not frames_opt:
                clip_opt = self.snap.make_temp_clip(tid, fps=8)
                if not clip_opt:
                    continue

            frame_for_track = copy.deepcopy(frame)

            try:
                self.job_q.put_nowait(
                    (tid, frames_opt, clip_opt, bbox, frame_for_track)
                )
                _next_ok[tid] = now + random.uniform(VQA_MIN, VQA_MAX)
                self.last_sent[tid] = now
            except queue.Full:
                pass

        return tracks
