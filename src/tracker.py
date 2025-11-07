import time, math, cv2
import numpy as np
from typing import Dict, Tuple, List
import onnxruntime as ort

class StableTracker:
    def __init__(self,
                 iou_thresh: float = 0.4,
                 max_age: int = 90,
                 alpha_iou: float = 0.55,
                 smooth_alpha: float = 0.7,
                 lost_ttl_sec: float = 20.0,
                 hist_thresh: float = 0.35,
                 match_w_iou: float = 1.0,
                 match_w_hist: float = 0.3,
                 match_min_score: float = 0.05,
                 onnx_reid_path: str = None,
                 reemerge_sec: float = 180.0,
                 reid_match_thresh: float = 0.7
                 ):

        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.smooth_alpha = smooth_alpha
        self.lost_ttl_sec = lost_ttl_sec
        self.hist_thresh = hist_thresh

        # weights for smarter global matching (helps avoid ID swap when people cross)
        self.match_w_iou = match_w_iou
        self.match_w_hist = match_w_hist
        self.match_min_score = match_min_score

        self.tracks: Dict[int, dict] = {}
        self.recent_pool: Dict[int, dict] = {}
        self.reemerge_sec = reemerge_sec
        self.reid_match_thresh = reid_match_thresh

        # bookkeeping
        self.next_id = 1
        self.frame_count = 0

        self._reid_sess = None
        self._reid_input_name = None
        self._reid_output_name = None
        if onnx_reid_path is not None:
            self._init_reid_session(onnx_reid_path)

        # active tracks currently visible
        self.active: Dict[int, dict] = {}

        # lost tracks recently disappeared (can be revived)
        self.lost: Dict[int, dict] = {}

    def _init_reid_session(self, onnx_path: str):

        self._reid_sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._reid_input_name = self._reid_sess.get_inputs()[0].name
        self._reid_output_name = self._reid_sess.get_outputs()[0].name

    def _reid_preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:

        img = cv2.resize(crop_bgr, (224,224), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        img = (img - mean)/std
        img = np.transpose(img, (2,0,1)) 
        img = np.expand_dims(img, 0)     
        return img
    
    def _reid_embed(self, crop_bgr: np.ndarray) -> np.ndarray:

        if self._reid_sess is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        x = self._reid_preprocess(crop_bgr)
        feat = self._reid_sess.run(
            [self._reid_output_name],
            {self._reid_input_name: x}
        )[0]  # shape (1, D)
        v = feat[0].astype(np.float32)
        n = np.linalg.norm(v) + 1e-8
        v = v / n
        return v

    ##########################################
    # BOX SANITIZER
    ##########################################
    def _box4(self, box_like):
        try:
            x1 = int(box_like[0])
            y1 = int(box_like[1])
            x2 = int(box_like[2])
            y2 = int(box_like[3])
        except Exception:
            return None
        return (x1, y1, x2, y2)

    ##########################################
    # IOU / CROP / HIST
    ##########################################
    def _iou(self, a, b):
        a4 = self._box4(a)
        b4 = self._box4(b)
        if a4 is None or b4 is None:
            return 0.0

        ax1, ay1, ax2, ay2 = a4
        bx1, by1, bx2, by2 = b4

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
        area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
        denom = float(area_a + area_b - inter + 1e-6)
        return inter / denom

    def _crop_upper(self, frame, box):
        b4 = self._box4(box)
        if b4 is None:
            return None

        x1, y1, x2, y2 = b4
        h = y2 - y1
        if h <= 0:
            return None

        y2_upper = int(y1 + 0.6 * h)
        if y2_upper <= y1:
            y2_upper = y1 + 1

        H, W = frame.shape[:2]
        if y2_upper > H:
            y2_upper = H
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        if x2 > W:
            x2 = W
        if y2_upper > H:
            y2_upper = H

        x1c = max(0, min(x1, W-1))
        x2c = max(0, min(x2, W-1))
        y1c = max(0, min(y1, H-1))
        y2c = max(0, min(y2_upper, H-1))

        if x2c <= x1c or y2c <= y1c:
            return None

        crop = frame[y1c:y2c, x1c:x2c]
        if crop is None or crop.size <= 0:
            return None
        return crop

    def _crop_full(self, frame, box):
        b4 = self._box4(box)
        if b4 is None:
            return None
        x1, y1, x2, y2 = b4
        H, W = frame.shape[:2]
        x1 = max(0, min(x1, W-1))
        x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1))
        y2 = max(0, min(y2, H-1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop

    def _hist_signature(self, frame, box):
        crop = self._crop_upper(frame, box)
        if crop is None:
            return None
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        except Exception:
            return None

        hist = cv2.calcHist(
            [hsv], [0,1,2],
            None,
            [16,16,4],
            [0,180, 0,256, 0,256]
        )
        hist = cv2.normalize(hist, hist).flatten().astype("float32")
        return hist

    def _hist_distance(self, h1, h2):
        if h1 is None or h2 is None:
            return 999.0
        if not hasattr(h1, "shape") or not hasattr(h2, "shape"):
            return 999.0
        if h1.shape != h2.shape:
            return 999.0
        try:
            corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        except Exception:
            return 999.0
        dist = 1.0 - corr
        return float(dist)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        return float(np.dot(a, b))

    def _smooth_box(self, old_sbox, new_box):
        new4 = self._box4(new_box)
        if new4 is None:
            return old_sbox
        if old_sbox is None:
            return new4
        ax = self.smooth_alpha
        bx1 = ax*old_sbox[0] + (1-ax)*new4[0]
        by1 = ax*old_sbox[1] + (1-ax)*new4[1]
        bx2 = ax*old_sbox[2] + (1-ax)*new4[2]
        by2 = ax*old_sbox[3] + (1-ax)*new4[3]
        return (bx1, by1, bx2, by2)

    ##########################################
    # LOST / CLEANUP
    ##########################################
    def _mark_lost(self, tid: int):
        tr = self.active.get(tid)
        if not tr:
            return
        self.lost[tid] = {
            "hist": tr.get("hist"),
            "embed": tr.get("embed"),           
            "ts_lost": time.time(),
        }
        del self.active[tid]

    def _cleanup_lost(self):
        now = time.time()
        to_delete = []
        for tid, info in self.lost.items():

            if now - info["ts_lost"] > self.reemerge_sec:
                to_delete.append(tid)
        for tid in to_delete:
            del self.lost[tid]

    ##########################################
    # MAIN UPDATE
    ##########################################
    def update(self, dets: List[Tuple[int,int,int,int]], frame):
        self.frame_count += 1
        now = time.time()

        # -------------------------------------------------
        # STEP 0: sanitize detections and precompute hists + embeds
        # -------------------------------------------------
        det_infos = []
        for raw_box in dets:
            box4 = self._box4(raw_box)
            if box4 is None:
                continue
            hist = self._hist_signature(frame, box4)
            crop_full = self._crop_full(frame, box4)
            embed = self._reid_embed(crop_full) if crop_full is not None else None

            det_infos.append({
                "box": box4,
                "hist": hist,
                "embed": embed,         
                "matched": False,
                "track_id": None,
            })

        num_dets = len(det_infos)
        if num_dets == 0 and len(self.active) == 0:
            return []

        unmatched_det_idxs = set(range(num_dets))

        # -------------------------------------------------
        # STEP 1: GLOBAL MATCH ACTIVE <-> DETECTIONS (IOU+HIST as before)
        # -------------------------------------------------
        pairs = []  # (score, tid, di, iou_val)
        for tid, tr in self.active.items():
            prev_box = tr["bbox"]
            prev_hist = tr.get("hist")

            for di, dinfo in enumerate(det_infos):
                new_box = dinfo["box"]
                new_hist = dinfo["hist"]

                iou_val = self._iou(prev_box, new_box)

                hist_sim = 0.0
                if prev_hist is not None and new_hist is not None:
                    hd = self._hist_distance(prev_hist, new_hist)
                    hist_sim = max(0.0, 1.0 - hd)

                score = (self.match_w_iou * iou_val) + (self.match_w_hist * hist_sim)
                pairs.append((score, tid, di, iou_val))

        pairs.sort(key=lambda x: x[0], reverse=True)

        used_tids = set()
        used_dis = set()
        for score, tid, di, iou_val in pairs:
            if tid in used_tids:
                continue
            if di in used_dis:
                continue
            if (iou_val >= self.iou_thresh) or (score >= self.match_min_score):
                used_tids.add(tid)
                used_dis.add(di)
                det_infos[di]["matched"] = True
                det_infos[di]["track_id"] = tid
                if di in unmatched_det_idxs:
                    unmatched_det_idxs.discard(di)

        # -------------------------------------------------
        # STEP 2: REVIVE FROM LOST (ReID first, then HIST fallback)
        # -------------------------------------------------
        for di in list(unmatched_det_idxs):
            d_hist = det_infos[di]["hist"]
            d_embed = det_infos[di]["embed"]
            box4 = det_infos[di]["box"]

            # ReID-based revival (within window)
            best_tid = None
            best_sim = -1.0
            if d_embed is not None:
                for lost_tid, linfo in self.lost.items():
                    if (now - linfo.get("ts_lost", now)) > self.reemerge_sec:
                        continue
                    sim = self._cosine_sim(d_embed, linfo.get("embed"))
                    if sim > best_sim:
                        best_sim = sim
                        best_tid = lost_tid
            if best_tid is not None and best_sim >= self.reid_match_thresh:
                det_infos[di]["matched"] = True
                det_infos[di]["track_id"] = best_tid

                revived_hist = self.lost[best_tid].get("hist")
                revived_embed = self.lost[best_tid].get("embed")

                self.active[best_tid] = {
                    "bbox": box4,
                    "sbox": self._smooth_box(None, box4),
                    "last_seen_frame": self.frame_count,
                    "hit_streak": 1,
                    "hist": revived_hist if revived_hist is not None else d_hist,
                    "embed": revived_embed if revived_embed is not None else d_embed,  # NEW
                }
                del self.lost[best_tid]
                unmatched_det_idxs.discard(di)
                continue 

            # Histogram fallback
            if d_hist is None:
                continue
            best_tid = None
            best_dist = 999.0
            for lost_tid, linfo in self.lost.items():
                if (now - linfo.get("ts_lost", now)) > self.reemerge_sec:
                    continue
                lost_hist = linfo.get("hist")
                if lost_hist is None:
                    continue
                dist = self._hist_distance(d_hist, lost_hist)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = lost_tid

            if best_tid is not None and best_dist <= self.hist_thresh:
                det_infos[di]["matched"] = True
                det_infos[di]["track_id"] = best_tid

                revived_hist = self.lost[best_tid].get("hist")

                self.active[best_tid] = {
                    "bbox": box4,
                    "sbox": self._smooth_box(None, box4),
                    "last_seen_frame": self.frame_count,
                    "hit_streak": 1,
                    "hist": revived_hist if revived_hist is not None else d_hist,
                    "embed": d_embed,  # NEW
                }

                del self.lost[best_tid]
                unmatched_det_idxs.discard(di)

        # -------------------------------------------------
        # STEP 3: NEW IDs for whatever is still unmatched
        # -------------------------------------------------
        for di in list(unmatched_det_idxs):
            box4 = det_infos[di]["box"]
            hist = det_infos[di]["hist"]
            embed = det_infos[di]["embed"]  # NEW

            new_tid = self.next_id
            self.next_id += 1

            self.active[new_tid] = {
                "bbox": box4,
                "sbox": self._smooth_box(None, box4),
                "last_seen_frame": self.frame_count,
                "hit_streak": 1,
                "hist": hist,
                "embed": embed,  # NEW
            }

            det_infos[di]["matched"] = True
            det_infos[di]["track_id"] = new_tid

        # -------------------------------------------------
        # STEP 4: UPDATE matched tracks' state
        # -------------------------------------------------
        for dinfo in det_infos:
            if not dinfo["matched"]:
                continue
            tid = dinfo["track_id"]
            box4 = dinfo["box"]
            hist = dinfo["hist"]
            embed = dinfo["embed"]  # NEW

            tr = self.active.get(tid)
            if tr is None:
                continue

            tr["bbox"] = box4
            tr["sbox"] = self._smooth_box(tr["sbox"], box4)
            tr["last_seen_frame"] = self.frame_count
            tr["hit_streak"] = tr.get("hit_streak", 0) + 1

            if tr.get("hist") is None:
                tr["hist"] = hist
            elif hist is not None:
                tr["hist"] = cv2.normalize(
                    (0.7 * tr["hist"] + 0.3 * hist).astype("float32"),
                    None
                )

            if embed is not None:
                tr["embed"] = embed

        # -------------------------------------------------
        # STEP 5: AGE OUT inactive actives → move to LOST
        # -------------------------------------------------
        to_delete_active = []
        for tid, tr in self.active.items():
            if (self.frame_count - tr["last_seen_frame"]) > self.max_age:
                self.lost[tid] = {
                    "hist": tr.get("hist"),
                    "embed": tr.get("embed"),   # NEW
                    "ts_lost": now
                }
                to_delete_active.append(tid)

        for tid in to_delete_active:
            if tid in self.active:
                del self.active[tid]

        # -------------------------------------------------
        # STEP 6: CLEANUP LOST that are too old
        # -------------------------------------------------
        self._cleanup_lost()

        # -------------------------------------------------
        # STEP 7: RETURN ONLY STABLE TRACKS
        # -------------------------------------------------
        out = []
        for tid, tr in self.active.items():
            if tr.get("hit_streak", 0) < 2:
                continue
            if tr.get("sbox") is None:
                continue
            x1, y1, x2, y2 = tr["sbox"]
            out.append((tid, (int(x1), int(y1), int(x2), int(y2))))

        return out
