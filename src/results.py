import cv2, time, os
from pathlib import Path
from datetime import datetime
from .db import DB

def _crop_bbox(frame_bgr, bbox):

    if frame_bgr is None or bbox is None:
        return None

    try:
        x1, y1, x2, y2 = bbox
        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
    except Exception:
        return None

    H, W = frame_bgr.shape[:2]

    # clamp to frame
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2].copy()
    if crop is None or crop.size == 0:
        return None
    return crop


class ResultsWriter:

    def __init__(self,
                 db_path: str,
                 evidence_root: str,
                 session_id: str,
                 session_label: str,
                 source_type: str):
        # remember session context
        self.session_id    = session_id
        self.session_label = session_label
        self.source_type   = source_type

        # prepare evidence root like ./out/evidence/<session_label>/
        self.evidence_root = Path(evidence_root)
        (self.evidence_root / self.session_label).mkdir(parents=True, exist_ok=True)

        # open DB connection (lives in this thread)
        self.db = DB(db_path)

    def _mk_violation_dir(self, violation_id: int) -> Path:

        vdir = self.evidence_root / self.session_label / str(violation_id)
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir

    def _update_paths_in_db(self, violation_id: int, image_path: str, evidence_dir: str):

        self.db._conn.execute(
            """
            UPDATE violations
            SET image_path=?, evidence_dir=?
            WHERE id=?
            """,
            (image_path, evidence_dir, violation_id)
        )
        self.db._conn.commit()

    def save_main(self,
                  full_frame_bgr,
                  bbox,
                  track_id: int,
                  predicted_label: str,
                  description: str):

        # 1. crop suspect region
        crop_img = _crop_bbox(full_frame_bgr, bbox)
        if crop_img is None:
            crop_img = full_frame_bgr.copy()

        now = datetime.now()
        violation_id = self.db.insert_violation(
            ts_dt=now,
            session_id=self.session_id,
            session_label=self.session_label,
            track_id=track_id,
            predicted=predicted_label,
            description=description or "",
            source_type=self.source_type,
            image_path="",     # we fill after saving main.jpg
            evidence_dir=""    # we fill after folder creation
        )

        # 3. make folder for this violation
        vdir = self._mk_violation_dir(violation_id)

        # 4. save cropped main.jpg
        main_path = vdir / "main.jpg"
        cv2.imwrite(str(main_path), crop_img)

        # 5. update DB with real paths
        self._update_paths_in_db(
            violation_id,
            str(main_path),
            str(vdir)
        )

        # 6. return info for caller (pipeline uses this to push to UI)
        return {
            "violation_id": violation_id,
            "image_path": str(main_path),
            "evidence_dir": str(vdir),
        }

    def save_review_image(self,
                          violation_id: int,
                          idx: int,
                          full_frame_bgr,
                          bbox):

        vdir = self._mk_violation_dir(violation_id)

        crop_img = _crop_bbox(full_frame_bgr, bbox)
        if crop_img is None:
            crop_img = full_frame_bgr.copy()

        out_path = vdir / f"review_{idx}.jpg"
        cv2.imwrite(str(out_path), crop_img)

    def close(self):

        try:
            self.db.close()
        except:
            pass
