import sys, os, socket, shutil
from pathlib import Path
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QDate
from src.verifier import BehaviorVerifier

import config as cfg
from src.pipeline import SmokingVapingPipeline
from src.db import DB

def _tcp_ping(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False

def _vqa_host_port_from_url(url: str) -> tuple[str, int]:
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or "127.0.0.1"
        port = u.port or 80
        return host, port
    except Exception:
        return ("127.0.0.1", 8012)

def _open_folder(path: str):
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore
    elif sys.platform.startswith("darwin"):
        import subprocess
        subprocess.Popen(["open", str(p)])
    else:
        import subprocess
        subprocess.Popen(["xdg-open", str(p)])

def _get_latest_session(db: DB):
    rows = db._conn.execute(
        """
        SELECT session_id,
               session_label,
               MAX(ts_iso) AS latest_iso
        FROM violations
        GROUP BY session_id
        ORDER BY latest_iso DESC
        LIMIT 1
        """
    ).fetchall()
    if not rows:
        return None
    return (rows[0][0], rows[0][1])


class StatusPills(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10,10,10,10)
        lay.setSpacing(8)

        self.vqa = QtWidgets.QLabel("VQA SERVER: checking…")
        self.oll = QtWidgets.QLabel("OLLAMA: checking…")

        for w in (self.vqa, self.oll):
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            w.setFixedHeight(36)
            w.setStyleSheet(self._pill_style("#888"))
            lay.addWidget(w)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(3000)

    def _pill_style(self, color: str) -> str:
        return f"""
        QLabel {{
            background: {color};
            color: white;
            border-radius: 18px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}"""

    def _tick(self):
        host, port = _vqa_host_port_from_url(getattr(cfg, "VQA_HTTP_URL", "http://127.0.0.1:8012/vqa"))
        vqa_ok = _tcp_ping(host, port)
        self.vqa.setText(f"VQA SERVER: {'ONLINE' if vqa_ok else 'OFFLINE'}")
        self.vqa.setStyleSheet(self._pill_style("#47a447" if vqa_ok else "#a94442"))

        oll_ok = _tcp_ping("127.0.0.1", 11434)
        self.oll.setText(f"OLLAMA: {'ONLINE' if oll_ok else 'OFFLINE'}")
        self.oll.setStyleSheet(self._pill_style("#47a447" if oll_ok else "#a94442"))

class LiveFootagePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_v = QtWidgets.QVBoxLayout(self)
        main_v.setContentsMargins(10,10,10,10)
        main_v.setSpacing(10)

        title = QtWidgets.QLabel("LIVE FOOTAGE")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color:#ddd; background:#aaa; font-weight:700; "
            "border-radius:18px; padding:8px;"
        )
        main_v.addWidget(title)

        self.video_label = QtWidgets.QLabel("INPUT FOOTAGE")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#444; color:#eee; border-radius:24px; font-weight:700;"
        )
        self.video_label.setMinimumSize(640,360)
        main_v.addWidget(self.video_label, 4)

        rtitle = QtWidgets.QLabel("PICTURE OF VIOLATIONS")
        rtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rtitle.setStyleSheet(
            "color:#ddd; background:#666; font-weight:700; "
            "border-radius:18px; padding:8px;"
        )
        main_v.addWidget(rtitle)

        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setStyleSheet(
            "QListWidget{background:#5a5a5a;border-radius:16px;color:#eee;}"
            "QListWidget::item{padding:8px;}"
        )
        self.thumb_list.setIconSize(QtCore.QSize(160,120))
        main_v.addWidget(self.thumb_list, 2)
        self.thumb_list.clear()

    def clear_thumbs(self):
        self.thumb_list.clear()

    def update_frame(self, pixmap: QtGui.QPixmap):
        self.video_label.setPixmap(pixmap)

    def refresh_thumbs_from_db(self, db: DB, session_id: str | None):
        self.thumb_list.clear()
        if not session_id:
            return

        rows = db._conn.execute(
            """
            SELECT id,image_path,predicted,confirmed
            FROM violations
            WHERE session_id=?
            ORDER BY ts DESC
            """,
            (session_id,)
        ).fetchall()

        for vid, img_path, pred, conf in rows:
            label_txt = pred.upper() if pred else ""
            if conf:
                label_txt += f" [{conf.upper()}]"
            item = QtWidgets.QListWidgetItem(f"#{vid}  {label_txt}")
            if img_path and Path(img_path).exists():
                pix = QtGui.QPixmap(img_path).scaled(
                    160,120,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                item.setIcon(QtGui.QIcon(pix))
            self.thumb_list.addItem(item)

class ReviewPage(QtWidgets.QWidget):
    def __init__(self, db: DB, parent=None):
        super().__init__(parent)
        self.db = db

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10,10,10,10)
        v.setSpacing(10)

        title = QtWidgets.QLabel("REVIEW")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color:#ddd; background:#aaa; font-weight:700; "
            "border-radius:18px; padding:8px;"
        )
        v.addWidget(title)

        self.list = QtWidgets.QListWidget()
        self.list.setStyleSheet(
            "QListWidget{background:#5a5a5a;border-radius:16px;color:#eee;}"
            "QListWidget::item{padding:8px;}"
        )
        self.list.itemSelectionChanged.connect(self._load_selected)
        v.addWidget(self.list, 1)

        mid = QtWidgets.QHBoxLayout()
        self.imgs = [QtWidgets.QLabel() for _ in range(5)]
        for L in self.imgs:
            L.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            L.setStyleSheet("background:#444;border-radius:12px;color:#888;")
            L.setFixedSize(180,120)
            mid.addWidget(L)
        v.addLayout(mid)

        self.desc = QtWidgets.QTextEdit()
        self.desc.setReadOnly(True)
        self.desc.setStyleSheet("background:#3a3a3a;color:#ddd;border-radius:12px;padding:8px;")
        v.addWidget(self.desc)

        bbar = QtWidgets.QHBoxLayout()
        def bigbtn(text):
            btn = QtWidgets.QPushButton(text)
            btn.setFixedHeight(48)
            btn.setStyleSheet(
                "QPushButton{background:#8a8a8a;color:#fff;border-radius:20px;font-weight:700;}"
                "QPushButton:pressed{background:#777;}"
            )
            return btn

        self.b_smoke = bigbtn("1. SMOKING")
        self.b_vape  = bigbtn("2. VAPING")
        self.b_none  = bigbtn("3. NONE (DELETE)")
        self.b_undo  = bigbtn("UNDO")

        bbar.addWidget(self.b_smoke)
        bbar.addWidget(self.b_vape)
        bbar.addWidget(self.b_none)
        bbar.addWidget(self.b_undo)
        v.addLayout(bbar)

        self.b_smoke.clicked.connect(lambda: self._confirm_selected("smoking"))
        self.b_vape.clicked.connect(lambda: self._confirm_selected("vaping"))
        self.b_none.clicked.connect(lambda: self._confirm_selected("none"))
        self.b_undo.clicked.connect(lambda: self._confirm_selected(None))

        self.short_1 = QtGui.QShortcut(QtGui.QKeySequence("1"), self)
        self.short_1.activated.connect(lambda: self._confirm_selected("smoking"))
        self.short_2 = QtGui.QShortcut(QtGui.QKeySequence("2"), self)
        self.short_2.activated.connect(lambda: self._confirm_selected("vaping"))
        self.short_3 = QtGui.QShortcut(QtGui.QKeySequence("3"), self)
        self.short_3.activated.connect(lambda: self._confirm_selected("none"))
        self.short_u = QtGui.QShortcut(QtGui.QKeySequence("U"), self)
        self.short_u.activated.connect(lambda: self._confirm_selected(None))
        self.short_z = QtGui.QShortcut(QtGui.QKeySequence("Z"), self)
        self.short_z.activated.connect(lambda: self._confirm_selected(None))

    def refresh_from_db(self):
        self.list.clear()
        rows = self.db._conn.execute(
            """
            SELECT id, ts, predicted
            FROM violations
            WHERE confirmed IS NULL
            ORDER BY ts ASC
            """
        ).fetchall()

        for vid, ts, pred in rows:
            it = QtWidgets.QListWidgetItem(f"#{vid}  {pred.upper()}  {ts}")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, vid)
            self.list.addItem(it)

        if self.list.count() > 0:
            self.list.setCurrentRow(0)
            self._load_selected()
        else:
            for L in self.imgs:
                L.clear()
                L.setText("")
            self.desc.clear()

    def _load_selected(self):
        for L in self.imgs:
            L.clear()
            L.setText("")
        self.desc.clear()

        it = self.list.currentItem()
        if not it:
            return
        vid = it.data(QtCore.Qt.ItemDataRole.UserRole)
        row = self.db._conn.execute(
            "SELECT description, evidence_dir FROM violations WHERE id=?",
            (vid,)
        ).fetchone()
        if not row:
            return

        desc, edir = row[0], row[1]
        self.desc.setPlainText(desc or "")

        for i in range(5):
            p = Path(edir) / f"review_{i+1}.jpg"
            if p.exists():
                pix = QtGui.QPixmap(str(p)).scaled(
                    180,120,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self.imgs[i].setPixmap(pix)
            else:
                self.imgs[i].setText(f"review_{i+1}.jpg\nmissing")

    def _confirm_selected(self, label: str | None):
        it = self.list.currentItem()
        if not it:
            return

        vid = it.data(QtCore.Qt.ItemDataRole.UserRole)
        if vid is None:
            return

        if label == "none":
            row = self.db._conn.execute(
                "SELECT evidence_dir FROM violations WHERE id=?",
                (vid,)
            ).fetchone()
            if row and row[0]:
                edir = row[0]
                try:
                    shutil.rmtree(edir, ignore_errors=True)
                except Exception:
                    pass
            self.db.delete_violation(vid)
        elif label in ("smoking", "vaping"):
            self.db.set_confirmed(vid, label)
        else:
            self.db.set_confirmed(vid, None)

        self.refresh_from_db()

class RecordsPage(QtWidgets.QWidget):
    def __init__(self, db: DB, parent=None):
        super().__init__(parent)
        self.db = db

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10,10,10,10)
        v.setSpacing(10)

        title = QtWidgets.QLabel("RECORDS")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color:#ddd; background:#aaa; font-weight:700; "
            "border-radius:18px; padding:8px;"
        )
        v.addWidget(title)

        filters = QtWidgets.QHBoxLayout()

        lbl_from = QtWidgets.QLabel("DATE FROM")
        lbl_from.setStyleSheet("color:#ccc;")
        self.from_d = QtWidgets.QDateEdit()
        self.from_d.setDisplayFormat("dd-MM-yyyy")
        self.from_d.setCalendarPopup(True)

        lbl_to = QtWidgets.QLabel("DATE TO")
        lbl_to.setStyleSheet("color:#ccc;")
        self.to_d = QtWidgets.QDateEdit()
        self.to_d.setDisplayFormat("dd-MM-yyyy")
        self.to_d.setCalendarPopup(True)

        today = QDate.currentDate()
        yesterday = today.addDays(-1)
        self.from_d.setDate(yesterday)
        self.to_d.setDate(today)

        self.btn_search = QtWidgets.QPushButton("SEARCH")
        self.btn_search.setFixedHeight(36)
        self.btn_search.setStyleSheet(
            "QPushButton{background:#8a8a8a;color:#fff;border-radius:20px;font-weight:700;padding:8px 16px;}"
            "QPushButton:pressed{background:#777;}"
        )

        for w in (lbl_from, self.from_d, lbl_to, self.to_d, self.btn_search):
            filters.addWidget(w)
        v.addLayout(filters)

        self.table = QtWidgets.QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels([
            "ID","Date","Session","Track","Predicted","Confirmed","Source","Folder"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QTableWidget{background:#4a4a4a;color:#eee;border-radius:12px;}")
        v.addWidget(self.table, 1)

        self.table.doubleClicked.connect(self._open_selected_folder)
        self.btn_search.clicked.connect(self._run_search)

        self._run_search()

    def _run_search(self):
        fd = self.from_d.date()
        td = self.to_d.date()
        f = fd.toString("dd-MM-yyyy") if fd.isValid() else None
        t = td.toString("dd-MM-yyyy") if td.isValid() else None

        rows = self.db.list_violations(
            f if fd.isValid() else None,
            t if td.isValid() else None
        )

        self.table.setRowCount(0)
        for (vid, ts, sess_label, track, pred, conf, desc, src, img, edir) in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [
                str(vid),
                ts,
                sess_label,
                str(track),
                pred or "",
                (conf or ""),
                src or "",
                edir or ""
            ]
            for c, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(val)
                if c in (0,3):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)

    def _open_selected_folder(self):
        r = self.table.currentRow()
        if r < 0:
            return
        edir = self.table.item(r, 7).text()
        _open_folder(edir)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smoking/Vaping Detection UI")
        self.setMinimumSize(1280,768)
        self.setStyleSheet("QMainWindow{background:#2f2f2f;}")

        Path("./out").mkdir(parents=True, exist_ok=True)
        self.db = DB("./out/violations.db")

        self.pipe = SmokingVapingPipeline(cfg)
        self.pipe.on_result = self._on_pipeline_result
        self.track_info = {}  # tid -> {"class","confirmed"}

        self.cap = cv2.VideoCapture(cfg.RTSP_OR_FILE)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open source {cfg.RTSP_OR_FILE}")
        self.last_pixmap = None

        self.behavior_verifier = None
        try:
            self.behavior_verifier = BehaviorVerifier(cfg.BEHAVIOR_MODEL_PATH)
            self.behavior_verifier_thresh = float(getattr(cfg, "BEHAVIOR_MODEL_THRESH", 0.60))
        except Exception as e:
            self.behavior_verifier = None
        self.current_session = (self.pipe.session_id, self.pipe.session_label)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(10,10,10,10)
        outer.setSpacing(10)

        rail_wrap = QtWidgets.QWidget()
        rail_wrap.setStyleSheet("QWidget{background:#3a3a3a;border-radius:24px;}")
        rail_layout = QtWidgets.QVBoxLayout(rail_wrap)
        rail_layout.setContentsMargins(10,10,10,10)
        rail_layout.setSpacing(10)

        def rail_btn(text):
            b = QtWidgets.QPushButton(text)
            b.setFixedHeight(56)
            b.setStyleSheet(
                "QPushButton{background:#7a7a7a;color:#fff;border-radius:24px;font-weight:800;}"
                "QPushButton:pressed{background:#6a6a6a;}"
            )
            return b

        self.b_live    = rail_btn("LIVE FOOTAGE")
        self.b_review  = rail_btn("REVIEW")
        self.b_records = rail_btn("RECORDS")
        rail_layout.addWidget(self.b_live)
        rail_layout.addWidget(self.b_review)
        rail_layout.addWidget(self.b_records)

        self.pills = StatusPills()
        rail_layout.addWidget(self.pills)

        rail_layout.addStretch(1)
        self.b_exit = rail_btn("EXIT")
        self.b_exit.clicked.connect(self.close)
        rail_layout.addWidget(self.b_exit)

        rail_wrap.setFixedWidth(220)
        outer.addWidget(rail_wrap)

        self.stack = QtWidgets.QStackedWidget()
        self.page_live    = LiveFootagePage()
        self.page_review  = ReviewPage(self.db)
        self.page_records = RecordsPage(self.db)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_review)
        self.stack.addWidget(self.page_records)
        outer.addWidget(self.stack, 1)

        self.b_live.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_live))
        self.b_review.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_review))
        self.b_records.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_records))

        src_type = "RTSP" if str(cfg.RTSP_OR_FILE).lower().startswith(("rtsp://","rtsps://")) else "FILE"
        badge = QtWidgets.QLabel(f"Source: {src_type}")
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        badge.setStyleSheet("color:#888;")
        self.statusBar().addPermanentWidget(badge)

        self.page_live.clear_thumbs()

        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.timeout.connect(self._tick_frame)
        self.frame_timer.start(33)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._tick_refresh)
        self.refresh_timer.start(1000)

    def _on_pipeline_result(self, d: dict):
        tid = d.get("track_id")
        if tid is None:
            return
        cls_val      = d.get("class", "unknown")
        confirmed    = bool(d.get("confirmed", False))

        info = self.track_info.get(tid, {})
        info["class"]       = cls_val
        info["confirmed"]   = confirmed
        self.track_info[tid] = info

    def _choose_color_for_track(self, tid: int):
        info = self.track_info.get(tid, {})
        cls_txt     = str(info.get("class","")).lower()
        confirmed   = bool(info.get("confirmed", False))

        if confirmed and cls_txt == "smoking":
            return (0,0,255)
        if confirmed and cls_txt == "vaping":
            return (0,255,255)
        if confirmed and cls_txt == "none":
            return (0,255,0)
        return (128,128,128)

    def _draw_overlay(self, frame, tracks_in):
        safe_tracks = []
        for t in tracks_in:
            if not isinstance(t, (list, tuple)):
                continue
            if len(t) < 2:
                continue
            tid = t[0]
            bbox_any = t[1]
            if not isinstance(bbox_any, (list, tuple)):
                continue
            if len(bbox_any) < 4:
                continue
            try:
                x1 = int(bbox_any[0])
                y1 = int(bbox_any[1])
                x2 = int(bbox_any[2])
                y2 = int(bbox_any[3])
            except Exception:
                continue
            safe_tracks.append((tid, (x1, y1, x2, y2)))

        for (tid, (x1, y1, x2, y2)) in safe_tracks:
            h_img, w_img = frame.shape[:2]
            x1 = max(0, min(x1, w_img-1))
            x2 = max(0, min(x2, w_img-1))
            y1 = max(0, min(y1, h_img-1))
            y2 = max(0, min(y2, h_img-1))

            color_bgr = self._choose_color_for_track(tid)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color_bgr,
                2
            )

        return safe_tracks

    def _tick_frame(self):
        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return

            raw_tracks = self.pipe.process_frame(frame)

            norm_tracks = []
            for t in raw_tracks:
                if not isinstance(t, (list, tuple)):
                    continue
                if len(t) < 2:
                    continue
                tid = t[0]
                bbox_any = t[1]
                if not isinstance(bbox_any, (list, tuple)):
                    continue
                if len(bbox_any) < 4:
                    continue
                try:
                    x1 = int(bbox_any[0])
                    y1 = int(bbox_any[1])
                    x2 = int(bbox_any[2])
                    y2 = int(bbox_any[3])
                except Exception:
                    continue
                norm_tracks.append((tid, (x1, y1, x2, y2)))

            norm_tracks = self._draw_overlay(frame, norm_tracks)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(
                rgb.data,
                w,
                h,
                3 * w,
                QtGui.QImage.Format.Format_RGB888
            )
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.page_live.video_label.width() or w,
                self.page_live.video_label.height() or h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )

            self.last_pixmap = pix
            self.page_live.update_frame(pix)

        except Exception as e:
            import traceback
            print("[_tick_frame ERROR]", e)
            traceback.print_exc()

    def _tick_refresh(self):
        try:
            sid = None
            if self.current_session:
                sid = self.current_session[0]

            self.page_live.refresh_thumbs_from_db(self.db, sid)

            self.page_review.refresh_from_db()
        except Exception as e:
            print("[_tick_refresh ERROR]", e)


    def closeEvent(self, ev: QtGui.QCloseEvent):
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        try:
            self.db.close()
        except:
            pass
        super().closeEvent(ev)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
