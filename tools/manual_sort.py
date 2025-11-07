import sys, shutil, argparse, cv2
from pathlib import Path
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

CLASSES = {
    "1": "smoking",
    "2": "vaping",
    "3": "none",
}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def sample_video_frame_cv2(path: Path, which: str = "middle") -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return None
    idx = 0
    if which == "first":
        idx = 0
    elif which == "middle":
        idx = n // 2
    elif which == "last":
        idx = max(0, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None and frame.size > 0:
        return frame
    return None

def get_preview_frame(path: Path) -> np.ndarray | None:
    if is_image(path):
        img = cv2.imread(str(path))
        return img
    if is_video(path):
        fr = sample_video_frame_cv2(path, "middle")
        if fr is None:
            # fallback: try first frame
            fr = sample_video_frame_cv2(path, "first")
        return fr
    # unknown -> try as image
    return cv2.imread(str(path))

def bgr_to_qpixmap(img_bgr: np.ndarray, max_w=1280, max_h=720) -> QtGui.QPixmap:
    if img_bgr is None or img_bgr.size == 0:
        # make a blank pixmap with a notice
        pm = QtGui.QPixmap(max_w, max_h)
        pm.fill(QtGui.QColor("black"))
        painter = QtGui.QPainter(pm)
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(QtGui.QFont("Arial", 18))
        painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Unable to decode preview")
        painter.end()
        return pm

    h, w = img_bgr.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w2, h2, 3 * w2, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)

def safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = dst
    k = 1
    while out.exists():
        out = dst.with_name(f"{dst.stem} ({k}){dst.suffix}")
        k += 1
    shutil.move(str(src), str(out))
    return out

class Sorter(QtWidgets.QWidget):
    def __init__(self, src_dir: Path):
        super().__init__()
        self.setWindowTitle("Manual Sorter — 1: smoking, 2: vaping, 3: none")

        # Resolve destination: project parent / dataset_sorted
        proj_dir = Path(__file__).resolve().parents[1]  # current folder's parent
        self.dst_root = proj_dir / "dataset_sorted"
        self.dst_root.mkdir(parents=True, exist_ok=True)
        for c in CLASSES.values():
            (self.dst_root / c).mkdir(parents=True, exist_ok=True)

        self.files = self._gather_files(src_dir)
        self.idx = 0

        # --- UI
        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(640, 360)

        self.path_label = QtWidgets.QLabel("—")
        self.path_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Buttons with shortcuts
        btn_smoke = QtWidgets.QPushButton("1  —  Smoking")
        btn_vape  = QtWidgets.QPushButton("2  —  Vaping")
        btn_none  = QtWidgets.QPushButton("3  —  None")
        btn_skip  = QtWidgets.QPushButton("Skip (S)")
        btn_back  = QtWidgets.QPushButton("Undo Last (B)")

        btn_smoke.clicked.connect(lambda: self.assign("1"))
        btn_vape.clicked.connect(lambda: self.assign("2"))
        btn_none.clicked.connect(lambda: self.assign("3"))
        btn_skip.clicked.connect(self.skip)
        btn_back.clicked.connect(self.undo_last)

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self.img_label, 0, 0, 1, 5)
        grid.addWidget(self.path_label, 1, 0, 1, 5)
        grid.addWidget(btn_smoke, 2, 0)
        grid.addWidget(btn_vape,  2, 1)
        grid.addWidget(btn_none,  2, 2)
        grid.addWidget(btn_skip,  2, 3)
        grid.addWidget(btn_back,  2, 4)

        # history for undo
        self.history = []  # list of (src_path_before_move, dst_path_after_move)

        self._update_view()

        # enable key events
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _gather_files(self, src_dir: Path):
        exts = IMAGE_EXTS | VIDEO_EXTS
        files = [p for p in src_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
        files.sort()
        return files

    def _update_view(self):
        if self.idx >= len(self.files):
            pm = QtGui.QPixmap(800, 400)
            pm.fill(QtGui.QColor("black"))
            painter = QtGui.QPainter(pm)
            painter.setPen(QtGui.QColor("white"))
            painter.setFont(QtGui.QFont("Arial", 24))
            painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "All done! 🎉")
            painter.end()
            self.img_label.setPixmap(pm)
            self.path_label.setText("No more files.")
            return

        p = self.files[self.idx]
        frame = get_preview_frame(p)
        pix = bgr_to_qpixmap(frame, max_w=1280, max_h=720)
        self.img_label.setPixmap(pix)
        self.path_label.setText(str(p))

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        k = e.key()
        if k == QtCore.Qt.Key.Key_1:
            self.assign("1")
        elif k == QtCore.Qt.Key.Key_2:
            self.assign("2")
        elif k == QtCore.Qt.Key.Key_3:
            self.assign("3")
        elif k in (QtCore.Qt.Key.Key_S, ):
            self.skip()
        elif k in (QtCore.Qt.Key.Key_B, ):
            self.undo_last()
        else:
            super().keyPressEvent(e)

    def assign(self, key: str):
        if self.idx >= len(self.files):
            return
        label = CLASSES.get(key)
        if not label:
            return
        src = self.files[self.idx]
        dst = self.dst_root / label / src.name
        try:
            new_path = safe_move(src, dst)
            self.history.append((src, new_path))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Move failed", f"{e}")
            return
        self.idx += 1
        self._update_view()

    def skip(self):
        if self.idx < len(self.files):
            self.idx += 1
            self._update_view()

    def undo_last(self):
        if not self.history:
            return
        orig_src, moved_dst = self.history.pop()
        try:
            # Move it back near its original folder (same parent as original src)
            back_dst = orig_src
            back_dst.parent.mkdir(parents=True, exist_ok=True)
            safe_move(moved_dst, back_dst)
            # Insert it back in the list right after current index for re-labeling
            self.files.insert(self.idx, back_dst)
            self._update_view()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Undo failed", f"{e}")

def pick_source_dir_dialog() -> Path | None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dlg = QtWidgets.QFileDialog()
    dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
    if dlg.exec():
        sel = dlg.selectedFiles()
        if sel:
            return Path(sel[0])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="Folder containing images/videos to sort")
    args = parser.parse_args()

    src_dir = Path(args.src) if args.src else None
    if not src_dir:
        # ask the user
        src_dir = pick_source_dir_dialog()
        if not src_dir:
            print("No source folder selected. Exiting.")
            return
    if not src_dir.exists():
        print(f"Source folder not found: {src_dir}")
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Sorter(src_dir)
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
