import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from .pipeline import SmokingVapingPipeline
from .utils import draw_tracks
import config as cfg


class ResultsTable(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__(0, 5)
        self.setHorizontalHeaderLabels(["ID", "Class", "p", "Snapshot", "Probs"])
        self.horizontalHeader().setStretchLastSection(True)

    def add_row(self, d):
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(str(d.get("track_id", "-"))))
        self.setItem(r, 1, QtWidgets.QTableWidgetItem(d.get("class", "-")))
        self.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{d.get('p', 0.0):.3f}"))
        self.setItem(r, 4, QtWidgets.QTableWidgetItem(str(d.get("probs", {}))))

        if d.get("img_path"):
            btn = QtWidgets.QPushButton("Open")
            btn.clicked.connect(
                lambda _, p=d["img_path"]: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))
            )
            self.setCellWidget(r, 3, btn)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, _cfg):
        super().__init__()
        self.setWindowTitle("Smoke/Vape Detection — VQA Calibrated")

        # --- UI elements
        self.label = QtWidgets.QLabel()
        self.table = ResultsTable()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label, 3)
        layout.addWidget(self.table, 2)

        # --- tracking data
        self.track_info = {}  # tid -> {"probs": {...}, "class": str, "p": float}

        # --- pipeline
        self.pipe = SmokingVapingPipeline(cfg)
        self.pipe.on_result = self.on_result

        # --- video source
        self.cap = cv2.VideoCapture(cfg.RTSP_OR_FILE)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.loop)
        self.timer.start(33)

    def on_result(self, d: dict):
        tid = d.get("track_id")
        if tid is not None:
            self.track_info[tid] = {
                "probs": d.get("probs"),
                "class": d.get("class"),
                "p": d.get("p", 0.0),
            }
        QtCore.QTimer.singleShot(0, lambda d=d: self.table.add_row(d))

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        tracks = self.pipe.process_frame(frame)
        draw_tracks(frame, tracks, info=self.track_info)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.width() or w,
            self.label.height() or h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.label.setPixmap(pix)
