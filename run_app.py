import sys
from PyQt6 import QtWidgets
import config as cfg
from src.app import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(cfg)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())