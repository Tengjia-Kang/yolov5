import numpy as np
from utils.torch_utils import select_device, smart_inference_mode
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import torch
# 获取当前环境的根路径（兼容 Conda 和 venv/venv）
env_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX') or sys.prefix

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)

# 构造 Qt 插件路径
qt_plugins_path = os.path.join(
    env_path,
    'lib',
    'python3.9',
    'site-packages',
    'PyQt5',
    'Qt5',
    'plugins'
)

# 强制设置环境变量
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import os
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

class YOLOWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, model, source_path):
        super().__init__()
        self.model = model
        self.source_path = source_path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source_path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            rendered_frame = results[0].plot()
            self.result_ready.emit(rendered_frame)
            cv2.waitKey(1)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 目标检测")
        self.setGeometry(300, 100, 1100, 850)
        self.setWindowIcon(QIcon("icon.png"))

        self.label = QLabel("请加载图像或视频文件", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(960, 720)
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                background-color: #f9f9f9;
                font-size: 20px;
                color: #555;
            }
        """)

        self.btn_load_image = QPushButton("加载图片")
        self.btn_load_video = QPushButton("加载视频")
        self.set_button_style(self.btn_load_image)
        self.set_button_style(self.btn_load_video)

        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video.clicked.connect(self.load_video)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_load_image)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.btn_load_video)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None
        self.model = YOLO("yolov5s.pt")
        # self.model = YOLO("runs/train/exp12/weights/best.pt")

    def set_button_style(self, button):
        button.setFixedSize(150, 40)
        button.setFont(QFont("微软雅黑", 11))
        button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6ec1e4, stop:1 #2980b9
                );
                color: white;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
        """)

    def load_image(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            results = self.model(img)
            rendered_img = results[0].plot()
            self.display_image(rendered_img)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            if self.worker:
                self.worker.stop()
            self.worker = YOLOWorker(self.model, file_path)
            self.worker.result_ready.connect(self.display_image)
            self.worker.start()

    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
