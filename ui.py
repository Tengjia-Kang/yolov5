import os
import sys
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors, save_one_box
env_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX') or sys.prefix
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
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

class YOLOWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, model, source_path):
        super().__init__()
        self.model = model
        self.source_path = source_path
        self.running = True

    def run(self):
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((640, 640), s=stride)

        cap = cv2.VideoCapture(self.source_path)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            im0 = frame.copy()
            img = cv2.resize(frame, imgsz)  # resize 到模型输入大小
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB，再 → CHW
            img = np.ascontiguousarray(img)

            im = torch.from_numpy(img).to(self.model.device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)  # 加 batch 维度，变成 [1, 3, 640, 640]

            # 推理
            pred = self.model(im, augment=False, visualize=False)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], im0.shape).round()

                annotator = Annotator(im0.copy(), line_width=2, example=str(self.model.names))
                for *xyxy, conf, cls in reversed(pred):
                    cls_id = int(cls)
                    label = f'{self.model.names[cls_id]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(cls_id, True))
                rendered = annotator.result()
            else:
                rendered = im0  # 没检测到目标，显示原图

            self.result_ready.emit(rendered)
            self.msleep(30)  # 控制播放速度

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行人检测")
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


        # self.model = YOLO("yolov5s.pt")
        # device = ''
        # device = select_device(device)
        weights = "runs/train/exp12/weights/best.pt"
        self.model = DetectMultiBackend(weights)
        self.model.names = [str(i) for i in range(1)]
        self.model.names = ['person']

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
        if not file_path:
            return

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((640, 640), s=stride)

        dataset = LoadImages(file_path, img_size=imgsz, stride=stride, auto=pt)

        for path, im, im0, _, _ in dataset:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            pred = self.model(im, augment=False, visualize=False)[0]

            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
            if pred is None or len(pred) == 0:
                print("没有检测到目标")
                self.display_image(im0)  # 显示原图
                return

            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], im0.shape).round()

            annotator = Annotator(im0.copy(), line_width=2, example=str(self.model.names))
            for *xyxy, conf, cls in reversed(pred):
                cls_id = int(cls)
                if cls_id >= len(self.model.names):
                    print(f"类别索引超出范围: {cls_id}")
                    continue
                label = f'{self.model.names[cls_id]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(cls_id, True))

            rendered_img = annotator.result()
            self.display_image(rendered_img)
            break  # 只处理一张图像

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
