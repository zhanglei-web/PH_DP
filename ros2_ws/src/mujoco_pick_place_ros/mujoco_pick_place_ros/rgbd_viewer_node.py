from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def _depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    normalized = np.clip((depth_m - 0.30) / (2.0 - 0.30), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    color = (255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)
    color[depth_mm == 0] = 0
    return color


class CompressedRgbdSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_viewer")
        self.color: np.ndarray | None = None
        self.depth_color: np.ndarray | None = None
        self.create_subscription(
            CompressedImage,
            "camera/front/color/image_raw/compressed",
            self._on_color,
            10,
        )
        self.create_subscription(
            CompressedImage,
            "camera/front/depth/image_raw/compressedDepth",
            self._on_depth,
            10,
        )

    def _on_color(self, message: CompressedImage) -> None:
        with Image.open(BytesIO(bytes(message.data))) as image:
            self.color = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def _on_depth(self, message: CompressedImage) -> None:
        payload = bytes(message.data)
        if len(payload) <= 12:
            return
        with Image.open(BytesIO(payload[12:])) as image:
            depth_mm = np.asarray(image, dtype=np.uint16).copy()
        self.depth_color = _depth_colormap(depth_mm)


class RgbdWindow(QMainWindow):
    def __init__(self, node: CompressedRgbdSubscriber) -> None:
        super().__init__()
        self.node = node
        self.setWindowTitle("MuJoCo Front RGB-D Camera")
        self.resize(1320, 560)

        color_panel, self.color_image = self._panel("Color 640 x 480")
        depth_panel, self.depth_image = self._panel("Depth 640 x 480")
        layout = QHBoxLayout()
        layout.addWidget(color_panel)
        layout.addWidget(depth_panel)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(10)

    @staticmethod
    def _panel(title: str) -> tuple[QWidget, QLabel]:
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(640, 480)
        image_label.setScaledContents(True)
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(image_label)
        panel = QWidget()
        panel.setLayout(layout)
        return panel, image_label

    @staticmethod
    def _pixmap(array: np.ndarray) -> QPixmap:
        height, width, _ = array.shape
        image = QImage(
            array.data, width, height, width * 3, QImage.Format_RGB888
        ).copy()
        return QPixmap.fromImage(image)

    def _update(self) -> None:
        if not rclpy.ok():
            self.timer.stop()
            QApplication.instance().quit()
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except KeyboardInterrupt:
            self.timer.stop()
            QApplication.instance().quit()
            return
        if self.node.color is not None:
            self.color_image.setPixmap(self._pixmap(self.node.color))
            self.node.color = None
        if self.node.depth_color is not None:
            self.depth_image.setPixmap(self._pixmap(self.node.depth_color))
            self.node.depth_color = None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CompressedRgbdSubscriber()
    app = QApplication([])
    window = RgbdWindow(node)
    window.show()

    cleaned_up = False

    def cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    app.aboutToQuit.connect(cleanup)
    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
