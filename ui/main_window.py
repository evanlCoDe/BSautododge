
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import time

from capture.screen_capture import ScreenCapture
from control.keyboard_events import KeyboardEvents
from tracking.player_tracker import PlayerTracker
from tracking.roi_manager import ROIManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Tracker V2")

        self.cap = ScreenCapture()
        self.manager = ROIManager()
        self.player_tracker = PlayerTracker()
        self.keyboard = KeyboardEvents()
        self.key_sequence_started_at = time.monotonic()
        self.key_sequence_state = 0

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, self.cap.max_width)
        self.width_spin.setValue(self.cap.width)
        self.width_spin.setSuffix(" px")
        self.width_spin.valueChanged.connect(self.update_region)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, self.cap.max_height)
        self.height_spin.setValue(self.cap.height)
        self.height_spin.setSuffix(" px")
        self.height_spin.valueChanged.connect(self.update_region)

        controls = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Width:"))
        controls_layout.addWidget(self.width_spin)
        controls_layout.addWidget(QLabel("Height:"))
        controls_layout.addWidget(self.height_spin)
        controls_layout.addStretch(1)
        controls.setLayout(controls_layout)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(controls)
        layout.addWidget(self.label)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.update_region()

    def update_region(self):
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.cap.set_region(width, height)
        self.statusBar().showMessage(f"Capture region: {width}x{height}")

    def tick(self):
        self._run_key_sequence()
        frame = self.cap.grab()
        
        rois = self.manager.update(frame)
        
        # Update player tracker
        player_position = self.player_tracker.update(frame)
        self.player_tracker.draw(frame)

        # ROI boxes disabled because PlayerTracker has its own display box
        # for x, y, w, h in rois:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

        # show another window with the diff image from RoiDetector
        # if self.manager.detector.diff_img is not None:
        #     diff_rgb = cv2.cvtColor(self.manager.detector.diff_img, cv2.COLOR_GRAY2RGB)
        #     h, w, ch = diff_rgb.shape
        #     bytes_per_line = ch * w
        #     qimg_diff = QImage(diff_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #     if not hasattr(self, 'diff_window'):
        #         self.diff_window = QLabel()
        #         self.diff_window.setWindowTitle("Difference Image")
        #         self.diff_window.setAlignment(Qt.AlignCenter)
        #         self.diff_window.setScaledContents(True)
        #         self.diff_window.resize(w, h)
        #         self.diff_window.show()
        #     self.diff_window.setPixmap(QPixmap.fromImage(qimg_diff))

        # show another window with the binary image from RoiDetector
        # if self.manager.detector.binary_img is not None:
        #     binary_rgb = cv2.cvtColor(self.manager.detector.binary_img, cv2.COLOR_GRAY2RGB)
        #     h, w, ch = binary_rgb.shape
        #     bytes_per_line = ch * w
        #     qimg_binary = QImage(binary_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #     if not hasattr(self, 'binary_window'):
        #         self.binary_window = QLabel()
        #         self.binary_window.setWindowTitle("Binary Image")
        #         self.binary_window.setAlignment(Qt.AlignCenter)
        #         self.binary_window.setScaledContents(True)
        #         self.binary_window.resize(w, h)
        #         self.binary_window.show()
        #     self.binary_window.setPixmap(QPixmap.fromImage(qimg_binary))

        self.statusBar().showMessage(f"Capture region: {self.cap.width}x{self.cap.height}  ROIs={len(rois)}  FPS~30")

    def _run_key_sequence(self):
        """Hold WASD after 10 seconds, then leave only D held after 3 more."""
        elapsed = time.monotonic() - self.key_sequence_started_at

        if self.key_sequence_state == 0 and elapsed >= 10:
            for key in ("w", "a", "s", "d"):
                self.keyboard.key_down(key)
            self.key_sequence_state = 1

        elif self.key_sequence_state == 1 and elapsed >= 13:
            for key in ("w", "a", "s"):
                self.keyboard.key_up(key)
            self.key_sequence_state = 2

    def closeEvent(self, event):
        self.keyboard.release_all()
        super().closeEvent(event)
       
