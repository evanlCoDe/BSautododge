
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
        
        self.key_sequence_steps = (
            # t, keys_to_release, keys_to_press
            (0.0, [], ["w", "a", "s", "d"]),
            (0.3, ["w", "s", "d"], ["a"]),
            (0.5, [], ["w", "a", "s", "d"])
            
        )
        


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
        # self._run_key_sequence()
        frame = self.cap.grab()
        
        rois = self.manager.update(frame)
        
        # Update player tracker
        player_position = self.player_tracker.update(frame)
        self.player_tracker.draw(frame)


        # Display detector ROIs in blue on the captured frame.
        for x, y, w, h in rois:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
 # A tracker predicts 150 frames ahead.  Convert that displacement to a
        # 500 ms horizon (about 15 frames at this window's 33 ms update rate).
        if player_position is not None:
            prediction_frames = 150
            horizon_frames = 500 / self.timer.interval()

            for threat in self.manager.trackers:
                if threat.center is None or threat.predicted_point is None:
                    continue

                # `predicted_point - center` is the threat's projected motion.
                speed_x = (threat.predicted_point[0] - threat.center[0]) / prediction_frames
                speed_y = (threat.predicted_point[1] - threat.center[1]) / prediction_frames
                future_position = (
                    threat.center[0] + speed_x * horizon_frames,
                    threat.center[1] + speed_y * horizon_frames,
                )

                # Account for both the player marker and the tracked threat's
                # bounding box when deciding whether their paths collide.
                _, _, threat_w, threat_h = threat.roi
                hit_radius = 30 + max(threat_w, threat_h) / 2
                # Check the entire predicted path, not only the point where
                # the threat will be after 500 ms.  This catches bullets that
                # pass through the player before reaching that endpoint.
                path_x = future_position[0] - threat.center[0]
                path_y = future_position[1] - threat.center[1]
                path_length_squared = path_x ** 2 + path_y ** 2

                if path_length_squared:
                    progress = (
                        (player_position[0] - threat.center[0]) * path_x
                        + (player_position[1] - threat.center[1]) * path_y
                    ) / path_length_squared
                    progress = max(0.0, min(1.0, progress))
                    closest_point = (
                        threat.center[0] + path_x * progress,
                        threat.center[1] + path_y * progress,
                    )
                else:
                    closest_point = threat.center

                closest_distance = (
                    (closest_point[0] - player_position[0]) ** 2
                    + (closest_point[1] - player_position[1]) ** 2
                ) ** 0.5

                if closest_distance <= hit_radius:
                    cv2.circle(frame, threat.center, 100,
                               (0, 0, 255), 2)
                    self._run_key_sequence()
                    print(f"Threat at {threat.center} predicted to hit player at {player_position} in 500 ms. Closest path distance: {closest_distance:.2f}, Hit radius: {hit_radius:.2f}")
                elif self.key_sequence_state < 1 or self.key_sequence_state >= len(self.key_sequence_steps):
                    self.key_sequence_started_at = time.monotonic()
                    self.key_sequence_state =0

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
        if self.manager.detector.binary_img is not None:
            binary_rgb = cv2.cvtColor(
                self.manager.detector.binary_img,
                cv2.COLOR_GRAY2RGB,
            )
            binary_h, binary_w, binary_ch = binary_rgb.shape
            binary_bytes_per_line = binary_ch * binary_w
            binary_qimg = QImage(
                binary_rgb.data,
                binary_w,
                binary_h,
                binary_bytes_per_line,
                QImage.Format_RGB888,
            )
           
            if not hasattr(self, "binary_window"):
                self.binary_window = QLabel()
                self.binary_window.setWindowTitle("Binary Image")
                self.binary_window.setAlignment(Qt.AlignCenter)
                self.binary_window.setScaledContents(True)
                self.binary_window.resize(binary_w, binary_h)
                self.binary_window.show()


            self.binary_window.setPixmap(QPixmap.fromImage(binary_qimg.copy()))

        self.statusBar().showMessage(f"Capture region: {self.cap.width}x{self.cap.height}  ROIs={len(rois)}  FPS~30")

    def _run_key_sequence(self):
        """Hold WASD after 10 seconds, then leave only D held after 3 more."""
        elapsed = time.monotonic() - self.key_sequence_started_at
        # print(f"Elapsed time: {elapsed:.2f} seconds, Key sequence state: {self.key_sequence_state}")
        # print(f"Key sequence steps: {self.key_sequence_steps}")
        
        if self.key_sequence_state < len(self.key_sequence_steps) and elapsed >= self.key_sequence_steps[self.key_sequence_state][0]:
            for key in self.key_sequence_steps[self.key_sequence_state][1]:
                self.keyboard.key_up(key)
            for key in self.key_sequence_steps[self.key_sequence_state][2]:
                self.keyboard.key_down(key)
            self.key_sequence_state += 1
            # print(f"Key sequence step {self.key_sequence_state} executed at {elapsed:.2f} seconds.")

            

    def closeEvent(self, event):
        self.keyboard.release_all()
        super().closeEvent(event)
       
