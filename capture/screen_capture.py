
from mss import mss
import numpy as np, cv2

class ScreenCapture:
    def __init__(self):
        self.sct = mss()
        monitor = self.sct.monitors[1]
        self.left = monitor['left']
        self.top = monitor['top']
        self.max_width = monitor['width']
        self.max_height = monitor['height']
        self.width = min(600, self.max_width)
        self.height = min(800, self.max_height)
        self.monitor = {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }
        print(f"ScreenCapture initialized with monitor region: {self.monitor}")

    def set_region(self, width, height):
        self.width = max(100, min(width, self.max_width))
        self.height = max(100, min(height, self.max_height))
        self.monitor['width'] = self.width
        self.monitor['height'] = self.height
        print(f"ScreenCapture region updated to: {self.monitor}")

    def grab(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
