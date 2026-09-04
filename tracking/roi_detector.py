
import cv2
import numpy as np


class RoiDetector:
    def __init__(self):
        self.prev=None
        self.binary_img = None
        self.diff_img = None

    def detect(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.prev is None:
            self.prev = hsv
            return []

        previous_hue = self.prev[:, :, 0].astype(np.int16)
        current_hue = hsv[:, :, 0].astype(np.int16)
        hue_delta = np.abs(previous_hue - current_hue)
        # Hue wraps at 180, so red near 0 and red near 179 are close colors.
        diff = np.minimum(hue_delta, 180 - hue_delta).astype(np.uint8)
        self.diff_img = diff

        # Hue is undefined for nearly gray pixels; ignore them to reduce noise.
        saturation = np.minimum(self.prev[:, :, 1], hsv[:, :, 1])
        colored_pixels = cv2.inRange(saturation, 40, 255)
        _, th = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        th = cv2.bitwise_and(th, colored_pixels)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev = hsv
        out = []
        binary_img = th.copy()
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 300:
                out.append((x, y, w, h))
                cv2.rectangle(binary_img, (x, y), (x + w, y + h), 255, 2)

        self.binary_img = binary_img

        return out
