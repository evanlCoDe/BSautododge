import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self, frame, roi):
        self.roi = roi
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute motion map
        diff = cv2.absdiff(self.prev_gray, gray)
        mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

        # reduce noise
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # pick largest motion region
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            area = w * h

            # only update ROI if motion is meaningful
            if area > 10:
                self.roi = (x, y, w, h)

        # always update reference frame
        self.prev_gray = gray

        # always keep tracker alive (important for drawing green box)
        return True
