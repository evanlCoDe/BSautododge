import cv2
import numpy as np


class OpticalFlowTracker:
    """Track a moving object using a local ROI instead of full-frame diff.

    This is tuned for game projectiles: a bullet moves fast, can be surrounded by
    smoke and lighting effects, and the background may also move. We therefore
    avoid treating the whole screen as a motion mask and instead estimate the
    object motion inside a small, previous ROI with sparse optical flow.
    """

    def __init__(self, frame, roi):
        self.roi = tuple(roi)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.prev_hue = hsv[:, :, 0]

        self.center = self._center_from_roi(self.roi)
        self.prev_center = self.center
        self.velocity_history = []
        self.predicted_point = None
        self.frame_count = 0

    @staticmethod
    def _center_from_roi(roi):
        x, y, w, h = roi
        return (int(x + w / 2), int(y + h / 2))

    def _safe_roi(self, center, size):
        x, y = center
        w, h = size
        pad_x = max(20, int(w * 0.8))
        pad_y = max(20, int(h * 0.8))

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = max(x0 + 1, x + pad_x)
        y1 = max(y0 + 1, y + pad_y)
        return x0, y0, x1, y1

    def _filter_candidate(self, center, prev_center):
        if prev_center is None:
            return True

        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]
        speed = abs(dx) + abs(dy)
        print(f"{self.roi} ==> candidate:{center} speed:{speed:.2f}")

        # Ignore tiny jitter and reject implausible jumps caused by background
        # motion or wall-like texture changes.
        if 1< speed < 4:
            return False
        # if speed > 150:
        #     return False
        return True

    def _estimate_motion_from_flow(self, frame, hue):
        x, y, w, h = self.roi
        print(self.roi)
        pad = max(20, int(max(w, h) * 2))

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)

        prev_patch = self.prev_hue[y0:y1, x0:x1]
        cur_patch = hue[y0:y1, x0:x1]

        if prev_patch.size == 0 or cur_patch.size == 0:
            return None

        # Track only stable feature points within the local ROI. This reduces the
        # influence of background objects moving through the frame.
        prev_pts = cv2.goodFeaturesToTrack(
            prev_patch,
            maxCorners=100,
            qualityLevel=0.02,
            minDistance=7,
            blockSize=7,
        )

        if prev_pts is None or len(prev_pts) < 4:
            print(f"{self.roi} ==> pre_pts is None or len(prev_pts):{len(prev_pts) if prev_pts is not None else 0}")
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_patch,
            cur_patch,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_pts is None:
            print(f"{self.roi} ==> next_pts is None")
            return None

        good_vectors = []
        for prev_pt, next_pt, st in zip(prev_pts, next_pts, status.ravel()):
            if st == 1:
                prev_x, prev_y = prev_pt.ravel()
                next_x, next_y = next_pt.ravel()
                dx = next_x - prev_x
                dy = next_y - prev_y
                if abs(dx) + abs(dy) < 60:
                    good_vectors.append((dx, dy))

        if len(good_vectors) < 4:
            print(f"{self.roi} ==> len(good_vectors):{len(good_vectors)} < 4  return")
            return None

        avg_dx = float(np.mean([v[0] for v in good_vectors]))
        avg_dy = float(np.mean([v[1] for v in good_vectors]))

        candidate = (
            int(self.prev_center[0] + avg_dx),
            int(self.prev_center[1] + avg_dy),
        )

        if not self._filter_candidate(candidate, self.prev_center):
            print(f"{self.roi} ==> candidate:{candidate} rejected by _filter_candidate")
            return None

        return candidate

    def _estimate_motion_from_diff(self, frame, hue):
        x, y, w, h = self.roi
        pad = max(20, int(max(w, h) * 1.5))

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)

        prev_patch = self.prev_hue[y0:y1, x0:x1]
        cur_patch = hue[y0:y1, x0:x1]

        if prev_patch.size == 0 or cur_patch.size == 0:
            return None

        diff = cv2.absdiff(prev_patch, cur_patch)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 8:
            return None

        bx, by, bw, bh = cv2.boundingRect(contour)
        center = (int(x0 + bx + bw / 2), int(y0 + by + bh / 2))

        if not self._filter_candidate(center, self.prev_center):
            return None

        return center

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]

        # A small local ROI is much more stable than a whole-frame motion mask.
        if self.prev_center is None:
            self.prev_center = self.center

        candidate = self._estimate_motion_from_flow(frame, hue)
        if candidate is None:
            return 
            # candidate = self._estimate_motion_from_diff(frame, hue)

        if candidate is not None:
            dx = candidate[0] - self.prev_center[0]
            dy = candidate[1] - self.prev_center[1]

            # Smooth very noisy velocity estimates.
            self.velocity_history.append((dx, dy))
            if len(self.velocity_history) > 6:
                self.velocity_history.pop(0)

            avg_dx = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
            avg_dy = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)

            new_center = (
                int(candidate[0] * 0.6 + self.prev_center[0] * 0.4),
                int(candidate[1] * 0.6 + self.prev_center[1] * 0.4),
            )
            if abs(avg_dx) + abs(avg_dy) > 0:
                prediction_distance = 150
                self.predicted_point = (
                    int(new_center[0] + avg_dx * prediction_distance),
                    int(new_center[1] + avg_dy * prediction_distance),
                )

            # The tracked object is the center of the ROI; use the current center as
            # the object position, while allowing the ROI box to move with it.
            self.center = new_center
            self.roi = (
                int(new_center[0] - self.roi[2] / 2),
                int(new_center[1] - self.roi[3] / 2),
                self.roi[2],
                self.roi[3],
            )

            self.prev_center = self.center
            cv2.circle(frame, self.center, 3, (0, 255, 255), 2)

            if self.predicted_point is not None:
                cv2.line(frame, self.center, self.predicted_point, (0, 0, 255), 2)

        self.prev_hue = hue
        self.frame_count += 1
        return True
