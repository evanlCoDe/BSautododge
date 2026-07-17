import cv2
import numpy as np
import time


class PlayerTracker:

    def __init__(
        self,
        stability_seconds=1.5,
        position_tolerance=6,
        template_match_threshold=0.55,
        template_search_padding=120,
    ):
        self.last_position = None
        self.last_box = None
        self.lost_frames = 0

        # A player must remain nearly still for this long before its green
        # pixels are used as a tracking template.
        self.stability_seconds = stability_seconds
        self.position_tolerance = position_tolerance
        self._stable_box = None
        self._stable_since = None

        self.green_template = None
        self.template_match_threshold = template_match_threshold
        self.template_search_padding = template_search_padding
        self.template_score = None


    @staticmethod
    def _green_mask(frame):
        """Return a binary image containing only player-green pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([35, 80, 50]),
            np.array([90, 255, 255]),
        )
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )


    def _reset_stability(self):
        self._stable_box = None
        self._stable_since = None


    def _box_is_stable(self, box):
        """Record a detection and report when it has been stable long enough."""
        now = time.monotonic()
        if self._stable_box is None:
            self._stable_box = box
            self._stable_since = now
            return False

        x, y, w, h = box
        old_x, old_y, old_w, old_h = self._stable_box
        is_similar = (
            abs(x - old_x) <= self.position_tolerance
            and abs(y - old_y) <= self.position_tolerance
            and abs(w - old_w) <= self.position_tolerance
            and abs(h - old_h) <= self.position_tolerance
        )
        if not is_similar:
            self._stable_box = box
            self._stable_since = now
            return False

        # Update the reference slowly so small detection jitter does not reset
        # the timer, while meaningful movement still does.
        self._stable_box = box
        return now - self._stable_since >= self.stability_seconds


    def _start_template_tracking(self, frame, box):
        """Capture the stable green region inside ``box`` as the template."""
        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return False

        template = self._green_mask(frame)[y1:y2, x1:x2]
        green_pixels = cv2.countNonZero(template)
        if green_pixels < 50 or green_pixels == template.size:
            return False

        self.green_template = template
        self.template_score = None
        return True


    def _track_template(self, frame):
        """Track the stored green template near the last player box."""
        if self.green_template is None or self.last_box is None:
            return False

        x, y, w, h = self.last_box
        template_h, template_w = self.green_template.shape
        frame_h, frame_w = frame.shape[:2]
        padding = self.template_search_padding
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(frame_w, x + w + padding)
        bottom = min(frame_h, y + h + padding)

        search_mask = self._green_mask(frame)[top:bottom, left:right]
        if search_mask.shape[0] < template_h or search_mask.shape[1] < template_w:
            return False

        result = cv2.matchTemplate(
            search_mask,
            self.green_template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, location = cv2.minMaxLoc(result)
        self.template_score = score
        if score < self.template_match_threshold:
            return False

        new_x = left + location[0]
        new_y = top + location[1]
        self.last_box = (new_x, new_y, template_w, template_h)
        self.last_position = (
            new_x + template_w // 2,
            new_y + template_h // 2,
        )
        self.lost_frames = 0
        return True


    def detect_green_circle(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # green circle detection
        mask = cv2.inRange(
            hsv,
            np.array([35, 80, 50]),
            np.array([90, 255, 255])
        )


        kernel = np.ones(
            (3,3),
            np.uint8
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        best = None
        best_area = 0


        for c in contours:

            area = cv2.contourArea(c)


            if area < 50:
                continue


            x,y,w,h = cv2.boundingRect(c)
            green_ratio = cv2.countNonZero(mask[y:y+h, x:x+w]) / (w*h)
            print(f"green_ratio: {green_ratio:.2f}, area: {area}, w: {w}, h: {h}")

            ratio = w / max(h,1)


            # circle / semi-circle shape
            if (
                0.4 < ratio < 2.5
                and w > 10
                and h > 10
                and green_ratio < 0.5
            ):

                if area > best_area:

                    best_area = area

                    best = (
                        x,
                        y,
                        w,
                        h
                    )


        return best



    def detect_health_bar(self, frame, circle):

        if circle is None:
            return None


        cx,cy,cw,ch = circle


        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        green = cv2.inRange(
            hsv,
            np.array([35,80,80]),
            np.array([90,255,255])
        )


        yellow = cv2.inRange(
            hsv,
            np.array([15,80,80]),
            np.array([35,255,255])
        )


        red1 = cv2.inRange(
            hsv,
            np.array([0,80,80]),
            np.array([10,255,255])
        )


        red2 = cv2.inRange(
            hsv,
            np.array([170,80,80]),
            np.array([180,255,255])
        )


        mask = (
            green |
            yellow |
            red1 |
            red2
        )


        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        best = None


        for c in contours:

            x,y,w,h = cv2.boundingRect(c)


            ratio = w / max(h,1)


            # only search above player circle
            if (
                y < cy
                and abs((x+w//2)-(cx+cw//2)) < 40
                and w > 15
                and h < 15
                and ratio > 2
            ):

                best = (
                    x,
                    y,
                    w,
                    h
                )


        return best



    def update(self, frame):

        # After the initial detection has been stable for long enough, template
        # matching is faster and does not depend on the health bar remaining
        # visible every frame.  If it loses the target, fall back to detection.
        if self.green_template is not None:
            if self._track_template(frame):
                return self.last_position
            self.green_template = None
            self.template_score = None
            self._reset_stability()

        circle = self.detect_green_circle(frame)


        health = self.detect_health_bar(
            frame,
            circle
        )


        if circle and health:


            x,y,w,h = circle


            player_x = x + w//2
            player_y = y + h//2


            detected_box = (
                player_x-30,
                player_y-30,
                60,
                60
            )

            self.last_position = (
                player_x,
                player_y
            )


            self.last_box = detected_box


            self.lost_frames = 0

            if self._box_is_stable(detected_box):
                if self._start_template_tracking(frame, detected_box):
                    self._reset_stability()


        else:

            # Continuous detection is required before a template is captured.
            self._reset_stability()

            self.lost_frames += 1


            if self.lost_frames > 30:

                self.last_position = None
                self.last_box = None



        return self.last_position



    def draw(self, frame):

        if self.last_box:

            x,y,w,h = self.last_box


            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,255),
                2
            )


            cv2.circle(
                frame,
                self.last_position,
                5,
                (0,255,255),
                -1
            )
