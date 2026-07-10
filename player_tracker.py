import cv2
import numpy as np


class PlayerTracker:

    def __init__(self):
        self.last_position = None
        self.last_box = None
        self.lost_frames = 0


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


            ratio = w / max(h,1)


            # circle / semi-circle shape
            if (
                0.4 < ratio < 2.5
                and w > 10
                and h > 10
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

        circle = self.detect_green_circle(frame)


        health = self.detect_health_bar(
            frame,
            circle
        )


        if circle and health:


            x,y,w,h = circle


            player_x = x + w//2
            player_y = y + h//2


            self.last_position = (
                player_x,
                player_y
            )


            self.last_box = (
                player_x-30,
                player_y-30,
                60,
                60
            )


            self.lost_frames = 0


        else:

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