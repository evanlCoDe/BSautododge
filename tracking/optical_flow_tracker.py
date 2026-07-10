import cv2
import numpy as np


class OpticalFlowTracker:

    def __init__(self, frame, roi):
        self.roi = roi

        self.prev_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.prev_center = None

        # store recent movement directions
        self.velocity_history = []

        # smooth prediction endpoint
        self.predicted_point = None


    def update(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # detect movement
        diff = cv2.absdiff(
            self.prev_gray,
            gray
        )


        _, mask = cv2.threshold(
            diff,
            25,
            255,
            cv2.THRESH_BINARY
        )


        mask = cv2.medianBlur(
            mask,
            5
        )


        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        if contours:

            contour = max(
                contours,
                key=cv2.contourArea
            )


            x, y, w, h = cv2.boundingRect(
                contour
            )


            area = w * h


            if area > 10:

                self.roi = (
                    x,
                    y,
                    w,
                    h
                )


                center = (
                    x + w // 2,
                    y + h // 2
                )


                if self.prev_center is not None:

                    dx = center[0] - self.prev_center[0]
                    dy = center[1] - self.prev_center[1]


                    # ignore detection noise
                    if abs(dx) + abs(dy) > 4:

                        # save velocity
                        self.velocity_history.append(
                            (dx, dy)
                        )


                        # only keep recent movement
                        if len(self.velocity_history) > 5:
                            self.velocity_history.pop(0)


                        # average velocity
                        avg_dx = sum(
                            v[0] for v in self.velocity_history
                        ) / len(self.velocity_history)


                        avg_dy = sum(
                            v[1] for v in self.velocity_history
                        ) / len(self.velocity_history)


                        prediction_distance = 150


                        new_prediction = (
                            int(center[0] + avg_dx * prediction_distance),
                            int(center[1] + avg_dy * prediction_distance)
                        )


                        # smooth the line endpoint
                        if self.predicted_point is None:
                            self.predicted_point = new_prediction

                        else:
                            self.predicted_point = (
                                int(
                                    self.predicted_point[0] * 0.8 +
                                    new_prediction[0] * 0.2
                                ),

                                int(
                                    self.predicted_point[1] * 0.8 +
                                    new_prediction[1] * 0.2
                                )
                            )


                        # draw stable red prediction line
                        cv2.line(
                            frame,
                            center,
                            self.predicted_point,
                            (0, 0, 255),
                            2
                        )


                self.prev_center = center


        self.prev_gray = gray


        return True
