
from tracking.roi_detector import RoiDetector
from tracking.optical_flow_tracker import OpticalFlowTracker

class ROIManager:
    def __init__(self):
        self.detector = RoiDetector()
        self.trackers = []
        self.counter = 0

    def update(self, frame):
        detected_rois = self.detector.detect(frame)

        if not self.trackers:
            self.trackers = []
            for roi in detected_rois:
                self.trackers.append(OpticalFlowTracker(frame, roi))

        alive = []
        for t in self.trackers:
            if t.update(frame):
                alive.append(t)
        self.trackers = alive
        return [t.roi for t in self.trackers]
