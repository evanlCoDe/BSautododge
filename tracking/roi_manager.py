
from tracking.roi_detector import RoiDetector
from tracking.optical_flow_tracker import OpticalFlowTracker

class ROIManager:
    def __init__(self):
        self.detector = RoiDetector()
        self.trackers = []
        self.counter = 0

    def update(self, frame , player_position=None):
        detected_rois = self.detector.detect(frame)

        if not self.trackers:
            self.trackers = []
            for roi in detected_rois:
                self.trackers.append(OpticalFlowTracker(frame, roi))

        alive = []

        for t in self.trackers:         
            if t.update(frame):                
                if player_position is not None:
                    # Check if the tracker is too close to the player
                    distance_to_player = ((t.center[0] - player_position[0]) ** 2 + (t.center[1] - player_position[1]) ** 2) ** 0.5
                    if distance_to_player > 50:  # Adjust this threshold as needed
                        alive.append(t)
                else:
                    alive.append(t)
                
        self.trackers = alive
        return [t.roi for t in self.trackers]
