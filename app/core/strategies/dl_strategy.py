from ..interfaces import BaseLarvaDetector

class DLStrategy(BaseLarvaDetector):
    def __init__(self):
        # Initialize YOLO and ByteTrack models
        self.detections = []

    def detect(self, frame):
        # Implement deep learning detection using YOLO + ByteTrack
        # Process frame and update detections
        # self.detections = ... (list of detected larvae)
        pass

    def get_detections(self):
        return self.detections