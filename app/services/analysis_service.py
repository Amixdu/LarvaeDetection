import cv2
import os
import tempfile
from app.core.factory import DetectorFactory
from app.services.s3_service import S3Service

class AnalysisService:
    def __init__(self):
        self.s3_service = S3Service()

    def process_video(self, video_key: str, mode: str):
        # 1. Setup Temp File
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close() # Close handle so S3 client can write to it

        try:
            # 2. Download from S3
            print(f"Downloading {video_key}...")
            self.s3_service.download_file(video_key, temp_path)
            print("Download complete. Starting processing...")

            # 3. Instantiate Strategy
            detector = DetectorFactory.get_detector(mode)

            # 4. Run OpenCV Loop
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise ValueError("Could not open downloaded video file.")

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # The strategy handles all logic (stabilization, detection, tracking)
                detector.detect(frame, frame_idx)
                frame_idx += 1

            cap.release()

            # 5. Get Result
            final_count = detector.get_total_unique_count()
            
            return {
                "video_key": video_key,
                "mode": mode,
                "frames_processed": frame_idx,
                "larvae_count": final_count
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Deleted temp file: {temp_path}")