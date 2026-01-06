import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.factory import DetectorFactory
import tempfile
import os

router = APIRouter()

@router.post("/analyze_video")
async def analyze_video(
    mode: str = Form(..., description="Mode: 'traditional' or 'deep_learning'"),
    file: UploadFile = File(...)
):
    # Validate File
    if not file.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload MP4/MOV/AVI.")

    # Instantiate the correct Strategy
    try:
        detector = DetectorFactory.get_detector(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save Upload to Temp File (OpenCV needs a file path)
    # Note: For production, stream chunks instead of saving full file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # Process Video Loop
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run the Strategy
            detector.detect(frame, frame_idx)
            frame_idx += 1

        cap.release()

        # Get Final Result
        final_count = detector.get_total_unique_count()

        return {
            "filename": file.filename,
            "mode_used": mode,
            "total_frames_processed": frame_idx,
            "final_larvae_count": final_count,
            "status": "success"
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)