from fastapi import APIRouter, HTTPException
from app.services.s3_service import S3Service
from app.services.analysis_service import AnalysisService
from app.api.schemas import AnalyzeRequest, PresignedUrlRequest

router = APIRouter()

# Dependency Injection
s3_service = S3Service()
analysis_service = AnalysisService()

@router.post("/generate-upload-url")
async def generate_upload_url(payload: PresignedUrlRequest):
    """
    Get a secure URL to upload video directly to S3.
    """
    try:
        # payload.extension is strictly typed as "mp4", "mov", or "avi"
        result = s3_service.generate_presigned_post(payload.extension)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_video(payload: AnalyzeRequest):
    """
    Trigger backend to download from S3 and process.
    """
    try:
        # payload.mode is strictly typed as "traditional" or "deep_learning"
        result = analysis_service.process_video(payload.video_key, payload.mode)
        return {"status": "success", "data": result}
        
    except ValueError as e:
        # Catch known logic errors (like bad file or invalid S3 key)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected system errors
        raise HTTPException(status_code=500, detail=str(e))