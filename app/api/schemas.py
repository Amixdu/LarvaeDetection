from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Use Literal to strictly enforce specific string values
class AnalyzeRequest(BaseModel):
    video_key: str = Field(..., min_length=5, description="S3 Key of the uploaded video")
    mode: Literal["traditional", "deep_learning"] = Field(
        default="traditional", 
        description="The analysis strategy to use"
    )

class PresignedUrlRequest(BaseModel):
    extension: Literal["mp4", "mov", "avi"] = Field(
        default="mp4", 
        description="Video file extension allowed"
    )