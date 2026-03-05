from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # REQUIRED FIELDS
    # If these are missing from .env or system env vars, the app will NOT start.
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS Access Key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS Secret Access Key")
    AWS_S3_BUCKET_NAME: str = Field(..., description="S3 Bucket Name for video storage")

    # OPTIONAL FIELDS WITH DEFAULTS
    AWS_REGION: str = Field("ap-southeast-2", description="AWS Region")
    
    # APP CONFIG
    PROJECT_NAME: str = "Larvae Detection API"
    API_V1_STR: str = "/api/v1"

    # Pydantic Settings Config
    # This tells Pydantic to read from a .env file if present
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore extra env vars (prevents errors on some systems)
    }

# Instantiate settings once
settings = Settings()