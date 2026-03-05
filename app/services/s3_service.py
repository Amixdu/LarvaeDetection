import boto3
from botocore.exceptions import ClientError
from app.config import settings
import uuid

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_S3_BUCKET_NAME

    def generate_presigned_post(self, extension: str = "mp4"):
        """Generates a secure URL for the frontend to upload directly to S3."""
        file_uuid = str(uuid.uuid4())
        object_name = f"uploads/{file_uuid}.{extension}"

        try:
            response = self.s3_client.generate_presigned_post(
                self.bucket,
                object_name,
                Fields={"acl": "private", "Content-Type": f"video/{extension}"},
                Conditions=[
                    {"acl": "private"},
                    {"Content-Type": f"video/{extension}"},
                    ["content-length-range", 1048576, 52428800]  # 1MB - 50MB
                ],
                ExpiresIn=300
            )
            return {"url": response, "video_key": object_name}
        except ClientError as e:
            raise Exception(f"AWS Error: {str(e)}")

    def download_file(self, object_key: str, dest_path: str):
        """Downloads a file from S3 to a local path."""
        try:
            self.s3_client.download_file(self.bucket, object_key, dest_path)
        except ClientError as e:
            raise Exception(f"Failed to download video: {str(e)}")