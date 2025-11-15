from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[str] = None
    metrics_url: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None


class VideoUploadParams(BaseModel):
    bowler_height_m: float
    handed: Literal["right", "left"]