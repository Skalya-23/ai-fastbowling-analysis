from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path
import asyncio

from app.models import UploadResponse, JobStatusResponse, JobStatus
from app.processing import (
    process_video, 
    create_job, 
    get_job, 
    UPLOADS_DIR, 
    OUTPUTS_DIR,
    extract_frames,
    jobs,
    cleanup_old_files
)

app = FastAPI(title="Cricket Bowling Analysis API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frame thumbnails as static files
UPLOADS_DIR.mkdir(exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(UPLOADS_DIR)), name="frames")


# Background cleanup task
async def periodic_cleanup():
    """Run cleanup every 6 hours"""
    while True:
        await asyncio.sleep(6 * 3600)  # Wait 6 hours
        cleanup_old_files(max_age_hours=24)


@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    # Start periodic cleanup task
    asyncio.create_task(periodic_cleanup())
    # Run initial cleanup
    cleanup_old_files(max_age_hours=24)


@app.get("/")
async def root():
    return {
        "message": "Cricket Bowling Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /v1/video/upload",
            "start_processing": "POST /v1/video/{job_id}/process",
            "status": "GET /v1/video/{job_id}/status",
            "frames": "GET /v1/video/{job_id}/frames",
            "download_video": "GET /v1/video/{job_id}/download/video",
            "download_metrics": "GET /v1/video/{job_id}/download/metrics"
        }
    }


@app.post("/v1/video/upload")
async def upload_video(
    video: UploadFile = File(...),
    bowler_height_m: float = Form(...),
    handed: str = Form(...)
):
    """
    Step 1: Upload a cricket bowling video
    Returns job_id and extracts frames for release frame selection
    """
    
    # Validate handed parameter
    if handed not in ["right", "left"]:
        raise HTTPException(status_code=400, detail="handed must be 'right' or 'left'")
    
    # Validate file type
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(
            status_code=400, 
            detail="Only video files (.mp4, .avi, .mov) are supported"
        )
    
    # Create job first to get job_id
    job_id = create_job(video.filename)
    
    # Save uploaded file with job_id prefix to avoid conflicts
    video_filename = f"{job_id}_{video.filename}"
    video_path = UPLOADS_DIR / video_filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Update job with video info
    jobs[job_id]["video_path"] = str(video_path)
    jobs[job_id]["bowler_height_m"] = bowler_height_m
    jobs[job_id]["handed"] = handed
    jobs[job_id]["original_filename"] = video.filename
    
    # Extract frames for release frame selection
    try:
        total_frames, fps = extract_frames(video_path, job_id)
        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["fps"] = fps
        jobs[job_id]["status"] = JobStatus.PENDING
        jobs[job_id]["progress"] = "Frames extracted. Please select release frame."
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to extract frames: {str(e)}")
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Video uploaded and frames extracted. Please select the release frame.",
        "total_frames": total_frames,
        "fps": fps,
        "frames_url": f"/v1/video/{job_id}/frames"
    }


@app.get("/v1/video/{job_id}/frames")
async def get_frames_info(job_id: str):
    """
    Get information about extracted frames for a job
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    total_frames = job.get("total_frames", 0)
    fps = job.get("fps", 30)
    
    # Generate frame URLs
    frame_urls = []
    for i in range(total_frames):
        frame_urls.append({
            "frame_number": i,
            "url": f"/frames/{job_id}_frames/frame_{i:05d}.jpg",
            "time_seconds": round(i / fps, 3)
        })
    
    return {
        "job_id": job_id,
        "total_frames": total_frames,
        "fps": fps,
        "frames": frame_urls
    }


@app.post("/v1/video/{job_id}/process")
async def start_processing(
    job_id: str,
    release_frame: int = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Step 2: Start processing with selected release frame
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Job is already processing")
    
    if job["status"] == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is already completed")
    
    # Validate release frame
    total_frames = job.get("total_frames", 0)
    if release_frame < 0 or release_frame >= total_frames:
        raise HTTPException(
            status_code=400, 
            detail=f"release_frame must be between 0 and {total_frames - 1}"
        )
    
    # Get video processing parameters
    video_path = Path(job["video_path"])
    bowler_height_m = job["bowler_height_m"]
    handed = job["handed"]
    
    # Store release frame in job
    jobs[job_id]["release_frame"] = release_frame
    
    # Start processing in background
    background_tasks.add_task(
        process_video, 
        job_id, 
        video_path, 
        bowler_height_m, 
        handed,
        release_frame
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": f"Processing started with release frame {release_frame}",
        "release_frame": release_frame
    }


@app.get("/v1/video/{job_id}/status", response_model=JobStatusResponse)
async def get_video_status(job_id: str):
    """
    Get the status of a video processing job
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        error=job.get("error")
    )
    
    # Add download URLs if completed
    if job["status"] == JobStatus.COMPLETED:
        response.video_url = f"/v1/video/{job_id}/download/video"
        response.metrics_url = f"/v1/video/{job_id}/download/metrics"
    
    return response


@app.get("/v1/video/{job_id}/download/video")
async def download_annotated_video(job_id: str):
    """
    Download the annotated video
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video processing not complete")
    
    video_filename = job.get("video_filename")
    if not video_filename:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    video_path = OUTPUTS_DIR / video_filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on server")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_filename
    )


@app.get("/v1/video/{job_id}/download/metrics")
async def download_metrics(job_id: str):
    """
    Download or view the metrics JSON
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video processing not complete")
    
    # Return metrics directly as JSON
    return job.get("metrics", {})


@app.delete("/v1/video/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    """
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Delete associated files from uploads and outputs
    # TODO: Remove job from jobs dictionary
    
    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)