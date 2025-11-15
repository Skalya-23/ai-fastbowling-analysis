import subprocess
import os
import uuid
from pathlib import Path
from typing import Dict
import json
import asyncio
import logging
from app.models import JobStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store job statuses in memory (use Redis/DB for production)
jobs: Dict[str, dict] = {}

# Paths
BASE_DIR = Path(__file__).parent.parent  # cricket-analysis-api/
PROJECT_ROOT = BASE_DIR.parent  # Cricket-YOLO-Pose/
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Use main project outputs folder
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Log paths on startup
logger.info(f"=== Cricket Analysis API Paths ===")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"UPLOADS_DIR: {UPLOADS_DIR}")
logger.info(f"OUTPUTS_DIR: {OUTPUTS_DIR}")
logger.info(f"SCRIPTS_DIR: {SCRIPTS_DIR}")
logger.info(f"MODELS_DIR: {MODELS_DIR}")
logger.info(f"Scripts exists: {SCRIPTS_DIR.exists()}")
logger.info(f"Models exists: {MODELS_DIR.exists()}")
logger.info(f"====================================")


def get_video_base_name(video_path: Path) -> str:
    """Get base name without extension for output files"""
    return video_path.stem


async def process_video(job_id: str, video_path: Path, bowler_height_m: float, handed: str, release_frame: int):
    """
    Process video through the cricket analysis pipeline
    """
    try:
        # Update status
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = "Running pose and ball detection..."
        logger.info(f"Starting processing for job {job_id}")

        base_name = get_video_base_name(video_path)
        logger.info(f"Base name: {base_name}")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Release frame: {release_frame}")
        
        # Step 1: Run combined pipeline
        # Use the Python from the virtual environment
        python_exe = "python"
        
        combined_cmd = [
            python_exe,
            str(SCRIPTS_DIR / "combined_pipeline.py"),
            "--video", str(video_path),
            "--bowler_height_m", str(bowler_height_m),
            "--handed", handed,
            "--pose_model", str(MODELS_DIR / "yolov8s-pose.pt"),
            "--det_model", str(MODELS_DIR / "ball_yolov8n.pt"),
            "--device", "cpu",
            "--use_hybrid_tracking",
            "--manual_release_frame", str(release_frame),
            "--skip_frame_selector"
        ]
        
        logger.info(f"Running command: {' '.join(combined_cmd)}")
        
        # Use synchronous subprocess (works better on Windows)
        import subprocess as sp
        result = sp.run(
            combined_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        
        logger.info(f"Combined pipeline return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Combined pipeline stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Combined pipeline stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"Combined pipeline failed (code {result.returncode}): {result.stderr or result.stdout or 'Unknown error'}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Step 2: Compute metrics
        jobs[job_id]["progress"] = "Computing biomechanics metrics..."
        
        metrics_cmd = [
            python_exe,
            str(SCRIPTS_DIR / "compute_metrics.py"),
            "--summary_json", str(OUTPUTS_DIR / f"{base_name}_summary.json"),
            "--ball_csv", str(OUTPUTS_DIR / f"{base_name}_ball_track.csv"),
            "--kpts_csv", str(OUTPUTS_DIR / f"{base_name}_keypoints.csv"),
            "--handed", handed,
            "--out_json", str(OUTPUTS_DIR / f"{base_name}_metrics_paper.json")
        ]
        
        logger.info(f"Running command: {' '.join(metrics_cmd)}")
        
        result = sp.run(
            metrics_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        
        logger.info(f"Compute metrics return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Compute metrics stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Compute metrics stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"Compute metrics failed (code {result.returncode}): {result.stderr or result.stdout or 'Unknown error'}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Load metrics JSON
        metrics_path = OUTPUTS_DIR / f"{base_name}_metrics_paper.json"
        logger.info(f"Loading metrics from: {metrics_path}")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Update job with results
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = "Analysis complete!"
        jobs[job_id]["metrics"] = metrics
        jobs[job_id]["video_filename"] = f"{base_name}_annotated.mp4"
        jobs[job_id]["metrics_filename"] = f"{base_name}_metrics_paper.json"
        logger.info(f"Job {job_id} completed successfully!")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id} failed with error: {error_msg}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = error_msg


def create_job(video_filename: str) -> str:
    """Create a new job and return job_id"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "video_filename": video_filename,
        "progress": "Video uploaded, waiting to process...",
        "metrics": None,
        "error": None
    }
    logger.info(f"Created job {job_id} for video {video_filename}")
    logger.info(f"Total jobs in memory: {len(jobs)}")
    return job_id


def get_job(job_id: str) -> dict:
    """Get job status"""
    logger.info(f"Looking for job {job_id}")
    logger.info(f"Available jobs: {list(jobs.keys())}")
    return jobs.get(job_id)


def cleanup_old_files(max_age_hours=24):
    """Delete uploads and outputs older than max_age_hours"""
    import time
    from datetime import datetime, timedelta
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    deleted_count = 0
    
    # Clean uploads folder
    if UPLOADS_DIR.exists():
        for item in UPLOADS_DIR.iterdir():
            try:
                # Check file/folder age
                item_age = current_time - item.stat().st_mtime
                if item_age > max_age_seconds:
                    if item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file/folder: {item}")
            except Exception as e:
                logger.error(f"Error deleting {item}: {e}")
    
    # Clean outputs folder (only files with job_id prefix - UUID pattern)
    if OUTPUTS_DIR.exists():
        for item in OUTPUTS_DIR.iterdir():
            try:
                # Only delete files that look like job outputs (have UUID in name)
                # UUID pattern: 8-4-4-4-12 characters
                name = item.name
                if len(name) > 36 and name[8] == '-' and name[13] == '-':
                    item_age = current_time - item.stat().st_mtime
                    if item_age > max_age_seconds:
                        if item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old output: {item}")
            except Exception as e:
                logger.error(f"Error deleting {item}: {e}")
    
    logger.info(f"Cleanup complete: deleted {deleted_count} old files/folders")
    return deleted_count


def extract_frames(video_path: Path, job_id: str) -> int:
    """
    Extract all frames from video as thumbnails
    Returns the total number of frames
    """
    import cv2
    
    frames_dir = UPLOADS_DIR / f"{job_id}_frames"
    frames_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Extracting {total_frames} frames from video")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for thumbnail (smaller file size)
        h, w = frame.shape[:2]
        scale = 320 / w  # 320px width thumbnails
        new_w = 320
        new_h = int(h * scale)
        thumbnail = cv2.resize(frame, (new_w, new_h))
        
        # Save as JPG
        thumb_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(thumb_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {frame_idx} frames to {frames_dir}")
    
    return frame_idx, fps