# Cricket Bowling Analysis API

FastAPI backend for analyzing cricket bowling videos using computer vision.

## Features

- Upload bowling videos
- Async processing (2-3 minutes per video)
- Returns biomechanics metrics + annotated video
- Job status tracking

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Project Structure:**
Ensure your project is structured like this:
```
cricket-analysis-api/
├── app/
├── uploads/          # Created automatically
├── outputs/          # Created automatically
├── requirements.txt
../scripts/           # Your existing scripts folder
├── combined_pipeline.py
└── compute_metrics.py
../models/           # Your existing models folder
├── yolov8s-pose.pt
└── ball_yolov8n.pt
```

3. **Run the server:**
```bash
python -m app
```

Or:
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

## API Endpoints

### 1. Upload Video
```bash
POST /v1/video
```

**Parameters:**
- `video`: Video file (multipart/form-data)
- `bowler_height_m`: Bowler height in meters (e.g., 1.75)
- `handed`: "right" or "left"

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/video" \
  -F "video=@bowling_video.mp4" \
  -F "bowler_height_m=1.75" \
  -F "handed=right"
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Video uploaded successfully. Processing started."
}
```

### 2. Check Status
```bash
GET /v1/video/{job_id}/status
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "progress": "Running pose and ball detection...",
  "metrics_url": null,
  "video_url": null
}
```

When completed:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": "Analysis complete!",
  "metrics_url": "/v1/video/{job_id}/download/metrics",
  "video_url": "/v1/video/{job_id}/download/video"
}
```

### 3. Download Metrics
```bash
GET /v1/video/{job_id}/download/metrics
```

Returns JSON with biomechanics metrics.

### 4. Download Annotated Video
```bash
GET /v1/video/{job_id}/download/video
```

Returns the annotated MP4 video file.

## Testing

Test with curl:

```bash
# 1. Upload
RESPONSE=$(curl -X POST "http://localhost:8000/v1/video" \
  -F "video=@test_video.mp4" \
  -F "bowler_height_m=1.75" \
  -F "handed=right")

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# 2. Check status (repeat until completed)
curl "http://localhost:8000/v1/video/$JOB_ID/status"

# 3. Download results
curl "http://localhost:8000/v1/video/$JOB_ID/download/metrics" > metrics.json
curl "http://localhost:8000/v1/video/$JOB_ID/download/video" > annotated.mp4
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

For production deployment, see deployment guides for:
- Railway
- Render
- DigitalOcean
- AWS/GCP/Azure

Remember to:
1. Set proper CORS origins in `api.py`
2. Use a proper database instead of in-memory job storage
3. Implement file cleanup for old uploads/outputs
4. Add authentication if needed
5. Set file size limits