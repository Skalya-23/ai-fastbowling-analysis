Cricket-YOLO-Pose (Starter)

What this does
--------------
- Runs Ultralytics YOLOv8 Pose on a bowling video.
- Detects Back Foot Contact (BFC) and Front Foot Contact (FFC).
- Computes:
  • BFC timestamp (s) and BFC→FFC time (s)
  • Back knee angle at BFC (deg)
  • Hip–Shoulder separation at FFC (deg)
  • Stride length (px, and meters if you pass a scale)
- Exports:
  • Overlayed video with skeleton + event markers
  • CSV with per-frame keypoints
  • JSON with summary metrics

Install (Windows PowerShell)
----------------------------
# 1) Create a folder and virtual env
# (Replace C:\Cricket-YOLO-Pose with your preferred path)
PS> cd C:\
PS> mkdir Cricket-YOLO-Pose
PS> cd .\Cricket-YOLO-Pose
PS> python -m venv .venv
PS> .\.venv\Scripts\Activate.ps1

# 2) Install deps
PS> pip install -U pip
PS> pip install -r requirements.txt

# 3) Place a video in .\data\videos\ (e.g., Bowler1.mp4)

# 4) Run the pipeline (bottom 3/4 ROI example)
PS> python .\scripts\run_pipeline.py ^
    --video ".\data\videos\Bowler1.mp4" ^
    --out_video ".\outputs\Bowler1_overlay.mp4" ^
    --out_csv ".\outputs\Bowler1_keypoints.csv" ^
    --out_json ".\outputs\Bowler1_metrics.json" ^
    --model "yolov8s-pose.pt" ^
    --roi "bottom0.75"

# Minimal run (full frame, auto-download model)
PS> python .\scripts\run_pipeline.py --video ".\data\videos\Bowler1.mp4"

Notes
-----
- Model weights: "yolov8s-pose.pt" will auto-download on first run.
  You can switch to "yolov8m-pose.pt" or "yolov8l-pose.pt" for more accuracy (slower).
- ROI: Set --roi "none" or "bottom0.75" (bottom 75% of frame). This helps ignore crowd/sky.
- Scale: If you know meters-per-pixel, pass --scale_m_per_px to get stride length in meters.
- Single bowler clips work best. The script picks the most confident person per frame.
- If detection flickers, try a larger model (yolov8m/l) or a higher input size (--imgsz 1280).

One-liners
----------
# Create venv + install + run in one line (edit the video path):
PS> cd C:\ && mkdir Cricket-YOLO-Pose; `
    cd .\Cricket-YOLO-Pose; `
    python -m venv .venv; `
    .\.venv\Scripts\Activate.ps1; `
    echo ultralytics>=8.2.0`nopencv-python>=4.8.0`nnumpy>=1.24`npandas>=2.0`nscipy>=1.10 > requirements.txt; `
    pip install -U pip; pip install -r requirements.txt; `
    mkdir data; mkdir data\videos; mkdir outputs; mkdir scripts; `
    echo (Get-Content -Raw .\README.txt); `
    python .\scripts\run_pipeline.py --video ".\data\videos\Bowler1.mp4" --roi bottom0.75

Good luck! – This is a starting point you can tweak to your footage and camera angle.
