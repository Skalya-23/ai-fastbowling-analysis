import argparse
import os
import json
import csv
import math
import numpy as np
import cv2
from ultralytics import YOLO

from draw_utils import draw_skeleton, put_multiline_text
from metrics_util import hip_shoulder_separation, angle_at_joint, euclid, smooth
from event_detect import detect_bfc_ffc


def normalize_device(d):
    if d is None:
        return None
    s = str(d).strip().lower()
    if s == 'cpu':
        return 'cpu'
    if s.startswith('cuda'):
        return s
    if s.isdigit():
        return f'cuda:{s}'
    return s


COCO_KPTS = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lwri": 9,
    "rwri": 10, "lhip": 11, "rhip": 12, "lknee": 13, "rknee": 14,
    "lank": 15, "rank": 16
}

STUMP_HEIGHT_M = 0.711


class KF2D:
    def __init__(self, x, y):
        self.x = np.array([[x, y, 0., 0.]], dtype=float).T
        self.P = np.eye(4) * 100.0
        self.Q = np.diag([4, 4, 10, 10])
        self.R = np.diag([25, 25])
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

    def predict(self, dt):
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z):
        z = np.array(z, dtype=float).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def pick_person_keypoints(result, prev_center=None):
    if result.keypoints is None or result.keypoints.xy is None:
        return None, None

    kpts_xy = result.keypoints.xy
    boxes = getattr(result, 'boxes', None)

    if kpts_xy is None or len(kpts_xy) == 0 or boxes is None or boxes.xyxy is None:
        return None, None

    kpts_np = kpts_xy.cpu().numpy()
    bx = boxes.xyxy.cpu().numpy()
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bx]

    cand = list(range(len(kpts_np)))
    if prev_center is not None:
        d = [np.hypot(centers[i][0] - prev_center[0], centers[i][1] - prev_center[1]) for i in cand]
        best = cand[int(np.argmin(d))]
    else:
        areas = [(bx[i][2] - bx[i][0]) * (bx[i][3] - bx[i][1]) for i in cand]
        best = cand[int(np.argmax(areas))]

    return kpts_np[best], centers[best]


def detect_stumps_and_crease(frame):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    mask_far = np.zeros_like(bw)
    mask_far[:int(H * 0.60), :] = 255
    vv = cv2.bitwise_and(bw, mask_far)
    lines = cv2.HoughLinesP(vv, rho=1, theta=np.pi / 180,
                            threshold=80, minLineLength=int(H * 0.08),
                            maxLineGap=6)

    st_mid, st_y0, st_y1 = None, None, None
    if lines is not None:
        vlines = []
        for l in lines[:, 0, :]:
            x0, y0, x1, y1 = l
            if abs(x1 - x0) < 6 and abs(y1 - y0) > int(H * 0.06):
                vlines.append((x0, y0, x1, y1))

        xs = sorted([(min(p[0], p[2]) + max(p[0], p[2])) / 2.0 for p in vlines])
        if len(xs) >= 3:
            best_span = 1e9
            best_triplet = None
            for i in range(len(xs) - 2):
                span = xs[i + 2] - xs[i]
                if span < best_span:
                    best_span = span
                    best_triplet = (xs[i], xs[i + 1], xs[i + 2])

            st_mid = np.mean(best_triplet)
            ys_top, ys_bot = [], []
            for (x0, y0, x1, y1) in vlines:
                cx = (x0 + x1) / 2.0
                if abs(cx - st_mid) <= best_span * 0.75:
                    ys_top.append(min(y0, y1))
                    ys_bot.append(max(y0, y1))
            if ys_top and ys_bot:
                st_y0 = int(np.median(ys_top))
                st_y1 = int(np.median(ys_bot))

    mask_near = np.zeros_like(bw)
    mask_near[int(H * 0.55):, :] = 255
    hh = cv2.bitwise_and(bw, mask_near)
    hlines = cv2.HoughLinesP(hh, rho=1, theta=np.pi / 180,
                             threshold=60, minLineLength=int(W * 0.25),
                             maxLineGap=8)

    crease_y = None
    if hlines is not None:
        best_len = 0
        for l in hlines[:, 0, :]:
            x0, y0, x1, y1 = l
            if abs(y1 - y0) < 6:
                length = abs(x1 - x0)
                midx = (x0 + x1) / 2
                if (W * 0.25) < midx < (W * 0.75) and length > best_len:
                    best_len = length
                    crease_y = int((y0 + y1) / 2)

    return (st_mid, st_y0, st_y1), crease_y


def ball_detection(model_det, frame, conf=0.10, class_name="ball",
                   imgsz=640, min_area=5, max_area=5000, max_aspect_ratio=4.0):
    res = model_det.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]

    cls_id = None
    for k, v in model_det.names.items():
        if str(v).lower() == class_name.lower():
            cls_id = int(k)
            break
    if cls_id is None:
        return []

    detections = []
    if res.boxes is not None and res.boxes.cls is not None:
        for i in range(len(res.boxes)):
            if int(res.boxes.cls[i].item()) == cls_id:
                xyxy = res.boxes.xyxy[i].cpu().numpy()
                conf_val = float(res.boxes.conf[i].item())
                x0, y0, x1, y1 = xyxy
                w, h = (x1 - x0), (y1 - y0)
                area = w * h
                aspect = max(w, h) / (min(w, h) + 1e-6)

                if area < min_area or area > max_area:
                    continue
                if aspect > max_aspect_ratio:
                    continue

                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                detections.append((cx, cy, conf_val))

    return detections


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def select_release_frame(video_path):
    """
    Interactive frame selector for choosing the release frame.
    Returns the selected frame number, or None if cancelled.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video for frame selection")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    current_frame = 0
    selected_frame = None
    paused = True
    
    # Window setup
    window_name = "Select Release Frame (SPACE: Play/Pause | LEFT/RIGHT: Navigate | ENTER: Confirm | ESC: Cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    def on_trackbar(val):
        nonlocal current_frame
        current_frame = val
    
    # Create trackbar
    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, on_trackbar)
    
    print("\n" + "="*60)
    print("FRAME SELECTOR CONTROLS:")
    print("  SPACE     : Play/Pause")
    print("  LEFT/RIGHT: Previous/Next frame (when paused)")
    print("  A/D       : Jump backward/forward 10 frames")
    print("  S         : Jump to start")
    print("  ENTER     : Confirm selection")
    print("  ESC       : Cancel and exit")
    print("="*60 + "\n")
    
    while True:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            current_frame = 0
            continue
        
        # Draw info overlay
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Semi-transparent overlay bar at top
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Frame info
        time_sec = current_frame / fps
        info_text = f"Frame: {current_frame}/{total_frames-1} | Time: {time_sec:.2f}s"
        cv2.putText(display_frame, info_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status = "PAUSED" if paused else "PLAYING"
        status_color = (0, 255, 255) if paused else (0, 255, 0)
        cv2.putText(display_frame, status, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Release frame marker
        if selected_frame is not None:
            marker_text = f"Selected: Frame {selected_frame}"
            cv2.putText(display_frame, marker_text, (w - 350, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ENTER to confirm", (w - 350, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Crosshair at center
        cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
        cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)
        
        cv2.imshow(window_name, display_frame)
        
        # Update trackbar
        cv2.setTrackbarPos('Frame', window_name, current_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30 if not paused else 100) & 0xFF
        
        if key == 27:  # ESC
            print("\n[INFO] Frame selection cancelled")
            selected_frame = None
            break
        elif key == 13:  # ENTER
            if selected_frame is None:
                selected_frame = current_frame
            print(f"\n[INFO] Release frame selected: {selected_frame}")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
        elif key == 83 or key == ord('d'):  # RIGHT arrow or 'd'
            if paused:
                current_frame = min(current_frame + (10 if key == ord('d') else 1), total_frames - 1)
        elif key == 81 or key == ord('a'):  # LEFT arrow or 'a'
            if paused:
                current_frame = max(current_frame - (10 if key == ord('a') else 1), 0)
        elif key == ord('s'):  # 's' - jump to start
            current_frame = 0
        elif key == ord('c'):  # 'c' - mark current as selected
            selected_frame = current_frame
            print(f"[INFO] Marked frame {selected_frame} as release point")
        
        # Auto-advance if playing
        if not paused:
            current_frame = (current_frame + 1) % total_frames
    
    cap.release()
    cv2.destroyAllWindows()
    
    return selected_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--bowler_height_m", type=float, required=True,
                    help="Bowler height in meters")
    ap.add_argument("--handed", required=True, choices=["left", "right"],
                    help="Bowler's bowling hand")
    ap.add_argument("--pose_model", required=True, help="YOLOv8 pose model path")
    ap.add_argument("--det_model", required=True, help="YOLO ball detection model path")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device (cpu, cuda, 0, 1, ...)")
    ap.add_argument("--imgsz_pose", type=int, default=640,
                    help="Image size for pose model")
    ap.add_argument("--imgsz_det", type=int, default=640,
                    help="Image size for detection model")
    ap.add_argument("--use_hybrid_tracking", action="store_true",
                    help="Use wrist before release, detector after")
    ap.add_argument("--manual_release_frame", type=int, default=None,
                    help="Manual release frame (if None, interactive selector or auto-detect)")
    ap.add_argument("--skip_frame_selector", action="store_true",
                    help="Skip interactive frame selector and use auto-detection")

    args = ap.parse_args()
    
    device = normalize_device(args.device)
    
    # Determine release frame
    manual_release_frame = args.manual_release_frame
    
    if manual_release_frame is None and not args.skip_frame_selector:
        print("\n[INFO] Opening interactive frame selector...")
        manual_release_frame = select_release_frame(args.video)
        
        if manual_release_frame is None:
            print("[INFO] No frame selected. Will use automatic release detection.")
    
    if manual_release_frame is not None:
        print(f"[INFO] Using manual release frame: {manual_release_frame}")
    else:
        print("[INFO] Manual release frame not provided. Will use automatic detection.")

    # Load models
    print("\n[INFO] Loading models...")
    m_pose = YOLO(args.pose_model)
    if device:
        m_pose.to(device)
    m_det = YOLO(args.det_model)
    if device:
        m_det.to(device)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    out_video = os.path.join(out_dir, f"{base_name}_annotated.mp4")
    out_csv_ball = os.path.join(out_dir, f"{base_name}_ball_track.csv")
    out_kpts = os.path.join(out_dir, f"{base_name}_keypoints.csv")
    out_json = os.path.join(out_dir, f"{base_name}_summary.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] FPS={fps}, Resolution={W}x{H}, TotalFrames={total_frames}")
    print(f"[INFO] Pose model: {args.pose_model}")
    print(f"[INFO] Ball det model: {args.det_model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Bowler height: {args.bowler_height_m} m")
    print(f"[INFO] Hybrid tracking: {args.use_hybrid_tracking}")
    print("[INFO] Processing frames...")

    # First pass: pose + stumps
    kpts_all = []
    frame_ids = []
    prev_center = None
    f = 0
    st_mid, st_y0, st_y1 = None, None, None
    crease_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if f == 0:
            (st_mid, st_y0, st_y1), crease_y = detect_stumps_and_crease(frame)

        res = m_pose.predict(frame, imgsz=args.imgsz_pose, verbose=False)[0]
        kpts, prev_center = pick_person_keypoints(res, prev_center)
        kpts_all.append(kpts)
        frame_ids.append(f)
        f += 1

    cap.release()

    if not kpts_all:
        raise RuntimeError("No frames with keypoints detected.")

    # Compute scale
    heights = []
    for k in kpts_all:
        if k is None:
            continue
        nose_y = k[COCO_KPTS["nose"], 1]
        lank_y = k[COCO_KPTS["lank"], 1]
        rank_y = k[COCO_KPTS["rank"], 1]
        if not np.isnan([nose_y, lank_y, rank_y]).any():
            avg_ankle = (lank_y + rank_y) / 2.0
            h_px = abs(avg_ankle - nose_y)
            if h_px > 10:
                heights.append(h_px)

    if not heights:
        raise RuntimeError("Could not measure body height in pixels.")
    avg_h_px = np.median(heights)
    m_per_px = args.bowler_height_m / avg_h_px

    # Second pass: ball tracking
    wrist_idx = COCO_KPTS["rwri"] if args.handed.lower() == "right" else COCO_KPTS["lwri"]
    
    ball_track = []
    kf = None
    release_detected = False
    release_idx = 0
    release_time = None
    near_hand_count = 0
    far_hand_count = 0
    last_near_idx = None
    hand_detected = False

    cap = cv2.VideoCapture(args.video)
    f = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kpts = kpts_all[f]
        t = f / fps

        # Check if we've reached manual release frame
        if manual_release_frame is not None and f == manual_release_frame and not release_detected:
            release_detected = True
            release_idx = len(ball_track)  # Will be set correctly after ball_pos is added
            release_time = t
            print(f"[INFO] Using manual release at frame {manual_release_frame}, t={release_time:.3f}s")

        # HYBRID BALL TRACKING
        ball_pos = None
        is_measurement = 0
        
        if args.use_hybrid_tracking and not release_detected and kpts is not None:
            # Use wrist position before release
            wrist = kpts[wrist_idx]
            if not np.isnan(wrist).any():
                ball_pos = (float(wrist[0]), float(wrist[1]))
                is_measurement = 2  # Code 2 = wrist position
        else:
            # Use ball detector
            detections = ball_detection(m_det, frame, conf=0.15, imgsz=args.imgsz_det)
            pred = kf.predict(1.0 / fps) if kf is not None else None

            best_det = None
            if detections:
                if pred is not None:
                    gating_radius = 80.0
                    scored = []
                    for (cx, cy, conf) in detections:
                        dist = np.hypot(cx - pred[0], cy - pred[1])
                        score = conf - (dist / gating_radius)
                        scored.append((score, (cx, cy)))
                    scored.sort(reverse=True, key=lambda x: x[0])
                    best_det = scored[0][1] if scored else None
                else:
                    best_det = max(detections, key=lambda d: d[2])[:2]

            if kf is None and best_det is not None:
                kf = KF2D(best_det[0], best_det[1])

            if kf is not None:
                if best_det is not None:
                    kf.update(best_det)
                    is_measurement = 1
                pred = (kf.x[0, 0], kf.x[1, 0])

            ball_pos = best_det if best_det is not None else pred

        if ball_pos is not None:
            ball_track.append({
                "frame": f,
                "t": t,
                "x": float(ball_pos[0]),
                "y": float(ball_pos[1]),
                "meas": is_measurement
            })
            
            # Update release_idx if manual release just triggered
            if manual_release_frame == f and release_detected and release_idx == 0:
                release_idx = len(ball_track) - 1

        # Auto release detection (only if manual not provided)
        if manual_release_frame is None and not release_detected and kpts is not None and ball_pos is not None and len(ball_track) > 5:
            wrist = kpts[wrist_idx]
            d_hand = float(np.hypot(ball_pos[0] - wrist[0], ball_pos[1] - wrist[1]))

            NEAR_THRESH = 50.0
            FAR_THRESH = 90.0
            NEAR_MIN = 3
            FAR_MIN = 3

            cur_idx = len(ball_track) - 1

            if d_hand < NEAR_THRESH:
                near_hand_count += 1
                far_hand_count = 0
                last_near_idx = cur_idx
                hand_detected = True
            elif d_hand > FAR_THRESH and hand_detected:
                if near_hand_count >= NEAR_MIN:
                    far_hand_count += 1
                    if far_hand_count >= FAR_MIN:
                        release_idx = last_near_idx + 1 if last_near_idx is not None else cur_idx - 2
                        release_idx = max(0, min(release_idx, cur_idx - 1))
                        release_time = ball_track[release_idx]["t"]
                        release_detected = True
                        print(f"[INFO] Auto release detected at frame {ball_track[release_idx]['frame']}, t={release_time:.3f}s")
                else:
                    near_hand_count = 0
                    far_hand_count = 0

        # Draw overlay
        overlay = frame.copy()
        if kpts is not None:
            draw_skeleton(overlay, kpts)
        if ball_pos is not None:
            color = (0, 255, 255) if is_measurement == 2 else (0, 0, 255)
            cv2.circle(overlay, (int(ball_pos[0]), int(ball_pos[1])), 6, color, -1, cv2.LINE_AA)
        if st_mid is not None and st_y0 is not None and st_y1 is not None:
            cv2.line(overlay, (int(st_mid), int(st_y0)), (int(st_mid), int(st_y1)),
                     (0, 255, 255), 3, cv2.LINE_AA)
        if crease_y is not None:
            cv2.line(overlay, (int(W * 0.2), int(crease_y)), (int(W * 0.8), int(crease_y)),
                     (0, 165, 255), 2, cv2.LINE_AA)
        if release_time is not None:
            cv2.putText(overlay, f"Release @ {release_time:.3f}s",
                        (30, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (50, 220, 255), 2, cv2.LINE_AA)

        mode_text = "Wrist" if (is_measurement == 2) else "Ball"
        put_multiline_text(overlay,
                           [f"FPS:{fps:.2f}",
                            f"Scale:{m_per_px:.5f} m/px",
                            f"Mode: {mode_text}"],
                           (10, 30))

        out.write(overlay)
        f += 1

    cap.release()
    out.release()

    # Fallback release detection ONLY if manual was not provided
    if manual_release_frame is None and not release_detected and len(ball_track) > 0:
        print("[WARN] Auto release detection failed, estimating from trajectory...")
        ys = [r["y"] for r in ball_track]
        release_idx = int(np.argmin(ys[:len(ys)//2])) if len(ys) > 10 else 0
        release_time = ball_track[release_idx]["t"]
        print(f"[INFO] Estimated release at frame {ball_track[release_idx]['frame']}, t={release_time:.3f}s")

    # Event detection - PASS RELEASE FRAME
    release_frame_for_events = ball_track[release_idx]["frame"] if release_idx is not None and release_idx < len(ball_track) else None
    bfc_idx, ffc_idx, info = detect_bfc_ffc(kpts_all, fps=fps, handed=args.handed, release_frame=release_frame_for_events)
    
    bfc_frame = frame_ids[bfc_idx] if bfc_idx is not None else None
    ffc_frame = frame_ids[ffc_idx] if ffc_idx is not None else None
    t_bfc = (bfc_frame / fps) if bfc_frame is not None else None
    t_ffc = (ffc_frame / fps) if ffc_frame is not None else None

    # Save outputs
    with open(out_csv_ball, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "x", "y", "is_measurement"])
        for r in ball_track:
            w.writerow([r["frame"], r["t"], r["x"], r["y"], r["meas"]])

    with open(out_kpts, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame"]
        for name in COCO_KPTS.keys():
            header.extend([f"{name}_x", f"{name}_y"])
        w.writerow(header)

        for fid, k in zip(frame_ids, kpts_all):
            if k is None:
                w.writerow([fid] + [None] * (2 * len(COCO_KPTS)))
            else:
                row = [fid]
                for j in range(len(COCO_KPTS)):
                    row.extend(k[j].tolist())
                w.writerow(row)

    outj = {
        "video": os.path.abspath(args.video),
        "fps": float(fps),
        "frame_w": int(W),
        "frame_h": int(H),
        "scale_m_per_px": float(m_per_px) if m_per_px else None,
        "bowler_height_m": float(args.bowler_height_m),
        "release_time_s": float(release_time) if release_time else None,
        "release_frame": int(ball_track[release_idx]["frame"]) if release_idx is not None and release_idx < len(ball_track) else None,
        "hybrid_tracking_used": args.use_hybrid_tracking,
        "bowler_events": {
            "BFC_time_s": float(t_bfc) if t_bfc else None,
            "BFC_frame": int(bfc_frame) if bfc_frame else None,
            "FFC_time_s": float(t_ffc) if t_ffc else None,
            "FFC_frame": int(ffc_frame) if ffc_frame else None,
            "BFC_foot": info.get("bfc_foot") if isinstance(info, dict) else None
        },
        "stumps_detected": {
            "x_mid": float(st_mid) if st_mid is not None else None,
            "y_top": int(st_y0) if st_y0 is not None else None,
            "y_bot": int(st_y1) if st_y1 is not None else None
        },
        "crease_y": int(crease_y) if crease_y is not None else None
    }

    outj = make_json_safe(outj)
    with open(out_json, "w") as f:
        json.dump(outj, f, indent=2)
    
    print(f"\n[DONE] overlay video: {out_video}")
    print(f"[DONE] ball track csv: {out_csv_ball}")
    print(f"[DONE] keypoints csv: {out_kpts}")
    print(f"[DONE] summary json: {out_json}")
    print(f"\n=== SUMMARY ===")
    print(f"Scale: {m_per_px:.6f} m/px")
    if release_time:
        print(f"Release: {release_time:.3f}s (frame {ball_track[release_idx]['frame'] if release_idx < len(ball_track) else 'N/A'})")
    if t_bfc:
        print(f"BFC: {t_bfc:.3f}s (frame {bfc_frame})")
    if t_ffc:
        print(f"FFC: {t_ffc:.3f}s (frame {ffc_frame})")


if __name__ == "__main__":
    main()