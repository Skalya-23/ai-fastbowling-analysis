import argparse
import os
import json
import csv
import math
import numpy as np
import cv2
from ultralytics import YOLO

# ---- Imports from your project ----
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


def instantaneous_speed(px_seq, t_seq, m_per_px):
    if len(px_seq) < 5:
        return None

    xs = np.array([p[0] for p in px_seq], float)
    ys = np.array([p[1] for p in px_seq], float)
    ts = np.array(t_seq, float)

    N = min(9, len(ts))
    xs, ys, ts = xs[-N:], ys[-N:], ts[-N:]

    t0 = ts[0]
    tt = ts - t0
    A = np.vstack([tt, np.ones_like(tt)]).T

    vx, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
    vy, _ = np.linalg.lstsq(A, ys, rcond=None)[0]

    v_px_s = np.hypot(vx, vy)
    return float(v_px_s * m_per_px)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--pose_model", default="yolov8s-pose.pt", help="YOLO pose model")
    ap.add_argument("--det_model", default="yolov8s.pt", help="YOLO detection model")
    ap.add_argument("--imgsz_pose", type=int, default=960, help="Pose model image size")
    ap.add_argument("--imgsz_det", type=int, default=1280, help="Detection model image size")
    ap.add_argument("--device", default=None, help="Computation device (cpu/cuda)")
    ap.add_argument("--out_video", default=None, help="Output overlay video")
    ap.add_argument("--out_json", default=None, help="Output JSON summary")
    ap.add_argument("--out_csv_ball", default=None, help="Output CSV for ball track")
    ap.add_argument("--out_kpts", default=None, help="Output JSON/CSV for keypoints")
    ap.add_argument("--scale_m_per_px", type=float, default=None,
                    help="Manual scale (meters per pixel)")
    ap.add_argument("--max_jump_px", type=float, default=260.0,
                    help="Guard for keypoint ID switches")
    ap.add_argument("--handed", choices=["right", "left"], default="right",
                    help="Bowling arm")
    ap.add_argument("--hand_roi_px", type=int, default=160,
                    help="Square crop size around wrist for pre-release ball search")
    ap.add_argument("--bowler_height_m", type=float, default=None,
                    help="Bowler height in meters (REQUIRED for accurate calibration)")

    args = ap.parse_args()

    if args.bowler_height_m is None:
        print("[ERROR] --bowler_height_m is REQUIRED for accurate calibration!")
        return

    kpts_all = []
    frame_ids = []
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(os.path.basename(args.video))[0]

    if args.out_video is None:
        args.out_video = os.path.join(os.path.dirname(args.video), f"{base}_combined.mp4")
    if args.out_json is None:
        args.out_json = os.path.join(os.path.dirname(args.video), f"{base}_combined.json")
    if args.out_csv_ball is None:
        args.out_csv_ball = os.path.join(os.path.dirname(args.video), f"{base}_ball_track.csv")
    if args.out_kpts is None:
        args.out_kpts = os.path.join(os.path.dirname(args.video), f"{base}_kpts.json")

    m_pose = YOLO(args.pose_model)
    m_det = YOLO(args.det_model)
    dev = normalize_device(args.device)
    if dev is not None:
        m_pose.to(dev)
        m_det.to(dev)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (W, H))

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty video")

    (st_mid, st_y0, st_y1), crease_y = detect_stumps_and_crease(first)

    # FIXED: Always use bowler height for calibration
    m_per_px = args.scale_m_per_px
    if m_per_px is None:
        res_p = m_pose.predict(first, imgsz=args.imgsz_pose, verbose=False)[0]
        k0, _ = pick_person_keypoints(res_p)
        if k0 is not None:
            nose_y = k0[COCO_KPTS["nose"]][1]
            lank_y = k0[COCO_KPTS["lank"]][1]
            rank_y = k0[COCO_KPTS["rank"]][1]
            ankle_y = max(lank_y, rank_y)
            pixel_height = abs(ankle_y - nose_y)
            if pixel_height > 0:
                m_per_px = args.bowler_height_m / pixel_height
                print(f"[INFO] Calibration using bowler height: {m_per_px:.6f} m/px")
                print(f"[INFO] Bowler height in pixels: {pixel_height:.1f}px")

    if m_per_px is None:
        print("[ERROR] Could not detect bowler in first frame for calibration!")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    prev_center = None
    kf = None
    ball_track = []
    release_idx = None
    release_time = None

    # FIXED: Better release detection state
    near_hand_count = 0
    far_hand_count = 0
    last_near_idx = None
    hand_detected = False

    f = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        f += 1
        t = f / fps

        res_p = m_pose.predict(frame, imgsz=args.imgsz_pose, verbose=False)[0]
        kpts, center = pick_person_keypoints(res_p, prev_center=prev_center)

        if kpts is not None:
            if prev_center is not None and center is not None:
                if np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) > args.max_jump_px:
                    kpts = None
            if center is not None:
                prev_center = center

        kpts_all.append(kpts)
        frame_ids.append(f)

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
            pred = (kf.x[0, 0], kf.x[1, 0])

        p = best_det if best_det is not None else pred

        if p is not None:
            ball_track.append({
                "frame": f,
                "t": t,
                "x": float(p[0]),
                "y": float(p[1]),
                "meas": 1 if best_det is not None else 0
            })

        # FIXED: Improved release detection
        if release_idx is None and kpts is not None and p is not None and len(ball_track) > 5:
            rw = kpts[COCO_KPTS["rwri"]] if args.handed == "right" else kpts[COCO_KPTS["lwri"]]
            d_hand = float(np.hypot(p[0] - rw[0], p[1] - rw[1]))

            NEAR_THRESH = 40.0  # Increased threshold
            FAR_THRESH = 70.0   # Increased threshold
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
                        print(f"[INFO] Release detected at frame {ball_track[release_idx]['frame']}, t={release_time:.3f}s")
                else:
                    near_hand_count = 0
                    far_hand_count = 0

        overlay = frame.copy()
        if kpts is not None:
            draw_skeleton(overlay, kpts)
        if p is not None:
            cv2.circle(overlay, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1, cv2.LINE_AA)
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

        put_multiline_text(overlay,
                           [f"FPS:{fps:.2f}",
                            f"Scale:{m_per_px:.5f} m/px"],
                           (10, 30))

        out.write(overlay)

    cap.release()
    out.release()

    # FIXED: If no release detected, estimate from wrist height peak
    if release_idx is None and len(ball_track) > 0:
        print("[WARN] Release detection failed, estimating from ball trajectory...")
        # Find when ball is highest (earliest Y minimum, since Y increases downward)
        ys = [r["y"] for r in ball_track]
        release_idx = int(np.argmin(ys[:len(ys)//2])) if len(ys) > 10 else 0
        release_time = ball_track[release_idx]["t"]
        print(f"[INFO] Estimated release at frame {ball_track[release_idx]['frame']}, t={release_time:.3f}s")

    speeds = {}
    if release_idx is not None:
        xs = [r["x"] for r in ball_track[:release_idx+1]]
        ys = [r["y"] for r in ball_track[:release_idx+1]]
        ts = [r["t"] for r in ball_track[:release_idx+1]]
        v_mps = instantaneous_speed(list(zip(xs, ys)), ts, m_per_px)
        if v_mps:
            speeds["release_speed_kph"] = v_mps * 3.6

    if release_idx is not None and len(ball_track) > release_idx + 5:
        xs = [r["x"] for r in ball_track[release_idx:]]
        ys = [r["y"] for r in ball_track[release_idx:]]
        ts = [r["t"] for r in ball_track[release_idx:]]
        v_mps = instantaneous_speed(list(zip(xs, ys)), ts, m_per_px)
        if v_mps:
            speeds["avg_flight_speed_kph"] = v_mps * 3.6

    bfc_idx, ffc_idx, info = detect_bfc_ffc(kpts_all, fps=fps, handed=args.handed)
    bfc_frame = frame_ids[bfc_idx] if bfc_idx is not None else None
    ffc_frame = frame_ids[ffc_idx] if ffc_idx is not None else None
    t_bfc = (bfc_frame / fps) if bfc_frame is not None else None
    t_ffc = (ffc_frame / fps) if ffc_frame is not None else None

    out_ball_csv = os.path.join(os.path.dirname(args.video), f"{base}_ball_track.csv")
    with open(out_ball_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "x", "y", "is_measurement"])
        for r in ball_track:
            w.writerow([r["frame"], r["t"], r["x"], r["y"], r["meas"]])
    print(f"[DONE] ball track csv: {out_ball_csv}")

    out_kpts_csv = os.path.join(os.path.dirname(args.video), f"{base}_kpts.csv")
    with open(out_kpts_csv, "w", newline="") as f:
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
    print(f"[DONE] keypoints csv: {out_kpts_csv}")

    out_summary_json = os.path.join(os.path.dirname(args.video), f"{base}_summary.json")
    outj = {
        "video": os.path.abspath(args.video),
        "fps": float(fps),
        "frame_w": int(W),
        "frame_h": int(H),
        "scale_m_per_px": float(m_per_px) if m_per_px else None,
        "bowler_height_m": float(args.bowler_height_m),
        "release_time_s": float(release_time) if release_time else None,
        "release_frame": int(ball_track[release_idx]["frame"]) if release_idx is not None else None,
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
    with open(out_summary_json, "w") as f:
        json.dump(outj, f, indent=2)
    
    print(f"[DONE] overlay video: {args.out_video}")
    print(f"[DONE] summary json: {args.out_json}")
    print(f"\n=== CALIBRATION INFO ===")
    print(f"Scale: {m_per_px:.6f} m/px")
    print(f"Bowler height: {args.bowler_height_m}m")
    if release_time:
        print(f"Release time: {release_time:.3f}s")
    if t_bfc:
        print(f"BFC time: {t_bfc:.3f}s")
    if t_ffc:
        print(f"FFC time: {t_ffc:.3f}s")


if __name__ == "__main__":
    main()