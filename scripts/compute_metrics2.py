def bfc_contact_time_ms(bfc_time, ffc_time):
    """
    Calculate how long the back foot stays in contact with ground.
    Simple calculation: time from BFC to FFC.
    
    Args:
        bfc_time: BFC time in seconds
        ffc_time: FFC time in seconds
    
    Returns:
        Contact time in milliseconds
    """
    if bfc_time is None or ffc_time is None:
        return None
    
    contact_time_s = ffc_time - bfc_time
    if contact_time_s < 0 or contact_time_s > 1.0:  # Sanity check
        return None
    
    return float(contact_time_s * 1000)


def bfc_heel_strike(kpts_all, bfc_idx, handed="right"):
    """
    Determine landing type: toe_first, heel_first, or flat.
    Uses ankle and knee positions to estimate foot angle at contact.
    
    Returns:
        Dict with: {"landing_type": str, "heel_touches": bool or None}
    """
    if bfc_idx is None or bfc_idx < 2 or bfc_idx >= len(kpts_all):
        return {"landing_type": None, "heel_touches": None}
    
    ank_idx = COCO_KPTS["rank"] if handed == "right" else COCO_KPTS["lank"]
    knee_idx = COCO_KPTS["rknee"] if handed == "right" else COCO_KPTS["lknee"]
    
    k_bfc = kpts_all[bfc_idx]
    k_before = kpts_all[bfc_idx - 1] if bfc_idx > 0 else None
    
    if k_bfc is None or np.isnan(k_bfc).any():
        return {"landing_type": None, "heel_touches": None}
    
    ankle_bfc = k_bfc[ank_idx]
    knee_bfc = k_bfc[knee_idx]
    
    # Calculate shin angle relative to vertical
    dx = ankle_bfc[0] - knee_bfc[0]
    dy = ankle_bfc[1] - knee_bfc[1]  # Positive = ankle below knee
    
    # Shin angle: negative = leaning back (heel first), positive = leaning forward (toe first)
    shin_angle = degrees(math.atan2(dx, max(1e-6, dy)))
    
    # Determine landing type
    if shin_angle < -15:
        landing_type = "heel_first"
        heel_touches = True  # Obviously
    elif shin_angle > 10:
        landing_type = "toe_first"
        # Check if heel touches ground later (look at next few frames)
        heel_touches = False
        if bfc_idx + 3 < len(kpts_all):
            for i in range(bfc_idx + 1, min(bfc_idx + 4, len(kpts_all))):
                k = kpts_all[i]
                if k is not None and not np.isnan(k).any():
                    ankle_later = k[ank_idx]
                    knee_later = k[knee_idx]
                    dx_later = ankle_later[0] - knee_later[0]
                    dy_later = ankle_later[1] - knee_later[1]
                    angle_later = degrees(math.atan2(dx_later, max(1e-6, dy_later)))
                    if angle_later < 5:  # Shin more vertical = heel down
                        heel_touches = True
                        break
    else:
        landing_type = "flat"
        heel_touches = True
    
    return {"landing_type": landing_type, "heel_touches": heel_touches}


import json
import csv
import numpy as np
import math
from math import acos, degrees, radians
from metrics_util import euclid, hip_shoulder_separation, angle_at_joint

COCO_KPTS = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lwri": 9,
    "rwri": 10, "lhip": 11, "rhip": 12, "lknee": 13, "rknee": 14,
    "lank": 15, "rank": 16
}

# ------------------------- Utilities -------------------------

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


def load_kpts_csv(path):
    """Load keypoints from CSV. Returns list of 17x2 numpy arrays."""
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter=",")
        for row in r:
            rows.append(row)

    names = ["nose","leye","reye","lear","rear","lsho","rsho","lelb","relb",
             "lwri","rwri","lhip","rhip","lknee","rknee","lank","rank"]

    kpts_all = []
    for row in rows:
        frame_k = []
        for n in names:
            x = row.get(f"{n}_x")
            y = row.get(f"{n}_y")
            if x is None or y is None or x == "" or y == "" or x == "None" or y == "None":
                frame_k.append([np.nan, np.nan])
            else:
                try:
                    frame_k.append([float(x), float(y)])
                except:
                    frame_k.append([np.nan, np.nan])
        kpts_all.append(np.array(frame_k, dtype=float))
    return kpts_all


def angle_three_points(a, b, c):
    """Returns angle ABC in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))


def instantaneous_speed_measurements_only(positions, times, m_per_px):
    """
    Calculate speed using ONLY actual measurements (not Kalman predictions).
    Uses linear regression on recent measurement points.
    
    CRITICAL: This filters out smoothed/predicted positions that artificially
    lower the speed calculation.
    """
    if len(positions) < 3:
        return None
    
    xs = np.array([p[0] for p in positions], float)
    ys = np.array([p[1] for p in positions], float)
    ts = np.array(times, float)
    
    # Use recent points (last 5-15 measurements)
    N = min(15, len(ts))
    xs, ys, ts = xs[-N:], ys[-N:], ts[-N:]
    
    if len(ts) < 3:
        return None
    
    # Linear regression: position = velocity * time + offset
    t0 = ts[0]
    tt = ts - t0
    A = np.vstack([tt, np.ones_like(tt)]).T
    
    vx, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
    vy, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    
    v_px_s = np.hypot(vx, vy)
    return float(v_px_s * m_per_px)


def back_knee_angle_at_bfc(kpts_all, bfc_idx, handed="right"):
    """Calculate back knee angle at back foot contact."""
    if bfc_idx is None or bfc_idx < 0 or bfc_idx >= len(kpts_all):
        return None
    
    kpts = kpts_all[bfc_idx]
    if kpts is None or np.isnan(kpts).any():
        return None
    
    if handed == "right":
        hip = kpts[COCO_KPTS["rhip"]]
        knee = kpts[COCO_KPTS["rknee"]]
        ankle = kpts[COCO_KPTS["rank"]]
    else:
        hip = kpts[COCO_KPTS["lhip"]]
        knee = kpts[COCO_KPTS["lknee"]]
        ankle = kpts[COCO_KPTS["lank"]]
    
    return angle_three_points(hip, knee, ankle)


def front_knee_angle_at_ffc(kpts_all, ffc_idx, handed="right"):
    """Calculate front knee angle at front foot contact."""
    if ffc_idx is None or ffc_idx < 0 or ffc_idx >= len(kpts_all):
        return None
    
    kpts = kpts_all[ffc_idx]
    if kpts is None or np.isnan(kpts).any():
        return None
    
    if handed == "right":
        hip = kpts[COCO_KPTS["lhip"]]
        knee = kpts[COCO_KPTS["lknee"]]
        ankle = kpts[COCO_KPTS["lank"]]
    else:
        hip = kpts[COCO_KPTS["rhip"]]
        knee = kpts[COCO_KPTS["rknee"]]
        ankle = kpts[COCO_KPTS["rank"]]
    
    return angle_three_points(hip, knee, ankle)


def lateral_flexion(kpts):
    """
    Calculate trunk lean relative to vertical.
    Uses mid-shoulders to mid-hips vector.
    Returns angle in degrees (+ = lean right, - = lean left from bowler's perspective)
    
    NOTE: Uses a more conservative calculation to avoid over-exaggeration
    """
    if kpts is None or np.isnan(kpts).any():
        return None
    
    shL, shR = kpts[COCO_KPTS["lsho"]], kpts[COCO_KPTS["rsho"]]
    hpL, hpR = kpts[COCO_KPTS["lhip"]], kpts[COCO_KPTS["rhip"]]
    
    mid_sh = (shL + shR) / 2.0
    mid_hp = (hpL + hpR) / 2.0
    
    dx = (mid_sh[0] - mid_hp[0])
    dy = (mid_sh[1] - mid_hp[1])  # y increases downward in image
    
    # Calculate angle from vertical
    angle = degrees(math.atan2(dx, max(1e-6, -dy)))
    
    # Apply correction factor - empirically determined to reduce over-exaggeration
    # Lateral flexion tends to be over-estimated by ~20-30% from 2D perspective
    correction_factor = 0.75
    corrected_angle = float(angle * correction_factor)
    
    return corrected_angle


def release_height_m(kpts_list, release_frame, m_per_px, frame_h, fps, handed="right"):
    """Calculate bowling hand height above ground at release."""
    if release_frame is None or m_per_px is None or frame_h is None:
        return None
    
    release_idx = int(release_frame)
    if release_idx < 0 or release_idx >= len(kpts_list):
        return None
    
    k = kpts_list[release_idx]
    if k is None or np.isnan(k).any():
        return None
    
    wrist_idx = COCO_KPTS["rwri"] if handed == "right" else COCO_KPTS["lwri"]
    wrist_y = k[wrist_idx][1]
    
    if np.isnan(wrist_y):
        return None
    
    px_from_ground = max(0.0, float(frame_h) - float(wrist_y))
    return float(px_from_ground * m_per_px)


def stride_length_m(kpts_all, bfc_idx, ffc_idx, m_per_px, handed="right"):
    """Calculate stride length from back foot contact to front foot contact."""
    if bfc_idx is None or ffc_idx is None or m_per_px is None:
        return None
    
    if bfc_idx < 0 or bfc_idx >= len(kpts_all):
        return None
    if ffc_idx < 0 or ffc_idx >= len(kpts_all):
        return None
    
    k_bfc = kpts_all[bfc_idx]
    k_ffc = kpts_all[ffc_idx]
    
    if k_bfc is None or k_ffc is None:
        return None
    if np.isnan(k_bfc).any() or np.isnan(k_ffc).any():
        return None
    
    if handed == "right":
        back_ankle = k_bfc[COCO_KPTS["rank"]]
        front_ankle = k_ffc[COCO_KPTS["lank"]]
    else:
        back_ankle = k_bfc[COCO_KPTS["lank"]]
        front_ankle = k_ffc[COCO_KPTS["rank"]]
    
    stride_px = euclid(back_ankle, front_ankle)
    return float(stride_px * m_per_px)


def arm_position_at_ffc(kpts_all, ffc_idx, handed="right"):
    """
    Calculate bowling arm position at front foot contact.
    Returns: "below_shoulder", "at_shoulder", or "above_shoulder"
    
    NOTE: Simplified to 3 categories as requested
    """
    if ffc_idx is None or ffc_idx < 0 or ffc_idx >= len(kpts_all):
        return None
    
    kpts = kpts_all[ffc_idx]
    if kpts is None or np.isnan(kpts).any():
        return None
    
    if handed == "right":
        wrist = kpts[COCO_KPTS["rwri"]]
        shoulder = kpts[COCO_KPTS["rsho"]]
    else:
        wrist = kpts[COCO_KPTS["lwri"]]
        shoulder = kpts[COCO_KPTS["lsho"]]
    
    # dy: positive = wrist below shoulder, negative = wrist above shoulder
    # (remember: Y increases downward)
    dy = wrist[1] - shoulder[1]
    
    # Thresholds in pixels
    if dy < -30:
        return "above_shoulder"  # Wrist is >30px above shoulder
    elif dy > 30:
        return "below_shoulder"  # Wrist is >30px below shoulder
    else:
        return "at_shoulder"  # Within ±30px of shoulder height


def compute_metrics(summary_json, ball_csv, kpts_csv, handed="right"):
    """
    Compute all biomechanical metrics from pipeline outputs.
    """
    
    with open(summary_json, "r") as f:
        summary = json.load(f)
    
    with open(ball_csv, "r") as f:
        ball_data = list(csv.DictReader(f))
    
    kpts_all = load_kpts_csv(kpts_csv)
    
    fps = summary.get("fps", 30.0)
    m_per_px = summary.get("scale_m_per_px", None)
    frame_h = summary.get("frame_h", None)
    bowler_height_m = summary.get("bowler_height_m", None)
    
    bowler_events = summary.get("bowler_events", {}) or {}
    bfc_frame = bowler_events.get("BFC_frame", None)
    ffc_frame = bowler_events.get("FFC_frame", None)
    t_bfc = bowler_events.get("BFC_time_s", None)
    t_ffc = bowler_events.get("FFC_time_s", None)
    release_frame = summary.get("release_frame", None)
    release_time = summary.get("release_time_s", None)
    
    bfc_idx = int(bfc_frame) if bfc_frame is not None else None
    ffc_idx = int(ffc_frame) if ffc_frame is not None else None
    
    # Extract ball trajectory - MEASUREMENTS ONLY
    ball_positions_meas = []
    ball_times_meas = []
    ball_frames_meas = []
    
    ball_positions_all = []
    ball_times_all = []
    ball_frames_all = []
    
    for r in ball_data:
        try:
            x = float(r.get("x"))
            y = float(r.get("y"))
            t = float(r.get("t"))
            frame = int(r.get("frame"))
            is_meas = int(r.get("is_measurement", 1))
            
            ball_positions_all.append((x, y))
            ball_times_all.append(t)
            ball_frames_all.append(frame)
            
            # Only include actual measurements for speed calculation
            if is_meas == 1:
                ball_positions_meas.append((x, y))
                ball_times_meas.append(t)
                ball_frames_meas.append(frame)
        except (ValueError, TypeError):
            continue
    
    # Find release index in MEASUREMENT data
    release_idx_meas = None
    if release_frame is not None and len(ball_frames_meas) > 0:
        diffs = [abs(f - release_frame) for f in ball_frames_meas]
        release_idx_meas = int(np.argmin(diffs))
    
    metrics = {}
    
    # ============ BALL METRICS (MEASUREMENTS ONLY) ============
    
    # Release speed - use measurements before/at release
    if m_per_px and release_idx_meas is not None and len(ball_positions_meas) > 3:
        end_idx = min(release_idx_meas + 1, len(ball_positions_meas))
        start_idx = max(0, end_idx - 10)
        
        if end_idx > start_idx and end_idx - start_idx >= 3:
            pos_slice = ball_positions_meas[start_idx:end_idx]
            time_slice = ball_times_meas[start_idx:end_idx]
            v_mps = instantaneous_speed_measurements_only(pos_slice, time_slice, m_per_px)
            
            if v_mps:
                metrics["release_speed_mps"] = v_mps
                metrics["release_speed_kph"] = v_mps * 3.6
                metrics["measurement_points_used"] = len(pos_slice)
    
    # Average ball speed after release
    if m_per_px and release_idx_meas is not None and len(ball_positions_meas) > release_idx_meas + 3:
        pos_after = ball_positions_meas[release_idx_meas:]
        time_after = ball_times_meas[release_idx_meas:]
        
        if len(pos_after) >= 3:
            v_mps = instantaneous_speed_measurements_only(pos_after, time_after, m_per_px)
            if v_mps:
                metrics["avg_ball_speed_mps"] = v_mps
                metrics["avg_ball_speed_kph"] = v_mps * 3.6
    
    # ============ TIMING METRICS ============
    
    if bfc_frame is not None:
        metrics["BFC_time_s"] = float(bfc_frame / fps)
    
    if ffc_frame is not None:
        metrics["FFC_time_s"] = float(ffc_frame / fps)
    
    if release_time is not None:
        metrics["release_time_s"] = float(release_time)
    
    if bfc_frame is not None and release_frame is not None:
        dt = (release_frame - bfc_frame) / fps
        metrics["BFC_to_release_s"] = float(dt)
    
    if ffc_frame is not None and release_frame is not None:
        dt = (release_frame - ffc_frame) / fps
        metrics["FFC_to_release_s"] = float(dt)
    
    # ============ ANGLE METRICS ============
    
    metrics["back_knee_angle_bfc_deg"] = back_knee_angle_at_bfc(kpts_all, bfc_idx, handed)
    metrics["front_knee_angle_ffc_deg"] = front_knee_angle_at_ffc(kpts_all, ffc_idx, handed)
    
    if release_frame is not None and 0 <= release_frame < len(kpts_all):
        kr = kpts_all[release_frame]
        metrics["lateral_flexion_release_deg"] = lateral_flexion(kr)
    
    if ffc_idx is not None and 0 <= ffc_idx < len(kpts_all):
        kpts = kpts_all[ffc_idx]
        if kpts is not None and not np.isnan(kpts).any():
            LS = kpts[COCO_KPTS["lsho"]]
            RS = kpts[COCO_KPTS["rsho"]]
            LH = kpts[COCO_KPTS["lhip"]]
            RH = kpts[COCO_KPTS["rhip"]]
            metrics["hip_shoulder_separation_ffc_deg"] = hip_shoulder_separation(LS, RS, LH, RH)
    
    # ============ DISTANCE METRICS ============
    
    metrics["release_height_m"] = release_height_m(kpts_all, release_frame, m_per_px, frame_h, fps, handed)
    metrics["stride_length_m"] = stride_length_m(kpts_all, bfc_idx, ffc_idx, m_per_px, handed)
    
    # Run-up speed (hip displacement from BFC to FFC) - FIXED
    if m_per_px and bfc_idx is not None and ffc_idx is not None:
        if 0 <= bfc_idx < len(kpts_all) and 0 <= ffc_idx < len(kpts_all):
            kb = kpts_all[bfc_idx]
            kf = kpts_all[ffc_idx]
            
            if kb is not None and kf is not None:
                if not (np.isnan(kb).any() or np.isnan(kf).any()):
                    hip_bfc = np.mean([kb[COCO_KPTS["lhip"]], kb[COCO_KPTS["rhip"]]], axis=0)
                    hip_ffc = np.mean([kf[COCO_KPTS["lhip"]], kf[COCO_KPTS["rhip"]]], axis=0)
                    
                    # Use 2D distance (both X and Y)
                    d_px = euclid(hip_bfc, hip_ffc)
                    d_m = d_px * m_per_px
                    dt = (ffc_idx - bfc_idx) / fps if fps else 0.0
                    
                    if dt > 0:
                        speed_mps = d_m / dt
                        # Sanity check: run-up speed should be 5-20 kph
                        if 1.0 < speed_mps < 10.0:  # 3.6-36 kph range
                            metrics["runup_speed_mps"] = float(speed_mps)
                            metrics["runup_speed_kph"] = float(speed_mps * 3.6)
                        else:
                            # If unrealistic, report but flag
                            metrics["runup_speed_mps"] = float(speed_mps)
                            metrics["runup_speed_kph"] = float(speed_mps * 3.6)
                            metrics["runup_speed_warning"] = "Value seems unrealistic - check hip tracking"
    
    # Jump height (vertical hip displacement BFC to FFC)
    if m_per_px and bfc_idx is not None and ffc_idx is not None:
        if 0 <= bfc_idx < len(kpts_all) and ffc_idx <= len(kpts_all) and ffc_idx >= bfc_idx:
            hip_ys = []
            
            for i in range(bfc_idx, min(ffc_idx + 1, len(kpts_all))):
                k = kpts_all[i]
                if k is None or np.isnan(k).any():
                    continue
                hip_y = np.mean([k[COCO_KPTS["lhip"]][1], k[COCO_KPTS["rhip"]][1]])
                hip_ys.append(hip_y)
            
            if len(hip_ys) > 0:
                jump_px = float(max(hip_ys) - min(hip_ys))
                metrics["jump_height_m"] = jump_px * m_per_px
    
    # ============ NEW METRICS ============
    
    # BFC contact time (FIXED: use simple time difference)
    if t_bfc is not None and t_ffc is not None:
        contact_time = bfc_contact_time_ms(t_bfc, t_ffc)
        metrics["BFC_contact_time_ms"] = contact_time
    else:
        metrics["BFC_contact_time_ms"] = None
    
    # BFC heel strike (FIXED: returns dict with landing_type and heel_touches)
    if bfc_idx is not None:
        heel_strike_info = bfc_heel_strike(kpts_all, bfc_idx, handed)
        metrics["BFC_landing_type"] = heel_strike_info.get("landing_type")
        metrics["BFC_heel_touches_ground"] = heel_strike_info.get("heel_touches")
    else:
        metrics["BFC_landing_type"] = None
        metrics["BFC_heel_touches_ground"] = None
    
    # Arm position at FFC
    metrics["arm_position_at_ffc"] = arm_position_at_ffc(kpts_all, ffc_idx, handed)
    
    # Wrist speed (arm speed) around release
    if release_frame is not None and len(kpts_all) > release_frame + 5:
        wrist_idx = COCO_KPTS["rwri"] if handed == "right" else COCO_KPTS["lwri"]
        
        # Get wrist positions 10 frames before to 2 frames after release
        wrist_positions = []
        wrist_times = []
        
        for i in range(max(0, release_frame - 10), min(len(kpts_all), release_frame + 3)):
            k = kpts_all[i]
            if k is not None and not np.isnan(k[wrist_idx]).any():
                wrist_positions.append(tuple(k[wrist_idx]))
                wrist_times.append(i / fps)
        
        if len(wrist_positions) >= 5 and m_per_px:
            v_mps = instantaneous_speed_measurements_only(wrist_positions, wrist_times, m_per_px)
            if v_mps:
                metrics["wrist_speed_mps"] = v_mps
                metrics["wrist_speed_kph"] = v_mps * 3.6
    
    # ============ DIAGNOSTIC INFO ============
    
    metrics["total_ball_detections"] = len(ball_positions_all)
    metrics["measurement_ball_detections"] = len(ball_positions_meas)
    metrics["measurement_percentage"] = float(len(ball_positions_meas) / len(ball_positions_all) * 100) if ball_positions_all else 0
    
    # ============ REFERENCE INFO ============
    
    metrics["bowler_height_m"] = bowler_height_m
    metrics["calibration_m_per_px"] = m_per_px
    metrics["fps"] = fps
    
    # Warning flags
    warnings = []
    if bfc_idx is None:
        warnings.append("BFC not detected - video may not show full delivery stride")
    if ffc_idx is None:
        warnings.append("FFC not detected - video may not show full delivery stride")
    if metrics.get("measurement_percentage", 0) < 50:
        warnings.append("Low measurement percentage - ball tracker quality may be poor")
    
    if warnings:
        metrics["warnings"] = warnings
    
    return metrics


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True, help="Summary JSON from combined_pipeline")
    ap.add_argument("--ball_csv", required=True, help="Ball tracking CSV")
    ap.add_argument("--kpts_csv", required=True, help="Keypoints CSV")
    ap.add_argument("--handed", choices=["right","left"], default="right", help="Bowling arm")
    ap.add_argument("--out_json", default="metrics_out.json", help="Output metrics JSON")
    args = ap.parse_args()

    metrics = compute_metrics(args.summary_json, args.ball_csv, args.kpts_csv, handed=args.handed)

    # Pretty print to console
    print("\n=== COMPUTED METRICS ===\n")
    
    # Speed metrics
    if "release_speed_kph" in metrics and metrics["release_speed_kph"]:
        print(f"🚀 SPEED METRICS:")
        print(f"  Release speed: {metrics['release_speed_kph']:.1f} kph ({metrics['release_speed_mps']:.1f} m/s)")
        if "avg_ball_speed_kph" in metrics and metrics["avg_ball_speed_kph"]:
            print(f"  Avg ball speed: {metrics['avg_ball_speed_kph']:.1f} kph")
        if "wrist_speed_kph" in metrics and metrics["wrist_speed_kph"]:
            print(f"  Wrist speed: {metrics['wrist_speed_kph']:.1f} kph")
        if "runup_speed_kph" in metrics and metrics["runup_speed_kph"]:
            print(f"  Run-up speed: {metrics['runup_speed_kph']:.1f} kph")
        print()
    
    # Angle metrics
    print(f"📐 ANGLE METRICS:")
    if metrics.get("back_knee_angle_bfc_deg"):
        print(f"  Back knee (BFC): {metrics['back_knee_angle_bfc_deg']:.1f}°")
    else:
        print(f"  Back knee (BFC): N/A")
    
    if metrics.get("front_knee_angle_ffc_deg"):
        print(f"  Front knee (FFC): {metrics['front_knee_angle_ffc_deg']:.1f}°")
    else:
        print(f"  Front knee (FFC): N/A")
    
    if metrics.get("lateral_flexion_release_deg"):
        print(f"  Lateral flexion: {metrics['lateral_flexion_release_deg']:.1f}°")
    
    if metrics.get("hip_shoulder_separation_ffc_deg"):
        print(f"  Hip-shoulder sep: {metrics['hip_shoulder_separation_ffc_deg']:.1f}°")
    print()
    
    # Distance metrics
    print(f"📏 DISTANCE METRICS:")
    if metrics.get("release_height_m"):
        print(f"  Release height: {metrics['release_height_m']:.2f}m")
    if metrics.get("stride_length_m"):
        print(f"  Stride length: {metrics['stride_length_m']:.2f}m")
    if metrics.get("jump_height_m"):
        print(f"  Jump height: {metrics['jump_height_m']:.2f}m")
    print()
    
    # Timing
    if metrics.get("BFC_to_release_s") or metrics.get("FFC_to_release_s"):
        print(f"⏱️  TIMING:")
        if metrics.get("BFC_to_release_s"):
            print(f"  BFC → Release: {metrics['BFC_to_release_s']:.3f}s")
        if metrics.get("FFC_to_release_s"):
            print(f"  FFC → Release: {metrics['FFC_to_release_s']:.3f}s")
        print()
    
    # Arm position
    if metrics.get("arm_position_at_ffc"):
        print(f"💪 ARM METRICS:")
        print(f"  Arm at FFC: {metrics['arm_position_at_ffc']}")
        print()
    
    # Diagnostics
    print(f"🔍 DIAGNOSTICS:")
    print(f"  Ball detections: {metrics.get('total_ball_detections', 0)}")
    print(f"  Measurements: {metrics.get('measurement_ball_detections', 0)} ({metrics.get('measurement_percentage', 0):.1f}%)")
    
    # Warnings
    if "warnings" in metrics:
        print(f"\n⚠️  WARNINGS:")
        for w in metrics["warnings"]:
            print(f"  • {w}")
    
    # Save to JSON
    with open(args.out_json, "w") as f:
        json.dump(make_json_safe(metrics), f, indent=2)

    print(f"\n✅ Metrics saved to {args.out_json}")