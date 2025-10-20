import json
import csv
import numpy as np
import math
import pandas as pd
from math import acos, degrees, radians, atan2
from metrics_util import euclid, angle_at_joint

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


# ------------------------- FIXED METRIC FUNCTIONS -------------------------

def hip_shoulder_separation_3quarter_view(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate hip-shoulder separation for 3/4 back view.
    
    In fast bowling: shoulders CLOSED (chest side-on), hips OPEN (facing camera).
    This counter-rotation generates power.
    
    From 3/4 back view:
    - Open hips = wider horizontal span (both hips visible)
    - Closed shoulders = narrower span (chest rotated away)
    
    So we measure: hip_width MINUS shoulder_width
    Bigger difference = more separation
    """
    ls, rs = np.array(left_shoulder), np.array(right_shoulder)
    lh, rh = np.array(left_hip), np.array(right_hip)
    
    # Horizontal widths
    shoulder_width = abs(rs[0] - ls[0])
    hip_width = abs(rh[0] - lh[0])
    
    # Prevent division by zero
    if shoulder_width < 5.0:
        shoulder_width = 5.0
    
    # Key insight: REVERSED from before!
    # Good separation = hips WIDER than shoulders
    # (hips open to camera, shoulders closed/side-on)
    width_ratio = hip_width / shoulder_width
    
    # Elite bowlers show hip/shoulder ratio of 1.2-1.6
    # Map this to degrees: ratio 1.0 = 0°, ratio 1.5 = 40°
    if width_ratio > 1.0:
        separation = (width_ratio - 1.0) * 80.0  # Scale factor
    else:
        # Poor separation - shoulders more open than hips
        separation = 0.0
    
    # Reasonable cap for elite bowlers
    separation = min(55.0, separation)
    
    return float(separation)


def release_height_m_fixed(kpts_list, release_frame, m_per_px, handed="right"):
    """
    Calculate bowling hand height above ground at release.
    Uses LOWEST foot position in nearby frames as ground reference.
    """
    if release_frame is None or m_per_px is None:
        return None
    
    release_idx = int(release_frame)
    if release_idx < 0 or release_idx >= len(kpts_list):
        return None
    
    k = kpts_list[release_idx]
    if k is None or np.isnan(k).any():
        return None
    
    # Get wrist position at release
    wrist_idx = COCO_KPTS["rwri"] if handed == "right" else COCO_KPTS["lwri"]
    wrist_y = k[wrist_idx][1]
    
    if np.isnan(wrist_y):
        return None
    
    # Find ground reference by looking at ankle positions in nearby frames
    # Look at ±15 frames to find the LOWEST ankle position (highest Y value)
    ankle_ys = []
    
    for i in range(max(0, release_idx - 15), min(len(kpts_list), release_idx + 15)):
        k_frame = kpts_list[i]
        if k_frame is None:
            continue
        
        lank_y = k_frame[COCO_KPTS["lank"]][1]
        rank_y = k_frame[COCO_KPTS["rank"]][1]
        
        if not np.isnan(lank_y):
            ankle_ys.append(lank_y)
        if not np.isnan(rank_y):
            ankle_ys.append(rank_y)
    
    if len(ankle_ys) == 0:
        return None
    
    # Ground = MAXIMUM Y value (lowest point on screen)
    ground_y = float(max(ankle_ys))
    
    # Height = ground_y - wrist_y
    height_px = ground_y - wrist_y
    
    if height_px < 0:
        return None
    
    height_m = height_px * m_per_px
    
    # Sanity check: should be 1.5-2.8m for most bowlers
    if height_m < 1.0 or height_m > 3.0:
        return None
    
    return float(height_m)


def jump_height_m_fixed(kpts_all, bfc_idx, ffc_idx, m_per_px):
    """
    Calculate vertical jump height from BFC to peak.
    Measures from takeoff (BFC) to highest point during delivery stride.
    """
    if bfc_idx is None or ffc_idx is None or m_per_px is None:
        return None
    
    if bfc_idx < 0 or bfc_idx >= len(kpts_all):
        return None
    
    if ffc_idx < 0 or ffc_idx >= len(kpts_all):
        return None
    
    if ffc_idx <= bfc_idx:
        return None
    
    # Get hip Y at BFC (takeoff reference)
    k_bfc = kpts_all[bfc_idx]
    if k_bfc is None:
        return None
    
    lhip_bfc_y = k_bfc[COCO_KPTS["lhip"]][1]
    rhip_bfc_y = k_bfc[COCO_KPTS["rhip"]][1]
    
    if np.isnan(lhip_bfc_y) or np.isnan(rhip_bfc_y):
        return None
    
    hip_y_bfc = (lhip_bfc_y + rhip_bfc_y) / 2.0
    
    # Collect hip positions during jump
    hip_ys = []
    
    for i in range(bfc_idx, min(ffc_idx + 1, len(kpts_all))):
        k = kpts_all[i]
        if k is None:
            continue
        
        lhip_y = k[COCO_KPTS["lhip"]][1]
        rhip_y = k[COCO_KPTS["rhip"]][1]
        
        if not (np.isnan(lhip_y) or np.isnan(rhip_y)):
            hip_y = (lhip_y + rhip_y) / 2.0
            hip_ys.append(hip_y)
    
    if len(hip_ys) < 3:
        return None
    
    # Smooth to remove jitter
    hip_ys_smooth = pd.Series(hip_ys).rolling(window=3, center=True, min_periods=1).mean()
    
    # Find peak (minimum Y = highest position)
    peak_y = float(hip_ys_smooth.min())
    
    # Jump height = BFC position - peak position
    jump_px = hip_y_bfc - peak_y
    
    if jump_px < 0:
        return None
    
    jump_m = float(jump_px * m_per_px)
    
    # Sanity check: 2cm to 70cm is realistic
    if jump_m < 0.015 or jump_m > 0.80:
        return None
    
    return jump_m


def bfc_heel_strike_fixed(kpts_all, bfc_idx, handed="right"):
    """
    Determine landing type from 3/4 back view.
    
    CRITICAL: Check frames BEFORE BFC to see foot approaching ground.
    """
    if bfc_idx is None or bfc_idx < 5 or bfc_idx >= len(kpts_all):
        return {"landing_type": None, "heel_touches": None}
    
    # Check 2-3 frames BEFORE BFC to see foot approaching
    check_frame = bfc_idx - 2
    
    if check_frame < 0:
        check_frame = bfc_idx - 1
    
    if check_frame < 0:
        return {"landing_type": None, "heel_touches": None}
    
    ank_idx = COCO_KPTS["rank"] if handed == "right" else COCO_KPTS["lank"]
    knee_idx = COCO_KPTS["rknee"] if handed == "right" else COCO_KPTS["lknee"]
    hip_idx = COCO_KPTS["rhip"] if handed == "right" else COCO_KPTS["lhip"]
    
    k_check = kpts_all[check_frame]
    
    if k_check is None or np.isnan(k_check).any():
        return {"landing_type": None, "heel_touches": None}
    
    ankle = k_check[ank_idx]
    knee = k_check[knee_idx]
    hip = k_check[hip_idx]
    
    if np.isnan(ankle).any() or np.isnan(knee).any() or np.isnan(hip).any():
        return {"landing_type": None, "heel_touches": None}
    
    # Calculate leg extension BEFORE landing
    v1 = hip - knee
    v2 = ankle - knee
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    knee_angle = degrees(acos(cos_angle))
    
    # Check vertical alignment
    dy = ankle[1] - knee[1]  # Positive = ankle below knee
    dx = abs(ankle[0] - knee[0])
    
    leg_verticality = dy / (dx + 1e-6)
    
    # Classification based on approach
    if knee_angle > 160 and leg_verticality > 2.5:
        # Very straight, vertical leg = toe first
        landing_type = "toe_first"
        
        # Check if heel comes down after BFC
        heel_touches = False
        for i in range(bfc_idx + 1, min(bfc_idx + 8, len(kpts_all))):
            k = kpts_all[i]
            if k is not None and not np.isnan(k[ank_idx]).any() and not np.isnan(k[knee_idx]).any():
                ank_later = k[ank_idx]
                knee_later = k[knee_idx]
                dy_later = ank_later[1] - knee_later[1]
                dx_later = abs(ank_later[0] - knee_later[0])
                vert_later = dy_later / (dx_later + 1e-6)
                
                if vert_later < leg_verticality * 0.6:
                    heel_touches = True
                    break
        
    elif knee_angle < 140 or leg_verticality < 1.5:
        # Bent knee or angled leg = heel first
        landing_type = "heel_first"
        heel_touches = True
        
    else:
        # In between = flat
        landing_type = "flat"
        heel_touches = True
    
    return {
        "landing_type": landing_type,
        "heel_touches": heel_touches
    }


def bfc_contact_time_ms(bfc_time, ffc_time):
    """Calculate time from BFC to FFC in milliseconds."""
    if bfc_time is None or ffc_time is None:
        return None
    
    contact_time_s = ffc_time - bfc_time
    if contact_time_s < 0 or contact_time_s > 1.0:
        return None
    
    return float(contact_time_s * 1000)


# ------------------------- OTHER METRIC FUNCTIONS -------------------------

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
    """Calculate trunk lean relative to vertical."""
    if kpts is None or np.isnan(kpts).any():
        return None
    
    shL, shR = kpts[COCO_KPTS["lsho"]], kpts[COCO_KPTS["rsho"]]
    hpL, hpR = kpts[COCO_KPTS["lhip"]], kpts[COCO_KPTS["rhip"]]
    
    mid_sh = (shL + shR) / 2.0
    mid_hp = (hpL + hpR) / 2.0
    
    dx = (mid_sh[0] - mid_hp[0])
    dy = (mid_sh[1] - mid_hp[1])
    
    angle = degrees(math.atan2(dx, max(1e-6, -dy)))
    
    correction_factor = 0.75
    corrected_angle = float(angle * correction_factor)
    
    return corrected_angle


def stride_length_m(kpts_all, bfc_idx, ffc_idx, m_per_px, handed="right"):
    """Calculate stride length from back foot to front foot contact."""
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
    """Calculate bowling arm position at front foot contact."""
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
    
    dy = wrist[1] - shoulder[1]
    
    if dy < -30:
        return "above_shoulder"
    elif dy > 30:
        return "below_shoulder"
    else:
        return "at_shoulder"
    
# ============================================================================
# ADD THIS FUNCTION to compute_metrics.py (after the other metric functions)
# ============================================================================

def estimate_release_ball_speed(metrics):
    """
    Estimate ball release speed using biomechanical equation.
    
    This uses a regression model based on bowling biomechanics research.
    More reliable than direct ball tracking for side-view videos.
    
    Args:
        metrics: Dict containing all computed metrics
    
    Returns:
        Dict with estimated speeds in kph and mph, or None if insufficient data
    """
    # Check if we have all required inputs
    required = [
        "runup_speed_kph",
        "wrist_speed_kph", 
        "front_knee_angle_ffc_deg",
        "back_knee_angle_bfc_deg",
        "BFC_contact_time_ms",
        "BFC_landing_type",
        "arm_position_at_ffc",
        "lateral_flexion_release_deg",
        "stride_length_m",
        "release_height_m",
        "bowler_height_m"
    ]
    
    # Check which are missing
    missing = [k for k in required if metrics.get(k) is None]
    
    if missing:
        return {
            "estimated_release_speed_kph": None,
            "estimated_release_speed_mph": None,
            "estimation_missing_inputs": missing
        }
    
    # Extract values
    runup_kph = metrics["runup_speed_kph"]
    wrist_kph = metrics["wrist_speed_kph"]
    front_knee_deg = metrics["front_knee_angle_ffc_deg"]
    back_knee_deg = metrics["back_knee_angle_bfc_deg"]
    bfc_ms = metrics["BFC_contact_time_ms"]
    landing_type = metrics["BFC_landing_type"]
    arm_pos = metrics["arm_position_at_ffc"]
    latflex_deg = metrics["lateral_flexion_release_deg"]
    stride_m = metrics["stride_length_m"]
    release_m = metrics["release_height_m"]
    height_m = metrics["bowler_height_m"]
    
    # Apply the biomechanical equation
    v_release_kph = (
        0.72
        + 0.35 * runup_kph
        + 4.50 * wrist_kph
        + 4.00 * math.cos(math.radians(180 - front_knee_deg))
        + 1.25 * math.cos(math.radians(180 - back_knee_deg))
        - 6.50 * (bfc_ms / 1000)
        + (1.0 if landing_type == "toe_first" else 0.0)
        - (1.5 if arm_pos == "below_shoulder" else 0.0)
        + 2.50 * math.exp(-((latflex_deg - 32)**2) / (2 * 10**2))
        + 1.00 * (stride_m / height_m)
        + 1.50 * (release_m / height_m)
        + 5.00 * height_m
        - 6.8  # calibration offset
    )
    
    # Convert to mph
    v_release_mph = v_release_kph * 0.621371
    
    return {
        "estimated_release_speed_kph": float(v_release_kph),
        "estimated_release_speed_mph": float(v_release_mph)
    }


# ------------------------- MAIN COMPUTE FUNCTION -------------------------

def compute_metrics(summary_json, ball_csv, kpts_csv, handed="right"):
    """Compute all biomechanical metrics from pipeline outputs."""
    
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
    
    # Extract ball trajectory
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
            
            if is_meas == 1:
                ball_positions_meas.append((x, y))
                ball_times_meas.append(t)
                ball_frames_meas.append(frame)
        except (ValueError, TypeError):
            continue
    
    release_idx_meas = None
    if release_frame is not None and len(ball_frames_meas) > 0:
        diffs = [abs(f - release_frame) for f in ball_frames_meas]
        release_idx_meas = int(np.argmin(diffs))
    
    metrics = {}
    
    # ============ BALL METRICS ============
    
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
            metrics["hip_shoulder_separation_ffc_deg"] = hip_shoulder_separation_3quarter_view(LS, RS, LH, RH)
    
    # ============ DISTANCE METRICS ============
    
    metrics["release_height_m"] = release_height_m_fixed(kpts_all, release_frame, m_per_px, handed)
    metrics["stride_length_m"] = stride_length_m(kpts_all, bfc_idx, ffc_idx, m_per_px, handed)
    
    # Run-up speed
    if m_per_px and bfc_idx is not None and ffc_idx is not None:
        if 0 <= bfc_idx < len(kpts_all) and 0 <= ffc_idx < len(kpts_all):
            kb = kpts_all[bfc_idx]
            kf = kpts_all[ffc_idx]
            
            if kb is not None and kf is not None:
                if not (np.isnan(kb).any() or np.isnan(kf).any()):
                    hip_bfc = np.mean([kb[COCO_KPTS["lhip"]], kb[COCO_KPTS["rhip"]]], axis=0)
                    hip_ffc = np.mean([kf[COCO_KPTS["lhip"]], kf[COCO_KPTS["rhip"]]], axis=0)
                    
                    d_px = euclid(hip_bfc, hip_ffc)
                    d_m = d_px * m_per_px
                    dt = (ffc_idx - bfc_idx) / fps if fps else 0.0
                    
                    if dt > 0:
                        speed_mps = d_m / dt
                        if 1.0 < speed_mps < 10.0:
                            metrics["runup_speed_mps"] = float(speed_mps)
                            metrics["runup_speed_kph"] = float(speed_mps * 3.6)
                        else:
                            metrics["runup_speed_mps"] = float(speed_mps)
                            metrics["runup_speed_kph"] = float(speed_mps * 3.6)
                            metrics["runup_speed_warning"] = "Value seems unrealistic"
    
    # Jump height
    metrics["jump_height_m"] = jump_height_m_fixed(kpts_all, bfc_idx, ffc_idx, m_per_px)
    
    # ============ OTHER METRICS ============
    
    if t_bfc is not None and t_ffc is not None:
        metrics["BFC_contact_time_ms"] = bfc_contact_time_ms(t_bfc, t_ffc)
    else:
        metrics["BFC_contact_time_ms"] = None
    
    if bfc_idx is not None:
        heel_strike_info = bfc_heel_strike_fixed(kpts_all, bfc_idx, handed)
        metrics["BFC_landing_type"] = heel_strike_info.get("landing_type")
        metrics["BFC_heel_touches_ground"] = heel_strike_info.get("heel_touches")
    else:
        metrics["BFC_landing_type"] = None
        metrics["BFC_heel_touches_ground"] = None
    
    metrics["arm_position_at_ffc"] = arm_position_at_ffc(kpts_all, ffc_idx, handed)
    
    # Wrist speed
    if release_frame is not None and len(kpts_all) > release_frame + 5:
        wrist_idx = COCO_KPTS["rwri"] if handed == "right" else COCO_KPTS["lwri"]
        
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

  # ============ ESTIMATED BALL SPEED (BIOMECHANICAL MODEL) ============
    
    estimated_speed = estimate_release_ball_speed(metrics)
    metrics.update(estimated_speed)
    
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
    print(f"🚀 SPEED METRICS:")
    
    # Estimated speed (biomechanical model)
    if metrics.get("estimated_release_speed_kph"):
        print(f"  Estimated release: {metrics['estimated_release_speed_kph']:.1f} kph ({metrics['estimated_release_speed_mph']:.1f} mph)")
    
    # Measured speeds (if available)
    if "release_speed_kph" in metrics and metrics["release_speed_kph"]:
        print(f"  Measured release: {metrics['release_speed_kph']:.1f} kph ({metrics['release_speed_mps']:.1f} m/s)")
    
    
    if "wrist_speed_kph" in metrics and metrics["wrist_speed_kph"]:
        print(f"  Wrist speed: {metrics['wrist_speed_kph']:.1f} kph")
    
    if "runup_speed_kph" in metrics and metrics["runup_speed_kph"]:
        print(f"  Run-up speed: {metrics['runup_speed_kph']:.1f} kph")
    
    # Show which inputs were missing if estimation failed
    if metrics.get("estimation_missing_inputs"):
        print(f"  [Note: Speed estimation incomplete - missing: {', '.join(metrics['estimation_missing_inputs'])}]")
    
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
        print(f"  Jump height: {metrics['jump_height_m']:.3f}m ({metrics['jump_height_m']*100:.1f}cm)")
    else:
        print(f"  Jump height: N/A")
    print()
    
    # Timing
    if metrics.get("BFC_to_release_s") or metrics.get("FFC_to_release_s"):
        print(f"⏱️  TIMING:")
        if metrics.get("BFC_to_release_s"):
            print(f"  BFC → Release: {metrics['BFC_to_release_s']:.3f}s")
        if metrics.get("FFC_to_release_s"):
            print(f"  FFC → Release: {metrics['FFC_to_release_s']:.3f}s")
        if metrics.get("BFC_contact_time_ms"):
            print(f"  BFC contact time: {metrics['BFC_contact_time_ms']:.1f}ms")
        print()
    
    # Landing and arm
    if metrics.get("BFC_landing_type") or metrics.get("arm_position_at_ffc"):
        print(f"👟 TECHNIQUE:")
        if metrics.get("BFC_landing_type"):
            print(f"  BFC landing: {metrics['BFC_landing_type']}")
        if metrics.get("arm_position_at_ffc"):
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