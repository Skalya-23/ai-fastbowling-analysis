import json
import csv
import numpy as np
import math
from math import acos, degrees
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
    """
    Load keypoints from the CSV produced by combined_pipeline.py.
    Returns a list of (17x2) numpy arrays in COCO order per frame.
    Missing values become NaN.
    """
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

def instantaneous_speed(px_seq, t_seq, m_per_px):
    """Linear fit of last ~5–9 points to estimate speed in m/s."""
    if m_per_px is None or len(px_seq) < 5:
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

def back_knee_collapse_at_bfc(kpts_all, bfc_idx, handed="right"):
    if bfc_idx is None or bfc_idx < 0 or bfc_idx >= len(kpts_all):
        return None
    kpts = kpts_all[bfc_idx]
    if kpts is None or np.isnan(kpts).any():
        return None
    if handed == "right":
        hip, knee, ankle = kpts[COCO_KPTS["rhip"]], kpts[COCO_KPTS["rknee"]], kpts[COCO_KPTS["rank"]]
    else:
        hip, knee, ankle = kpts[COCO_KPTS["lhip"]], kpts[COCO_KPTS["lknee"]], kpts[COCO_KPTS["lank"]]
    return angle_three_points(hip, knee, ankle)

def lateral_flexion(kpts):
    """
    Trunk lean relative to vertical using mid-shoulders -> mid-hips.
    0° = upright; + = lean toward bowler’s right; - = lean toward bowler’s left.
    """
    if kpts is None or np.isnan(kpts).any():
        return None
    shL, shR = kpts[COCO_KPTS["lsho"]], kpts[COCO_KPTS["rsho"]]
    hpL, hpR = kpts[COCO_KPTS["lhip"]], kpts[COCO_KPTS["rhip"]]
    mid_sh = (shL + shR) / 2.0
    mid_hp = (hpL + hpR) / 2.0
    dx = (mid_sh[0] - mid_hp[0])
    dy = (mid_sh[1] - mid_hp[1])  # image y increases downward
    # Use -dy so "up" is positive; atan2(x, y) gives angle from vertical.
    angle = degrees(math.atan2(dx, max(1e-6, -dy)))
    return float(angle)

def release_height_m(kpts_list, release_idx, m_per_px, frame_h=None):
    """
    Height of the bowling hand (right wrist) above ground (bottom of frame), in meters.
    Returns None if frame_h or m_per_px are missing.
    """
    if (release_idx is None) or (release_idx < 0) or (kpts_list is None) or (m_per_px is None) or (frame_h is None):
        return None
    if release_idx >= len(kpts_list):
        return None
    k = kpts_list[release_idx]
    if k is None or np.isnan(k).any():
        return None
    wrist_y = k[COCO_KPTS["rwri"]][1]
    if np.isnan(wrist_y):
        return None
    px_from_ground = max(0.0, float(frame_h) - float(wrist_y))
    return float(px_from_ground * m_per_px)

# ------------------------- Main metrics -------------------------

def compute_metrics(summary_json, ball_csv, kpts_json, handed="right"):
    # Summary and I/O
    with open(summary_json, "r") as f:
        summary = json.load(f)
    with open(ball_csv, "r") as f:
        rows = list(csv.DictReader(f))

    # Load keypoints from CSV (path passed via --kpts_json)
    kpts_all = load_kpts_csv(kpts_json)

    fps = summary.get("fps", 30.0)
    m_per_px = summary.get("scale_m_per_px", None)
    frame_h = summary.get("frame_h", None)  # add this in combined_pipeline summary if not present

    # Extract bowler events (note: some fields may be timestamps, not durations)
    bowler_events = summary.get("bowler_events", {}) or {}
    bfc_time = bowler_events.get("BFC_time_s", None)    # often a timestamp (start)
    ffc_time = bowler_events.get("FFC_time_s", None)    # may be null
    # Optional duration if you add it in event_detect → summary:
    bfc_ms_from_summary = bowler_events.get("BFC_ms", None) or bowler_events.get("bfc_ms", None)

    # Indices (only if times exist)
    bfc_idx = int(bfc_time * fps) if (bfc_time is not None) else None
    ffc_idx = int(ffc_time * fps) if (ffc_time is not None) else None

    # --- Extract ball trajectory ---
    xs, ts = [], []
    for r in rows:
        x_str, y_str, t_str = r.get("x"), r.get("y"), r.get("t")
        if x_str is None or y_str is None or t_str is None:
            continue
        if x_str == "None" or y_str == "None":
            continue
        try:
            x = float(x_str); y = float(y_str); t = float(t_str)
        except:
            continue
        xs.append((x, y))
        ts.append(t)

    xs = np.array(xs, dtype=float) if len(xs) > 0 else np.empty((0,2))
    ts = np.array(ts, dtype=float) if len(ts) > 0 else np.empty((0,))

    # Determine release index by nearest time (robust if ball is first detected after the nominal release time)
    release_time = summary.get("release_time_s", None)
    release_idx = None
    if (release_time is not None) and (ts.size > 0):
        release_idx = int(np.argmin(np.abs(ts - release_time)))

    metrics = {}

    # --- Ball release speed ---
    if (m_per_px is not None) and (release_idx is not None) and (len(xs) >= 5):
        rel_idx = int(np.clip(release_idx, 0, len(xs)-1))
        v_mps = instantaneous_speed(xs[:rel_idx+1], ts[:rel_idx+1], m_per_px)
        metrics["release_speed_mps"] = v_mps
        metrics["release_speed_kph"] = (v_mps * 3.6) if v_mps is not None else None
        metrics["release_index_in_ball_track"] = int(rel_idx)

    # --- Average flight speed release → stumps ---
    st_y1 = summary.get("stumps_detected", {}).get("y_bot", None)
    if (m_per_px is not None) and (st_y1 is not None) and (release_idx is not None) and (len(xs) >= 5):
        rel_idx = int(np.clip(release_idx, 0, len(xs)-1))
        y_rel = xs[rel_idx][1]
        t_rel = ts[rel_idx]
        hit_t, hit_xy = None, None
        for i in range(rel_idx + 1, len(xs)):
            y_now = xs[i][1]
            # image y increases downward; look for crossing y_bot
            if (y_rel <= st_y1 <= y_now) or (y_rel >= st_y1 >= y_now) or abs(y_now - st_y1) < 3:
                hit_t, hit_xy = ts[i], xs[i]
                break
        if hit_t is not None and hit_xy is not None:
            flight_dt = float(hit_t - t_rel)
            d_px = float(math.hypot(hit_xy[0] - xs[rel_idx][0], hit_xy[1] - y_rel))
            d_m = d_px * float(m_per_px)
            if flight_dt > 0:
                v_mps = d_m / flight_dt
                metrics["avg_flight_speed_mps"] = v_mps
                metrics["avg_flight_speed_kph"] = v_mps * 3.6
                metrics["flight_dt_s"] = flight_dt
                metrics["flight_dist_m"] = d_m

    # --- Run-up speed (hip center BFC→FFC) ---
    if (m_per_px is not None) and (bfc_idx is not None) and (ffc_idx is not None):
        if 0 <= bfc_idx < len(kpts_all) and 0 <= ffc_idx < len(kpts_all):
            kb = kpts_all[bfc_idx]
            kf = kpts_all[ffc_idx]
            if kb is not None and kf is not None and not (np.isnan(kb).any() or np.isnan(kf).any()):
                hip_bfc = np.mean([kb[COCO_KPTS["lhip"]], kb[COCO_KPTS["rhip"]]], axis=0)
                hip_ffc = np.mean([kf[COCO_KPTS["lhip"]], kf[COCO_KPTS["rhip"]]], axis=0)
                d_px = euclid(hip_bfc, hip_ffc)
                d_m = d_px * float(m_per_px)
                dt = (ffc_idx - bfc_idx) / float(fps) if fps else 0.0
                metrics["runup_speed_mps"] = (d_m / dt) if dt > 0 else None

    # --- Lateral flexion at release ---
    if (release_idx is not None) and (0 <= release_idx < len(kpts_all)):
        kr = kpts_all[release_idx]
        metrics["lateral_flexion_deg"] = lateral_flexion(kr)

    # --- Back knee collapse angle at BFC ---
    metrics["back_knee_angle_bfc"] = back_knee_collapse_at_bfc(kpts_all, bfc_idx, handed)

    # --- Hip–shoulder separation at FFC ---
    if (ffc_idx is not None) and (0 <= ffc_idx < len(kpts_all)):
        kpts = kpts_all[ffc_idx]
        if kpts is not None and not np.isnan(kpts).any():
            LS, RS = kpts[COCO_KPTS["lsho"]], kpts[COCO_KPTS["rsho"]]
            LH, RH = kpts[COCO_KPTS["lhip"]], kpts[COCO_KPTS["rhip"]]
            metrics["hip_shoulder_sep_ffc"] = hip_shoulder_separation(LS, RS, LH, RH)

    # --- Release height (meters) ---
    metrics["release_height_m"] = release_height_m(kpts_all, release_idx, m_per_px, frame_h)

    # --- BFC contact time (ms) ---
    # Prefer explicit duration if provided by summary; otherwise leave None
    if bfc_ms_from_summary is not None:
        metrics["BFC_contact_ms"] = float(bfc_ms_from_summary)
    else:
        metrics["BFC_contact_ms"] = None  # duration not available from current summary

    # --- Jump height (hip vertical displacement BFC→FFC) ---
    if (m_per_px is not None) and (bfc_idx is not None) and (ffc_idx is not None):
        if 0 <= bfc_idx < len(kpts_all) and 0 <= ffc_idx < len(kpts_all) and ffc_idx >= bfc_idx:
            hips = []
            for i in range(bfc_idx, ffc_idx + 1):
                k = kpts_all[i]
                if k is None or np.isnan(k).any():
                    continue
                hips.append(np.mean([k[COCO_KPTS["lhip"]], k[COCO_KPTS["rhip"]]], axis=0))
            if len(hips) > 0:
                ys = [float(h[1]) for h in hips]
                jump_px = float(max(ys) - min(ys))
                metrics["jump_height_px"] = jump_px
                metrics["jump_height_m"] = jump_px * float(m_per_px)

    # --- Impulse stride length (hip pre-BFC → at BFC) ---
    if (m_per_px is not None) and (bfc_idx is not None) and (bfc_idx > 3) and (bfc_idx < len(kpts_all)):
        k_pre = kpts_all[bfc_idx - 3]
        k_bfc = kpts_all[bfc_idx]
        if (k_pre is not None) and (k_bfc is not None) and not (np.isnan(k_pre).any() or np.isnan(k_bfc).any()):
            hip_pre = np.mean([k_pre[COCO_KPTS["lhip"]], k_pre[COCO_KPTS["rhip"]]], axis=0)
            hip_bfc = np.mean([k_bfc[COCO_KPTS["lhip"]], k_bfc[COCO_KPTS["rhip"]]], axis=0)
            stride_px = euclid(hip_pre, hip_bfc)
            metrics["impulse_stride_length_px"] = stride_px
            metrics["impulse_stride_length_m"] = stride_px * float(m_per_px)

    return metrics

# ------------------------- CLI -------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--ball_csv", required=True)
    ap.add_argument("--kpts_json", required=True, help="Path to kpts CSV (yes, CSV).")
    ap.add_argument("--handed", choices=["right","left"], default="right")
    ap.add_argument("--out_json", default="metrics_out.json")
    args = ap.parse_args()

    metrics = compute_metrics(args.summary_json, args.ball_csv, args.kpts_json, handed=args.handed)

    with open(args.out_json, "w") as f:
        json.dump(make_json_safe(metrics), f, indent=2)

    print("[DONE] metrics saved to", args.out_json)
