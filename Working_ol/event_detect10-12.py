# event_detect.py  (drop-in replacement)

import numpy as np
from metrics_util import smooth

L_ANK = 15
R_ANK = 16

def _extract_ankle_y(kpts_all, idx):
    ys = []
    for k in kpts_all:
        if k is None:
            ys.append(np.nan)
        else:
            ys.append(float(k[idx, 1]))
    return np.array(ys, dtype=float)

def smooth_series(y, win=7):
    """A bit more smoothing than before to kill jitter."""
    import pandas as pd
    return smooth(pd.Series(y), win=win).to_numpy()

def _contact_segments(y, vel_thresh=1.0, min_len=3, ground_pct=60):
    """
    Return list of (start_idx, end_idx) for likely foot-ground contact segments.
    Heuristic: low vertical speed AND near bottom of frame (larger y).
    """
    y_s = smooth_series(y, win=7)
    dy = np.gradient(y_s)  # px/frame
    speed_ok = np.abs(dy) < vel_thresh

    valid = np.isfinite(y_s)
    y_thresh = np.nanpercentile(y_s[valid], ground_pct) if valid.any() else np.nan
    y_ok = y_s >= y_thresh if np.isfinite(y_thresh) else np.zeros_like(y_s, dtype=bool)

    mask = speed_ok & y_ok & valid
    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs

def _pick_pair_with_realistic_gap(segs_b, segs_f, fps, gap_min=0.10, gap_max=0.60):
    """
    Choose a (BFC, FFC) where FFC starts within [gap_min, gap_max] seconds after BFC.
    Prefer the pair nearest to the middle of that window; fallback to closest-gap pair.
    """
    if not segs_b:
        return None, None
    best = None
    best_err = 1e9
    for (sb, eb) in segs_b:
        tb = sb / fps
        for (sf, ef) in segs_f:
            tf = sf / fps
            if tf >= tb:
                gap = tf - tb
                if gap_min <= gap <= gap_max:
                    err = abs(gap - (gap_min + gap_max) * 0.5)
                    if err < best_err:
                        best_err = err
                        best = (sb, sf)
    # fallback: closest forward-starting segment
    if best is None:
        best_gap = 1e9
        for (sb, eb) in segs_b:
            tb = sb / fps
            for (sf, ef) in segs_f:
                tf = sf / fps
                if tf >= tb:
                    gap = tf - tb
                    if gap < best_gap:
                        best_gap = gap
                        best = (sb, sf)
    return (best if best else (segs_b[0][0], None))  # at least return BFC

def detect_bfc_ffc(kpts_all, fps=30.0, handed="right"):
    """
    Detect BFC and FFC using ankle motion.
    Returns (bfc_idx, ffc_idx, info) where indices are into kpts_all.
    info: {"bfc_foot": 'L' or 'R'}
    """
    yL = _extract_ankle_y(kpts_all, L_ANK)
    yR = _extract_ankle_y(kpts_all, R_ANK)

    # contact segments for each ankle
    segsL = _contact_segments(yL, vel_thresh=1.0, min_len=3, ground_pct=60)
    segsR = _contact_segments(yR, vel_thresh=1.0, min_len=3, ground_pct=60)

    # Map back vs front based on handedness (R-arm -> back=R, front=L)
    if handed.lower() == "right":
        back_segs, front_segs = segsR, segsL
        back_letter = "R"
    else:
        back_segs, front_segs = segsL, segsR
        back_letter = "L"

    # If nothing found at all, fallback to deepest ankle minimum
    if not back_segs and not front_segs:
        minL = np.nanmin(yL) if np.isfinite(np.nanmin(yL)) else np.inf
        minR = np.nanmin(yR) if np.isfinite(np.nanmin(yR)) else np.inf
        if minL < minR and np.isfinite(minL):
            return int(np.nanargmin(yL)), None, {"bfc_foot": "L"}
        elif np.isfinite(minR):
            return int(np.nanargmin(yR)), None, {"bfc_foot": "R"}
        else:
            return None, None, {}

    # If only back foot segments exist → choose the last one (closest to release)
    if back_segs and not front_segs:
        bfc_idx = int(back_segs[-1][0])
        return bfc_idx, None, {"bfc_foot": back_letter}

    # If only front foot segments exist → we can’t infer BFC confidently, pick earliest
    if front_segs and not back_segs:
        return int(front_segs[0][0]), None, {"bfc_foot": ("L" if handed.lower()=="left" else "R")}

    # Both exist → choose a realistic pair
    bfc_idx, ffc_idx = _pick_pair_with_realistic_gap(back_segs, front_segs, fps)
    return (int(bfc_idx) if bfc_idx is not None else None,
            int(ffc_idx) if ffc_idx is not None else None,
            {"bfc_foot": back_letter})
