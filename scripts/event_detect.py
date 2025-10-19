# event_detect_smart.py - Smart BFC/FFC detection based on release frame

import numpy as np
from metrics_util import smooth

L_ANK = 15
R_ANK = 16


def _extract_ankle_y(kpts_all, idx):
    """Extract Y coordinates for a specific ankle across all frames."""
    ys = []
    for k in kpts_all:
        if k is None:
            ys.append(np.nan)
        else:
            ys.append(float(k[idx, 1]))
    return np.array(ys, dtype=float)


def smooth_series(y, win=5):
    """Smooth a series to reduce jitter. Using smaller window for better peak detection."""
    import pandas as pd
    return smooth(pd.Series(y), win=win).to_numpy()


def find_local_maxima(y, window=3):
    """
    Find local maxima (peaks) in ankle Y data.
    Higher Y = more grounded = peak indicates ground contact.
    
    Args:
        y: Ankle Y positions
        window: Frames on each side to compare
    
    Returns:
        List of (frame_idx, y_value) for peaks
    """
    peaks = []
    n = len(y)
    
    for i in range(window, n - window):
        if np.isnan(y[i]):
            continue
        
        # Check if this point is higher than neighbors
        is_peak = True
        for j in range(-window, window + 1):
            if j == 0:
                continue
            if np.isnan(y[i + j]):
                continue
            if y[i + j] >= y[i]:
                is_peak = False
                break
        
        if is_peak:
            peaks.append((i, y[i]))
    
    return peaks


def detect_bfc_ffc_from_release(kpts_all, release_frame, fps=30.0, handed="right"):
    """
    Detect BFC and FFC by looking backwards from release frame.
    
    Strategy:
    1. Start from release frame
    2. Look backwards 0.2-0.6 seconds for delivery stride contacts
    3. Find peaks in ankle Y (peaks = ground contacts)
    4. Back foot contact (BFC) should be ~0.3-0.5s before release
    5. Front foot contact (FFC) should be ~0.1-0.2s before release
    
    Args:
        kpts_all: List of keypoint arrays
        release_frame: Frame number of ball release
        fps: Frames per second
        handed: "right" or "left" bowling arm
    
    Returns:
        (bfc_idx, ffc_idx, info)
    """
    if release_frame is None:
        print("[WARN] No release frame provided, cannot detect BFC/FFC")
        return None, None, {}
    
    # Extract ankle data
    yL = _extract_ankle_y(kpts_all, L_ANK)
    yR = _extract_ankle_y(kpts_all, R_ANK)
    
    # Smooth to reduce noise
    yL_smooth = smooth_series(yL, win=5)
    yR_smooth = smooth_series(yR, win=5)
    
    # Determine back vs front foot
    if handed.lower() == "right":
        back_y = yR_smooth
        front_y = yL_smooth
        back_letter = "R"
    else:
        back_y = yL_smooth
        front_y = yR_smooth
        back_letter = "L"
    
    # Define search window (look backwards from release)
    # For fast bowling: BFC is ~0.3-0.5s before release, FFC is ~0.1-0.25s before release
    search_start = max(0, release_frame - int(0.6 * fps))  # 0.6s before release
    search_end = release_frame - int(0.05 * fps)  # Stop 0.05s before release
    
    if search_start >= search_end:
        print(f"[WARN] Invalid search window: {search_start} to {search_end}")
        return None, None, {}
    
    print(f"[INFO] Searching for BFC/FFC in frames {search_start}-{search_end} (release at {release_frame})")
    
    # Find peaks (local maxima) in search window
    back_peaks = []
    front_peaks = []
    
    for i in range(search_start, search_end):
        # Check back ankle
        if not np.isnan(back_y[i]):
            # Is this a local maximum?
            window = 3
            is_peak = True
            for j in range(-window, window + 1):
                if j == 0 or i + j < search_start or i + j >= search_end:
                    continue
                if not np.isnan(back_y[i + j]) and back_y[i + j] >= back_y[i]:
                    is_peak = False
                    break
            if is_peak and back_y[i] > np.nanpercentile(back_y[search_start:search_end], 70):
                back_peaks.append((i, back_y[i]))
        
        # Check front ankle
        if not np.isnan(front_y[i]):
            window = 3
            is_peak = True
            for j in range(-window, window + 1):
                if j == 0 or i + j < search_start or i + j >= search_end:
                    continue
                if not np.isnan(front_y[i + j]) and front_y[i + j] >= front_y[i]:
                    is_peak = False
                    break
            if is_peak and front_y[i] > np.nanpercentile(front_y[search_start:search_end], 70):
                front_peaks.append((i, front_y[i]))
    
    print(f"[INFO] Found {len(back_peaks)} back foot peaks, {len(front_peaks)} front foot peaks")
    
    if not back_peaks or not front_peaks:
        print("[WARN] Could not find peaks for both feet")
        return None, None, {}
    
    # Select the LAST peaks (closest to release) that satisfy timing constraints
    bfc_idx = None
    ffc_idx = None
    
    # Sort peaks by frame
    back_peaks.sort(key=lambda x: x[0])
    front_peaks.sort(key=lambda x: x[0])
    
    # For fast bowling:
    # - BFC should be 0.25-0.50s before release (7-15 frames at 30fps)
    # - FFC should be 0.08-0.25s before release (2-7 frames at 30fps)
    # - FFC must come after BFC
    
    bfc_min_frames = int(0.25 * fps)  # At least 0.25s before release
    bfc_max_frames = int(0.50 * fps)  # At most 0.50s before release
    
    ffc_min_frames = int(0.08 * fps)  # At least 0.08s before release
    ffc_max_frames = int(0.25 * fps)  # At most 0.25s before release
    
    # Find BFC: last back foot peak in the valid range
    for frame, y_val in reversed(back_peaks):
        frames_before = release_frame - frame
        if bfc_min_frames <= frames_before <= bfc_max_frames:
            bfc_idx = frame
            print(f"[INFO] BFC detected at frame {bfc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
            break
    
    # Find FFC: last front foot peak in the valid range AND after BFC
    for frame, y_val in reversed(front_peaks):
        frames_before = release_frame - frame
        if ffc_min_frames <= frames_before <= ffc_max_frames:
            if bfc_idx is None or frame > bfc_idx:  # FFC must be after BFC
                ffc_idx = frame
                print(f"[INFO] FFC detected at frame {ffc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
                break
    
    # Validate the pair
    if bfc_idx is not None and ffc_idx is not None:
        if ffc_idx <= bfc_idx:
            print(f"[WARN] FFC ({ffc_idx}) is not after BFC ({bfc_idx}), invalidating")
            ffc_idx = None
        else:
            gap_frames = ffc_idx - bfc_idx
            gap_s = gap_frames / fps
            print(f"[INFO] BFC→FFC gap: {gap_frames} frames ({gap_s:.3f}s)")
            
            # Sanity check: gap should be reasonable (0.05-0.30s for fast bowling)
            if gap_s < 0.05 or gap_s > 0.35:
                print(f"[WARN] BFC→FFC gap ({gap_s:.3f}s) seems unusual")
    
    return (
        int(bfc_idx) if bfc_idx is not None else None,
        int(ffc_idx) if ffc_idx is not None else None,
        {"bfc_foot": back_letter}
    )


def detect_bfc_ffc(kpts_all, fps=30.0, handed="right", release_frame=None):
    """
    Main entry point for BFC/FFC detection.
    
    If release_frame is provided, uses smart backwards search.
    Otherwise falls back to old method (not recommended).
    """
    if release_frame is not None:
        return detect_bfc_ffc_from_release(kpts_all, release_frame, fps, handed)
    else:
        print("[WARN] No release frame provided. BFC/FFC detection will be unreliable.")
        print("[WARN] Please provide release_frame for accurate detection.")
        return None, None, {}


if __name__ == "__main__":
    print("Event detection module loaded successfully")
    print("Use detect_bfc_ffc(kpts_all, fps, handed, release_frame) to detect events")