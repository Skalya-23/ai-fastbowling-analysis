# event_detect.py - Robust BFC/FFC detection for multiple camera angles

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


def smooth_series(y, win=3):
    """Smooth a series to reduce jitter."""
    import pandas as pd
    return smooth(pd.Series(y), win=win).to_numpy()


def find_peaks_in_window(y, search_start, search_end, percentile_threshold=50):
    """
    Find local maxima (peaks) in ankle Y data within search window.
    Lower percentile threshold to catch more peaks.
    
    Args:
        y: Ankle Y positions
        search_start: Start frame (inclusive)
        search_end: End frame (exclusive)
        percentile_threshold: Minimum percentile to qualify as peak (default 50)
    
    Returns:
        List of (frame_idx, y_value) for peaks
    """
    peaks = []
    window = 3  # Check 3 frames on each side
    
    # Calculate threshold
    y_slice = y[search_start:search_end]
    valid_y = y_slice[~np.isnan(y_slice)]
    if len(valid_y) == 0:
        return peaks
    
    threshold = np.percentile(valid_y, percentile_threshold)
    
    for i in range(search_start, search_end):
        if np.isnan(y[i]):
            continue
        
        # Skip if below threshold
        if y[i] < threshold:
            continue
        
        # Check if local maximum
        is_peak = True
        for j in range(-window, window + 1):
            if j == 0:
                continue
            neighbor_idx = i + j
            # Allow checking outside search window for neighbor comparison
            if neighbor_idx < 0 or neighbor_idx >= len(y):
                continue
            if np.isnan(y[neighbor_idx]):
                continue
            if y[neighbor_idx] >= y[i]:
                is_peak = False
                break
        
        if is_peak:
            peaks.append((i, y[i]))
    
    return peaks


def detect_bfc_ffc_from_release(kpts_all, release_frame, fps=30.0, handed="right"):
    """
    Detect BFC and FFC by looking backwards from release frame.
    
    Uses peak detection (maximum ankle Y = most grounded position).
    Works for both side and back camera angles.
    
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
    
    # Extract ankle data (use RAW data for peak detection to avoid smoothing artifacts)
    yL = _extract_ankle_y(kpts_all, L_ANK)
    yR = _extract_ankle_y(kpts_all, R_ANK)
    
    # Determine back vs front foot (use raw data for more precise peak detection)
    if handed.lower() == "right":
        back_y = yR
        front_y = yL
        back_letter = "R"
        front_letter = "L"
    else:
        back_y = yL
        front_y = yR
        back_letter = "L"
        front_letter = "R"
    
    # Define search window (wide range to catch all bowling styles)
    search_start = max(0, release_frame - int(0.8 * fps))  # 0.8s before release
    search_end = release_frame - int(0.02 * fps)  # Stop 0.02s before release
    
    if search_start >= search_end:
        print(f"[WARN] Invalid search window: {search_start} to {search_end}")
        return None, None, {}
    
    print(f"[INFO] Searching for BFC/FFC in frames {search_start}-{search_end} (release at {release_frame})")
    
    # Find peaks with lower threshold (50th percentile instead of 70th)
    back_peaks = find_peaks_in_window(back_y, search_start, search_end, percentile_threshold=50)
    front_peaks = find_peaks_in_window(front_y, search_start, search_end, percentile_threshold=50)
    
    print(f"[INFO] Found {len(back_peaks)} back foot peaks, {len(front_peaks)} front foot peaks")
    
    if back_peaks:
        print(f"[DEBUG] Back foot peaks at frames: {[p[0] for p in back_peaks]}")
    if front_peaks:
        print(f"[DEBUG] Front foot peaks at frames: {[p[0] for p in front_peaks]}")
    
    if not back_peaks and not front_peaks:
        print("[WARN] No peaks found for either foot")
        return None, None, {}
    
    # Sort peaks by frame
    back_peaks.sort(key=lambda x: x[0])
    front_peaks.sort(key=lambda x: x[0])
    
    # WIDE timing constraints to accommodate different bowling styles
    # BFC: 0.15-0.80s before release
    # FFC: 0.02-0.40s before release
    bfc_min_frames = int(0.15 * fps)
    bfc_max_frames = int(0.80 * fps)
    ffc_min_frames = int(0.02 * fps)
    ffc_max_frames = int(0.40 * fps)
    
    print(f"[DEBUG] BFC timing window: {bfc_min_frames}-{bfc_max_frames} frames ({bfc_min_frames/fps:.2f}-{bfc_max_frames/fps:.2f}s)")
    print(f"[DEBUG] FFC timing window: {ffc_min_frames}-{ffc_max_frames} frames ({ffc_min_frames/fps:.2f}-{ffc_max_frames/fps:.2f}s)")
    
    # Find BFC: last back foot peak in valid range
    bfc_idx = None
    for frame, y_val in reversed(back_peaks):
        frames_before = release_frame - frame
        if bfc_min_frames <= frames_before <= bfc_max_frames:
            bfc_idx = frame
            print(f"[INFO] BFC detected at frame {bfc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
            break
    
    # If no BFC found in timing window, take the highest peak
    if bfc_idx is None and back_peaks:
        # Find peak with maximum Y value (most grounded)
        best_peak = max(back_peaks, key=lambda x: x[1])
        bfc_idx = best_peak[0]
        frames_before = release_frame - bfc_idx
        print(f"[INFO] BFC detected at frame {bfc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
        print(f"       [Note: Selected by maximum Y, outside preferred timing window]")
    
    # Find FFC: last front foot peak in valid range AND after BFC
    ffc_idx = None
    for frame, y_val in reversed(front_peaks):
        frames_before = release_frame - frame
        if ffc_min_frames <= frames_before <= ffc_max_frames:
            if bfc_idx is None or frame > bfc_idx:
                ffc_idx = frame
                print(f"[INFO] FFC detected at frame {ffc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
                break
    
    # If no FFC in timing window, find highest peak after BFC
    if ffc_idx is None and front_peaks:
        candidates = [p for p in front_peaks if bfc_idx is None or p[0] > bfc_idx]
        if candidates:
            best_peak = max(candidates, key=lambda x: x[1])
            ffc_idx = best_peak[0]
            frames_before = release_frame - ffc_idx
            print(f"[INFO] FFC detected at frame {ffc_idx} ({frames_before} frames / {frames_before/fps:.3f}s before release)")
            print(f"       [Note: Selected by maximum Y, outside preferred timing window]")
    
    # Validate the pair
    if bfc_idx is not None and ffc_idx is not None:
        if ffc_idx <= bfc_idx:
            print(f"[WARN] FFC ({ffc_idx}) is not after BFC ({bfc_idx}), invalidating FFC")
            ffc_idx = None
        else:
            gap_frames = ffc_idx - bfc_idx
            gap_s = gap_frames / fps
            print(f"[INFO] BFC→FFC gap: {gap_frames} frames ({gap_s:.3f}s)")
            
            if gap_s < 0.05:
                print(f"[WARN] Gap very short - verify visually")
            if gap_s > 0.5:
                print(f"[WARN] Gap unusually long - verify visually")
    
    if bfc_idx is None:
        print("[WARN] No valid BFC found")
    if ffc_idx is None:
        print("[WARN] No valid FFC found")
    
    return (
        int(bfc_idx) if bfc_idx is not None else None,
        int(ffc_idx) if ffc_idx is not None else None,
        {"bfc_foot": back_letter, "ffc_foot": front_letter}
    )


def detect_bfc_ffc(kpts_all, fps=30.0, handed="right", release_frame=None):
    """
    Main entry point for BFC/FFC detection.
    
    If release_frame is provided, uses smart backwards search.
    Otherwise returns None for both.
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