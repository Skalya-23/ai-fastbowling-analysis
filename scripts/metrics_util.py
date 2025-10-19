"""
Utility functions for biomechanics metrics calculation
"""

import numpy as np
import pandas as pd
from math import atan2, degrees, acos


def euclid(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1, p2: Points as (x, y) tuples or arrays
    
    Returns:
        Distance in same units as input
    """
    p1, p2 = np.array(p1), np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def angle_at_joint(a, b, c):
    """
    Calculate angle ABC (angle at point B).
    
    Args:
        a, b, c: Points as (x, y) tuples or arrays
    
    Returns:
        Angle in degrees (0-180)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = degrees(acos(cos_angle))
    return float(angle)


def hip_shoulder_separation(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate hip-shoulder separation angle (counter-rotation).
    
    This measures how much the shoulders have rotated relative to the hips,
    which is important for generating bowling speed.
    
    Args:
        left_shoulder, right_shoulder: Shoulder keypoints (x, y)
        left_hip, right_hip: Hip keypoints (x, y)
    
    Returns:
        Separation angle in degrees
    """
    ls, rs = np.array(left_shoulder), np.array(right_shoulder)
    lh, rh = np.array(left_hip), np.array(right_hip)
    
    # Vector from left to right shoulder
    shoulder_vec = rs - ls
    # Vector from left to right hip
    hip_vec = rh - lh
    
    # Calculate angle between the two vectors
    shoulder_angle = degrees(atan2(shoulder_vec[1], shoulder_vec[0]))
    hip_angle = degrees(atan2(hip_vec[1], hip_vec[0]))
    
    # Separation is the difference
    separation = abs(shoulder_angle - hip_angle)
    
    # Normalize to 0-180 range
    if separation > 180:
        separation = 360 - separation
    
    return float(separation)


def smooth(series, win=5):
    """
    Smooth a pandas Series using rolling window.
    
    Args:
        series: pandas Series to smooth
        win: Window size for rolling average
    
    Returns:
        Smoothed pandas Series
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Use rolling mean with minimum periods
    smoothed = series.rolling(window=win, center=True, min_periods=1).mean()
    
    # Fill any remaining NaN values
    smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
    
    return smoothed


def lateral_trunk_angle(mid_shoulder, mid_hip):
    """
    Calculate trunk angle relative to vertical.
    
    Args:
        mid_shoulder: (x, y) coordinates of mid-point between shoulders
        mid_hip: (x, y) coordinates of mid-point between hips
    
    Returns:
        Angle in degrees (0 = vertical, + = lean right, - = lean left)
    """
    mid_shoulder = np.array(mid_shoulder)
    mid_hip = np.array(mid_hip)
    
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]  # y increases downward in image
    
    # Calculate angle from vertical
    # Using atan2(x, -y) because y-axis is inverted
    angle = degrees(atan2(dx, max(1e-6, -dy)))
    
    return float(angle)


def velocity_2d(positions, times):
    """
    Calculate 2D velocity from position trajectory using linear regression.
    
    Args:
        positions: List or array of (x, y) positions
        times: List or array of corresponding timestamps
    
    Returns:
        (vx, vy) velocity components, or (None, None) if insufficient data
    """
    if len(positions) < 3:
        return None, None
    
    positions = np.array(positions)
    times = np.array(times)
    
    # Use linear regression: position = v*t + offset
    t0 = times[0]
    dt = times - t0
    
    # Fit for x
    A = np.vstack([dt, np.ones(len(dt))]).T
    vx, _ = np.linalg.lstsq(A, positions[:, 0], rcond=None)[0]
    
    # Fit for y  
    vy, _ = np.linalg.lstsq(A, positions[:, 1], rcond=None)[0]
    
    return float(vx), float(vy)


def speed_2d(positions, times):
    """
    Calculate scalar speed from 2D trajectory.
    
    Args:
        positions: List or array of (x, y) positions
        times: List or array of corresponding timestamps
    
    Returns:
        Speed (magnitude of velocity), or None if insufficient data
    """
    vx, vy = velocity_2d(positions, times)
    
    if vx is None or vy is None:
        return None
    
    return float(np.hypot(vx, vy))


def joint_velocity(keypoints_sequence, joint_idx, times, smooth_window=5):
    """
    Calculate velocity of a specific joint over time.
    
    Args:
        keypoints_sequence: List of keypoint arrays (frames x 17 x 2)
        joint_idx: Index of the joint to track (0-16)
        times: Timestamps for each frame
        smooth_window: Window size for smoothing positions
    
    Returns:
        Array of velocities (one per frame), or None if insufficient data
    """
    if len(keypoints_sequence) < smooth_window:
        return None
    
    # Extract joint positions
    positions = []
    valid_times = []
    
    for i, kpts in enumerate(keypoints_sequence):
        if kpts is not None and not np.isnan(kpts[joint_idx]).any():
            positions.append(kpts[joint_idx])
            valid_times.append(times[i])
    
    if len(positions) < 3:
        return None
    
    positions = np.array(positions)
    valid_times = np.array(valid_times)
    
    # Smooth positions
    pos_smooth = np.column_stack([
        pd.Series(positions[:, 0]).rolling(smooth_window, center=True, min_periods=1).mean(),
        pd.Series(positions[:, 1]).rolling(smooth_window, center=True, min_periods=1).mean()
    ])
    
    # Calculate velocity
    velocities = []
    for i in range(len(pos_smooth)):
        if i == 0:
            velocities.append(0.0)
        else:
            dt = valid_times[i] - valid_times[i-1]
            if dt > 0:
                dx = pos_smooth[i, 0] - pos_smooth[i-1, 0]
                dy = pos_smooth[i, 1] - pos_smooth[i-1, 1]
                v = np.hypot(dx, dy) / dt
                velocities.append(v)
            else:
                velocities.append(0.0)
    
    return np.array(velocities)


if __name__ == "__main__":
    # Simple tests
    print("Testing metrics utilities...")
    
    # Test euclidean distance
    assert abs(euclid([0, 0], [3, 4]) - 5.0) < 0.01
    print("✓ euclid works")
    
    # Test angle calculation
    angle = angle_at_joint([0, 1], [0, 0], [1, 0])
    assert abs(angle - 90.0) < 0.01
    print("✓ angle_at_joint works")
    
    # Test smoothing
    data = pd.Series([1, 2, 3, 4, 5])
    smoothed = smooth(data, win=3)
    assert len(smoothed) == len(data)
    print("✓ smooth works")
    
    print("\nAll tests passed! ✅")