import numpy as np
import pandas as pd
from math import atan2, degrees

def smooth(series, win=5):
    if isinstance(series, list):
        series = pd.Series(series)
    if win <= 1:
        return series
    return series.rolling(win, min_periods=1, center=True).mean()

def euclid(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    return float(np.linalg.norm(p - q))

def line_angle(p1, p2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    return float(degrees(atan2(v[1], v[0])))

def hip_shoulder_separation(LS, RS, LH, RH):
    shoulder_ang = line_angle(LS, RS)
    hip_ang = line_angle(LH, RH)
    sep = abs(shoulder_ang - hip_ang) % 360.0
    if sep > 180: sep = 360 - sep
    if sep > 90: sep = 180 - sep
    return float(sep)

def angle_at_joint(p_proximal, p_joint, p_distal):
    """Angle at the middle point (in degrees) via cosine law."""
    a = euclid(p_joint, p_proximal)
    b = euclid(p_joint, p_distal)
    c = euclid(p_proximal, p_distal)
    if a == 0 or b == 0:
        return None
    cos_t = (a*a + b*b - c*c) / (2*a*b)
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))
