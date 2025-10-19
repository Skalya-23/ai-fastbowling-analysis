import cv2
import numpy as np

# COCO skeleton edges
EDGES = [
    (5,7),(7,9),     # left arm
    (6,8),(8,10),    # right arm
    (5,6),           # shoulders
    (5,11),(6,12),   # torso
    (11,12),         # hips
    (11,13),(13,15), # left leg
    (12,14),(14,16)  # right leg
]

def draw_skeleton(img, kpts, radius=4, thickness=2):
    kpts = np.array(kpts, dtype=float)
    for (i,j) in EDGES:
        p1 = tuple(np.int32(kpts[i]))
        p2 = tuple(np.int32(kpts[j]))
        cv2.line(img, p1, p2, (255,255,255), thickness, cv2.LINE_AA)
    for p in kpts:
        cv2.circle(img, tuple(np.int32(p)), radius, (0,0,0), -1)
        cv2.circle(img, tuple(np.int32(p)), radius-2, (255,255,255), -1)

def put_multiline_text(img, lines, org, line_h=22, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255,255,255), thickness=1):
    x, y = org
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i*line_h), font, scale, color, thickness, cv2.LINE_AA)
