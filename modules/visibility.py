import cv2
import numpy as np

# HSV ranges for high-visibility colors (yellow-green and orange)
HI_VIS_RANGES = [
    ([20, 100, 100], [35, 255, 255]),   # yellow-green
    ([0,  100, 100], [15, 255, 255]),   # orange
    ([160,100, 100], [180,255, 255]),   # red-orange wrap
]

def estimate_visibility(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    # clamp to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = roi.shape[0] * roi.shape[1]

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HI_VIS_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    hi_vis_pixels = cv2.countNonZero(mask)
    return round(hi_vis_pixels / total_pixels, 3)


def classify_visibility(score):
    if score >= 0.3:
        return "good"
    elif score >= 0.1:
        return "poor"
    else:
        return "none"