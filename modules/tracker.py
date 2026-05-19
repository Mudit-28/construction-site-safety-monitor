import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sort import Sort


class WorkerTracker:
    def __init__(self):
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        self.id_map = {}

    def update(self, detections):
        persons = [d for d in detections if d['class'] == 'person']

        if len(persons) == 0:
            self.tracker.update()
            return detections

        # format for SORT: [[x1,y1,x2,y2,conf], ...]
        sort_input = np.array([
            [*p['bbox'], p['conf']] for p in persons
        ])

        tracked = self.tracker.update(sort_input)

        # match track IDs back to detections by IoU
        for track in tracked:
            x1, y1, x2, y2, track_id = track
            best_idx, best_iou = -1, 0
            for i, p in enumerate(persons):
                iou = _iou([x1,y1,x2,y2], p['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                persons[best_idx]['track_id'] = int(track_id)

        return detections


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    return inter / (area_a + area_b - inter)