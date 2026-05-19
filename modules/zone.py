import cv2
import numpy as np


class ZoneFilter:
    def __init__(self, polygon_pts=None, frame_shape=None):
        if polygon_pts:
            self.poly = np.array(polygon_pts, dtype=np.int32)
        elif frame_shape:
            h, w = frame_shape[:2]
            self.poly = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.int32)
        else:
            self.poly = None

    def in_zone(self, bbox):
        if self.poly is None:
            return True
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cv2.pointPolygonTest(self.poly, (cx, cy), False) >= 0

    def filter_detections(self, detections):
        if self.poly is None:
            return detections
        return [d for d in detections if self.in_zone(d['bbox'])]

    def draw(self, frame):
        if self.poly is None:
            return frame
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.poly], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self.poly], True, (0, 255, 255), 2)
        cv2.putText(frame, "Construction Zone", 
                    tuple(self.poly[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        return frame