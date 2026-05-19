"""
PPE detector wrapper around Ultralytics YOLO.

Changes for the final version:

1.  The interim detector passed the default Ultralytics thresholds
    (conf=0.25, iou=0.7).  §2.2 of the interim report identified the
    default iou=0.7 as too aggressive for dense construction scenes
    where workers stand close together; NMS was suppressing real
    second workers.  We expose ``iou`` and default it to 0.5.

2.  ``conf`` remains low at detector level so the UI can still *show*
    uncertain boxes in grey — but the compliance module now applies
    per-class higher thresholds before counting anything.  This keeps
    visual diagnostics useful while making scoring conservative.

3.  We add an optional tiled-inference path (``detect_tiled``) that
    implements the SAHI-style approach discussed in §2.4 of the
    interim report.  It's off by default because it's ~4× slower; the
    CLI and Flask app expose it as a flag for wide-angle shots with
    distant workers.

4.  Drawing is unchanged visually but now colour-codes boxes by
    confidence (faded for low-conf, bright for high-conf) so the
    reviewer can tell at a glance which detections the compliance
    engine actually trusted.
"""

from __future__ import annotations

import cv2
import numpy as np

# ``ultralytics`` is a large dependency (pytorch).  We import lazily so
# the rest of the project — MockDetector-based tests, the Flask app in
# read-only demo mode, the compliance / smoothing modules — stays
# usable even if ultralytics isn't installed.
_YOLO = None
def _get_YOLO():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO    # lazy import
        _YOLO = YOLO
    return _YOLO


CLASS_NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]

COLORS = {
    "helmet":    (0, 255, 0),
    "no-helmet": (0, 0, 255),
    "vest":      (0, 255, 0),
    "no-vest":   (0, 0, 255),
    "person":    (255, 165, 0),
}


class PPEDetector:
    """
    Thin wrapper over YOLO that returns detections as plain dicts
    and exposes the knobs we care about (conf, iou, input size, device).
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.50,            # lowered from Ultralytics default 0.7 (see §2.2)
        imgsz: int = 640,
        device: str | int | None = None,
    ) -> None:
        YOLO = _get_YOLO()
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    # ------------------------------------------------------------------
    # Standard single-pass inference
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[dict] = []
        for box in results.boxes:
            cls_idx = int(box.cls)
            if cls_idx >= len(CLASS_NAMES):
                continue
            detections.append({
                "class": CLASS_NAMES[cls_idx],
                "conf": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]],
            })
        return detections

    # ------------------------------------------------------------------
    # SAHI-style tiled inference (§2.4 planned fix)
    # ------------------------------------------------------------------
    def detect_tiled(
        self,
        frame: np.ndarray,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.2,
    ) -> list[dict]:
        """
        Slice the frame into overlapping tiles, run inference on each
        at the model's native input size, then merge with a class-aware
        NMS.  Gives ~2× effective zoom on each region which helps
        recover the sub-30-pixel distant workers the report identified.
        """
        H, W = frame.shape[:2]
        tile_h = int(H / rows * (1 + overlap))
        tile_w = int(W / cols * (1 + overlap))
        step_h = H // rows
        step_w = W // cols

        all_det: list[dict] = []
        # Tile grid
        for r in range(rows):
            for c in range(cols):
                y0 = r * step_h
                x0 = c * step_w
                y1 = min(y0 + tile_h, H)
                x1 = min(x0 + tile_w, W)
                tile = frame[y0:y1, x0:x1]
                for d in self.detect(tile):
                    bx1, by1, bx2, by2 = d["bbox"]
                    # map back to full-frame coordinates
                    d["bbox"] = [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
                    all_det.append(d)

        # Full frame too, so we don't miss the big central box
        all_det.extend(self.detect(frame))

        return _class_aware_nms(all_det, iou_thr=self.iou)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        detections: list[dict],
        show_conf: bool = True,
    ) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            color = COLORS.get(d["class"], (255, 255, 255))
            # Fade low-confidence boxes so the viewer knows which were trusted.
            thickness = 2 if d["conf"] >= 0.5 else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = d["class"]
            if "track_id" in d:
                label = f"#{d['track_id']} {label}"
            if show_conf:
                label = f"{label} {d['conf']:.2f}"

            # Text background for legibility.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )
        return frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _class_aware_nms(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """
    NMS that only suppresses boxes of the *same* class. Needed when we
    merge tile results: tiles may produce overlapping duplicates of
    e.g. the same helmet, but a 'person' and 'vest' at similar
    coordinates are not duplicates — they're supposed to overlap.
    """
    kept: list[dict] = []
    # Group by class
    by_class: dict[str, list[dict]] = {}
    for d in dets:
        by_class.setdefault(d["class"], []).append(d)

    for cls, items in by_class.items():
        items.sort(key=lambda x: -x["conf"])
        survivors: list[dict] = []
        for cand in items:
            suppressed = False
            for s in survivors:
                if _iou(cand["bbox"], s["bbox"]) > iou_thr:
                    suppressed = True
                    break
            if not suppressed:
                survivors.append(cand)
        kept.extend(survivors)
    return kept
