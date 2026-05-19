"""
Mock detector for testing the full pipeline without YOLO weights.

Reads a ground-truth .truth.json file (produced by
scripts/generate_test_video.py) and returns, on each ``detect()``
call, the detections for the current frame index — with realistic
noise injected so the compliance / smoothing logic gets exercised.

Noise behaviours (all stochastic, seeded):

  * Per-detection confidence jitter (±0.15 around the truth)
  * Occasional missed detections (drop rate configurable)
  * Injection of the §2.3 failure mode: for ~10% of vest detections,
    also emit a spurious 'no-vest' box at the same location (same
    worker, simulating detector uncertainty). The revised compliance
    module should ignore this in favour of the positive vest.
  * Occasional low-confidence 'helmet' box on workers who don't have
    one, simulating the §2.1 turban false positive. The revised
    module should filter these out via the HELMET_CONF_MIN threshold.

Usage
-----
    from modules.mock_detector import MockDetector
    det = MockDetector("test_scene.truth.json", seed=7)
    for frame_idx in range(1, n+1):
        det.set_frame(frame_idx)
        dets = det.detect(frame)   # frame is ignored; we use the truth
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class MockDetector:
    """
    Drop-in replacement for PPEDetector that reads ground-truth
    annotations and returns noisy detections matching them.

    The ``frame`` argument to ``detect`` is ignored — we use
    ``set_frame(idx)`` to tell the mock which frame to emulate.
    """

    def __init__(
        self,
        truth_path: str | Path,
        seed: int = 1337,
        miss_rate: float = 0.02,           # probability of dropping a detection
        uncertain_vest_rate: float = 0.1,  # prob of firing no-vest on a vested torso (§2.3)
        turban_fp_rate: float = 0.03,      # prob of low-conf helmet on a helmet-less head (§2.1)
    ) -> None:
        with open(truth_path) as f:
            self._truth = json.load(f)
        # JSON object keys are strings
        self._frames = {int(k): v for k, v in self._truth["frames"].items()}
        self._current_frame = 1
        self._rng = random.Random(seed)
        self.miss_rate = miss_rate
        self.uncertain_vest_rate = uncertain_vest_rate
        self.turban_fp_rate = turban_fp_rate

    # API parity with the real detector
    @property
    def conf(self) -> float:  return 0.25
    @property
    def iou(self) -> float:   return 0.5

    def set_frame(self, idx: int) -> None:
        self._current_frame = idx

    def detect(self, frame: np.ndarray | None = None) -> list[dict]:
        gts = self._frames.get(self._current_frame, [])
        out: list[dict] = []

        for gt in gts:
            # Drop a small fraction of detections
            if self._rng.random() < self.miss_rate:
                continue

            conf = max(0.3, min(0.99,
                                gt["conf"] + self._rng.gauss(0, 0.05)))
            det = {
                "class": gt["class"],
                "conf": conf,
                "bbox": list(gt["bbox"]),
            }
            out.append(det)

            # §2.3 simulation: sometimes emit a co-located no-vest box
            if gt["class"] == "vest" and self._rng.random() < self.uncertain_vest_rate:
                out.append({
                    "class": "no-vest",
                    "conf": conf * 0.95,   # roughly matched confidence
                    "bbox": list(gt["bbox"]),
                })

        # §2.1 simulation: for each person w/o a helmet GT, small chance
        # we fire a low-confidence 'helmet' detection (turban FP).
        persons_no_helmet = [
            gt for gt in gts if gt["class"] == "person"
            and not any(
                g["class"] == "helmet" and g.get("gt_worker_id") == gt.get("gt_worker_id")
                for g in gts
            )
        ]
        for p in persons_no_helmet:
            if self._rng.random() < self.turban_fp_rate:
                x1, y1, x2, y2 = p["bbox"]
                out.append({
                    "class": "helmet",
                    "conf": 0.35,   # deliberately below HELMET_CONF_MIN
                    "bbox": [x1 + 10, y1, x2 - 10, y1 + 30],
                })

        return out

    def detect_tiled(self, frame, **kw):
        return self.detect(frame)

    def draw(self, frame, detections, show_conf: bool = True):
        """Forward to the real draw() so annotated videos still look right."""
        # Lazy import to avoid requiring ultralytics for the mock
        import cv2
        COLORS = {
            "helmet":    (0, 255, 0),
            "no-helmet": (0, 0, 255),
            "vest":      (0, 255, 0),
            "no-vest":   (0, 0, 255),
            "person":    (255, 165, 0),
        }
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            color = COLORS.get(d["class"], (255, 255, 255))
            thickness = 2 if d["conf"] >= 0.5 else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = d["class"]
            if "track_id" in d:
                label = f"#{d['track_id']} {label}"
            if show_conf:
                label = f"{label} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame
