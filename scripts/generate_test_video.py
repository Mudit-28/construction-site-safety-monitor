"""
Generate a synthetic construction scene video for testing.

This is not meant to look realistic — it's meant to exercise every
code path in the video pipeline with known ground truth:

  * Multiple moving workers (stresses the SORT tracker)
  * Workers entering and leaving frame
  * A worker who loses their helmet mid-video (should fire an alert)
  * A worker with no vest the whole time (worst offender)
  * A fully-compliant worker as control

The output can be passed through the real pipeline with the
MockDetector (modules.mock_detector) — which reads a ground-truth
annotation file we write alongside the video — to exercise the
full tracker + compliance + smoothing chain without needing the
actual YOLO weights.

Usage
-----
    python scripts/generate_test_video.py --out test_scene.mp4 \\
                                          --seconds 10 --fps 25

This writes:
    test_scene.mp4       : the video itself
    test_scene.truth.json : per-frame ground-truth detections
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

@dataclass
class FakeWorker:
    """A synthetic construction worker with deterministic motion + PPE state."""
    id: int
    colour: tuple[int, int, int]         # body colour (BGR)
    start_x: float
    start_y: float
    vel_x: float
    vel_y: float
    has_helmet: bool = True
    has_vest: bool = True
    helmet_lost_at_s: float | None = None
    vest_lost_at_s: float | None = None
    appears_at_s: float = 0.0
    disappears_at_s: float | None = None

    def position_at(self, t: float) -> tuple[float, float] | None:
        """Centre position at time t seconds. None if not visible."""
        if t < self.appears_at_s:
            return None
        if self.disappears_at_s is not None and t >= self.disappears_at_s:
            return None
        return (self.start_x + self.vel_x * t,
                self.start_y + self.vel_y * t)

    def ppe_state_at(self, t: float) -> tuple[bool, bool]:
        """(has_helmet, has_vest) at time t."""
        helmet = self.has_helmet
        if self.helmet_lost_at_s is not None and t >= self.helmet_lost_at_s:
            helmet = False
        vest = self.has_vest
        if self.vest_lost_at_s is not None and t >= self.vest_lost_at_s:
            vest = False
        return helmet, vest


def build_default_scene() -> list[FakeWorker]:
    """A deterministic 4-worker scene covering all edge cases."""
    return [
        # Worker 1: fully compliant, left-to-right
        FakeWorker(
            id=1, colour=(160, 100, 60),
            start_x=80, start_y=250, vel_x=30, vel_y=0,
        ),
        # Worker 2: compliant at first, loses helmet at t=3s
        FakeWorker(
            id=2, colour=(90, 130, 170),
            start_x=200, start_y=200, vel_x=20, vel_y=5,
            helmet_lost_at_s=3.0,
        ),
        # Worker 3: never has a vest — worst offender
        FakeWorker(
            id=3, colour=(100, 100, 100),
            start_x=400, start_y=300, vel_x=-15, vel_y=-3,
            has_vest=False,
        ),
        # Worker 4: appears at t=2s, fully compliant, stays briefly
        FakeWorker(
            id=4, colour=(140, 80, 120),
            start_x=550, start_y=350, vel_x=-10, vel_y=0,
            appears_at_s=2.0, disappears_at_s=7.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BODY_W, BODY_H = 60, 160
HELMET_H = 30
VEST_H = 70

HI_VIS_ORANGE = (0, 140, 255)
HELMET_YELLOW = (50, 200, 220)


def _draw_worker(
    frame: np.ndarray, cx: float, cy: float,
    body_colour: tuple[int, int, int],
    has_helmet: bool, has_vest: bool,
) -> dict:
    """Draw one worker and return the ground-truth bbox dict list."""
    x1 = int(cx - BODY_W / 2)
    y1 = int(cy - BODY_H / 2)
    x2 = int(cx + BODY_W / 2)
    y2 = int(cy + BODY_H / 2)

    # body (rectangle)
    cv2.rectangle(frame, (x1, y1 + HELMET_H), (x2, y2), body_colour, -1)
    # face patch (so we don't have a floating head outline)
    cv2.rectangle(frame, (x1 + 10, y1 + HELMET_H), (x2 - 10, y1 + HELMET_H + 30),
                  (190, 180, 170), -1)

    person_box = [x1, y1, x2, y2]
    dets = [{"class": "person", "conf": 0.98, "bbox": person_box}]

    # helmet: hard-hat yellow rectangle on top of the body
    helmet_box = [x1 + 5, y1, x2 - 5, y1 + HELMET_H]
    if has_helmet:
        cv2.rectangle(frame,
                      (helmet_box[0], helmet_box[1]),
                      (helmet_box[2], helmet_box[3]),
                      HELMET_YELLOW, -1)
        cv2.rectangle(frame,
                      (helmet_box[0], helmet_box[1]),
                      (helmet_box[2], helmet_box[3]),
                      (30, 100, 110), 1)
        dets.append({"class": "helmet", "conf": 0.88, "bbox": helmet_box})

    # vest: bright orange rectangle on the torso
    vest_y1 = y1 + HELMET_H + 35
    vest_box = [x1 + 3, vest_y1, x2 - 3, vest_y1 + VEST_H]
    if has_vest:
        cv2.rectangle(frame,
                      (vest_box[0], vest_box[1]),
                      (vest_box[2], vest_box[3]),
                      HI_VIS_ORANGE, -1)
        # reflective stripes
        for stripe_y in [vest_y1 + 20, vest_y1 + 45]:
            cv2.line(frame, (vest_box[0], stripe_y), (vest_box[2], stripe_y),
                     (255, 255, 255), 2)
        dets.append({"class": "vest", "conf": 0.82, "bbox": vest_box})

    # Put worker label for human viewing
    cv2.putText(frame, f"#{dets[0]['bbox'][0]}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return dets


def _draw_background(frame: np.ndarray, t: float) -> None:
    """Construction-site-ish background: ground, sky gradient, a crane."""
    h, w = frame.shape[:2]
    # sky
    frame[:] = (180, 150, 120)
    # ground
    cv2.rectangle(frame, (0, int(h * 0.7)), (w, h), (80, 85, 95), -1)
    # horizon line
    cv2.line(frame, (0, int(h * 0.7)), (w, int(h * 0.7)), (50, 55, 65), 2)

    # static "crane" silhouette
    cv2.rectangle(frame, (w - 120, int(h * 0.15)), (w - 100, int(h * 0.7)),
                  (40, 50, 60), -1)
    cv2.line(frame, (w - 110, int(h * 0.15)), (w - 300, int(h * 0.15)),
             (40, 50, 60), 4)
    cv2.line(frame, (w - 110, int(h * 0.25)), (w - 200, int(h * 0.15)),
             (40, 50, 60), 2)

    # timecode bottom-right so the viewer can see frame progress
    cv2.putText(frame, f"t={t:5.2f}s", (w - 120, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    out_path: Path,
    seconds: float = 10.0,
    fps: int = 25,
    width: int = 720,
    height: int = 480,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)

    workers = build_default_scene()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"Could not open writer: {out_path}")

    truth: dict[int, list[dict]] = {}
    n_frames = int(seconds * fps)

    for i in range(n_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        _draw_background(frame, t)

        frame_truth: list[dict] = []
        for w in workers:
            pos = w.position_at(t)
            if pos is None:
                continue
            cx, cy = pos
            # wrap around frame borders
            cx %= (width + 100)
            has_h, has_v = w.ppe_state_at(t)
            dets = _draw_worker(frame, cx, cy, w.colour, has_h, has_v)
            # Annotate with original worker id for evaluation
            for d in dets:
                d["gt_worker_id"] = w.id
            frame_truth.extend(dets)

        writer.write(frame)
        truth[i + 1] = frame_truth    # 1-indexed to match run_inference

    writer.release()

    # Sidecar ground-truth JSON
    truth_path = out_path.with_suffix("").with_suffix(".truth.json")
    with open(truth_path, "w") as f:
        json.dump({
            "fps": fps, "width": width, "height": height,
            "seconds": seconds, "n_frames": n_frames,
            "frames": truth,
        }, f)

    print(f"Wrote {out_path}  ({n_frames} frames, {seconds}s @ {fps}fps)")
    print(f"Wrote {truth_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("test_scene.mp4"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.seconds, args.fps, args.width, args.height, args.seed)
