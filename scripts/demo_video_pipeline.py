#!/usr/bin/env python3
"""
Demo: end-to-end video pipeline without YOLO weights.

Generates a synthetic test video, processes it through the full pipeline
(MockDetector → SORT → zone filter → compliance → smoothing), and writes
all the outputs into a single directory.  Useful for:

  * Validating your setup works before training the real model
  * Showing reviewers / markers that the pipeline is wired correctly
  * Debugging the compliance / smoothing / UI code in isolation

Usage
-----
    python scripts/demo_video_pipeline.py --out demo_output/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.mock_detector import MockDetector
from modules.tracker import WorkerTracker
from modules.zone import ZoneFilter
from modules.compliance import score_frame
from modules.smoothing import SmoothingState
from run_inference import _draw_hud
from scripts.generate_test_video import generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demo_output"),
                    help="Output directory")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic video + ground truth
    print("▶ Generating synthetic scene …")
    video_in = out_dir / "scene_raw.mp4"
    truth_path = video_in.with_suffix(".truth.json")
    generate(video_in, seconds=args.seconds, fps=args.fps)

    # 2. Set up pipeline
    cap = cv2.VideoCapture(str(video_in))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_out = out_dir / "scene_annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )
    assert writer.isOpened()

    detector = MockDetector(truth_path, seed=args.seed)
    tracker = WorkerTracker()
    zone = ZoneFilter(frame_shape=(h, w, 3))
    smoother = SmoothingState(alpha=0.1, alert_after=10)

    # 3. Run pipeline
    print(f"▶ Processing {total} frames …")
    csv_path = out_dir / "timeline.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(
        csv_f,
        fieldnames=["frame", "time_s", "raw_score", "smoothed_score",
                    "n_workers", "missing_helmets", "missing_vests", "violations"],
    )
    csv_w.writeheader()

    alerts: list[dict] = []
    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detector.set_frame(frame_idx)
        dets = detector.detect(frame)
        dets = zone.filter_detections(dets)
        dets = tracker.update(dets)

        score, violations, details = score_frame(dets, zone_active=True)
        smoothed, new_alerts = smoother.update(score, details)
        for a in new_alerts:
            alerts.append({"frame": frame_idx, "message": a})

        annotated = zone.draw(frame)
        annotated = detector.draw(annotated, dets)
        inst_fps = frame_idx / max(time.perf_counter() - t0, 1e-6)
        annotated = _draw_hud(annotated, score, violations,
                              smoothed=smoothed, fps=inst_fps)
        writer.write(annotated)

        csv_w.writerow({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "raw_score": score,
            "smoothed_score": smoothed,
            "n_workers": details.get("n_workers", 0),
            "missing_helmets": details.get("missing_helmets", 0),
            "missing_vests": details.get("missing_vests", 0),
            "violations": "; ".join(violations),
        })

    elapsed = time.perf_counter() - t0
    cap.release()
    writer.release()
    csv_f.close()

    # 4. Summary JSON
    summary = smoother.summary()
    summary.update({
        "input_video": str(video_in),
        "output_video": str(video_out),
        "timeline_csv": str(csv_path),
        "duration_s": round(frame_idx / fps, 2),
        "processing_time_s": round(elapsed, 2),
        "avg_processing_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        "playback_fps": fps,
        "resolution": [w, h],
        "alerts": alerts,
    })
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # 5. Print a nice report
    print()
    print("─" * 60)
    print("Pipeline complete.")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Wall-clock time  : {elapsed:.2f}s  ({frame_idx/elapsed:.1f} FPS)")
    print(f"  Final smoothed   : {summary['final_smoothed_score']}%")
    print(f"  Unique workers   : {len(summary['tracks'])}")
    print(f"  Alerts raised    : {len(alerts)}")
    print()
    print("Outputs:")
    print(f"  {video_in}")
    print(f"  {video_out}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print()
    print("Top 3 worst-offenders:")
    for t in summary["tracks"][:3]:
        print(f"  Worker #{t['worker_id']}: "
              f"no-helmet {t['pct_no_helmet']}%, no-vest {t['pct_no_vest']}%, "
              f"frames {t['frames_seen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
