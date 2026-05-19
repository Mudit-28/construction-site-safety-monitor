#!/usr/bin/env python3
"""
run_inference.py
================

End-to-end inference script for the Construction Zone Safety
Compliance Monitoring System.

Supports
--------
* Images   (.jpg/.jpeg/.png) — single-frame scoring
* Videos   (.mp4/.avi/.mov/.mkv) — full pipeline with tracking,
                                    temporal smoothing, CSV timeline,
                                    and per-worker alerts
* Folders  — processes every image in the folder (batch mode)

Usage
-----
    python run_inference.py --model models/ppe_detector/weights/best.pt \\
                            --input test.mp4 --output results/

    # With tiled inference for distant workers:
    python run_inference.py --model best.pt -i video.mp4 --tile

    # CPU-only, smaller image size for slower machines:
    python run_inference.py --model best.pt -i clip.mp4 \\
                            --device cpu --imgsz 416

The script writes three things for every video:
    <output>.mp4               : annotated video
    <output>_timeline.csv      : per-frame score + violations
    <output>_summary.json      : final stats + per-worker breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make modules importable whether we're run from project root or inside it
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.detector import PPEDetector
from modules.compliance import score_frame
from modules.smoothing import SmoothingState
from modules.zone import ZoneFilter
from modules.tracker import WorkerTracker
from modules.visibility import estimate_visibility, classify_visibility


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ======================================================================
# Core per-frame pipeline
# ======================================================================

def process_frame(
    frame: np.ndarray,
    detector: PPEDetector,
    tracker: WorkerTracker,
    zone: ZoneFilter,
    *,
    use_tiled: bool = False,
    show_visibility: bool = False,
) -> tuple[np.ndarray, float, list[str], dict]:
    """
    Run detection → zone filter → tracking → compliance on one frame.

    Returns (annotated_frame, score, violations, details).
    """
    # 1. detection
    detections = (
        detector.detect_tiled(frame) if use_tiled else detector.detect(frame)
    )

    # 2. zone filter
    detections = zone.filter_detections(detections)

    # 3. tracking (adds `track_id` to person detections in-place)
    detections = tracker.update(detections)

    # 4. optional visibility scoring (hi-vis colour check)
    if show_visibility:
        for d in detections:
            if d["class"] == "person":
                v = estimate_visibility(frame, d["bbox"])
                d["visibility"] = v
                d["visibility_label"] = classify_visibility(v)

    # 5. compliance
    score, violations, details = score_frame(detections, zone_active=True)

    # 6. draw overlays
    annotated = zone.draw(frame)
    annotated = detector.draw(annotated, detections)
    annotated = _draw_hud(annotated, score, violations)

    return annotated, score, violations, details


def _draw_hud(
    frame: np.ndarray,
    score: float,
    violations: list[str],
    smoothed: float | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """
    Compliance HUD drawn on the upper-left of the frame.
    """
    h, w = frame.shape[:2]
    # Colour: green → orange → red
    color = (
        (0, 200, 0) if score >= 80
        else (0, 165, 255) if score >= 50
        else (0, 0, 255)
    )

    # Panel
    panel_h = 60 + min(len(violations), 4) * 22
    cv2.rectangle(frame, (10, 10), (10 + 360, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (10 + 360, 10 + panel_h), color, 2)

    # Score line
    score_line = f"Compliance: {score:.1f}%"
    if smoothed is not None and abs(smoothed - score) >= 0.1:
        score_line += f"  (smoothed: {smoothed:.1f}%)"
    cv2.putText(
        frame, score_line, (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
    )

    # FPS
    if fps is not None:
        cv2.putText(
            frame, f"{fps:.1f} FPS", (w - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )

    # Violation list (max 4 lines so we don't occlude the whole frame)
    for i, v in enumerate(violations[:4]):
        cv2.putText(
            frame, f"! {v[:50]}", (20, 75 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1,
        )
    if len(violations) > 4:
        cv2.putText(
            frame, f"... and {len(violations) - 4} more", (20, 75 + 4 * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 200), 1,
        )
    return frame


# ======================================================================
# Input dispatch
# ======================================================================

def run_on_image(
    img_path: Path,
    out_path: Path,
    detector: PPEDetector,
    *,
    tracker: WorkerTracker | None = None,
    use_tiled: bool = False,
    show_visibility: bool = False,
    quiet: bool = False,
) -> dict:
    """Single-image inference. Returns a result dict."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise IOError(f"Could not read image: {img_path}")

    zone = ZoneFilter(frame_shape=frame.shape)
    tracker = tracker or WorkerTracker()   # fresh for standalone image

    annotated, score, violations, details = process_frame(
        frame, detector, tracker, zone,
        use_tiled=use_tiled, show_visibility=show_visibility,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    if not quiet:
        print(f"[image] {img_path.name} → {out_path.name}")
        print(f"        Score: {score}%  |  {len(violations)} violation(s)")
        for v in violations:
            print(f"          - {v}")

    return {
        "input": str(img_path),
        "output": str(out_path),
        "score": score,
        "violations": violations,
        "details": details,
    }


def run_on_video(
    vid_path: Path,
    out_path: Path,
    detector: PPEDetector,
    *,
    use_tiled: bool = False,
    show_visibility: bool = False,
    alpha: float = 0.1,
    alert_after: int = 10,
    write_csv: bool = True,
    progress_cb=None,
) -> dict:
    """
    Full video pipeline with tracking, smoothing, CSV timeline,
    and summary. Writes <out>.mp4, <out>_timeline.csv, <out>_summary.json.
    """
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {vid_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_video = out_path.with_suffix(".mp4")

    # Try H.264 (avc1) first — plays natively in HTML5 <video>.  Fall
    # back to mp4v which OpenCV always has but browsers can't decode.
    # If you need browser playback and avc1 isn't available on your
    # OpenCV build, re-encode the output with:
    #   ffmpeg -i out.mp4 -c:v libx264 -crf 23 out_h264.mp4
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*codec),
            fps, (w, h),
        )
        if writer.isOpened():
            break
    if not writer.isOpened():
        raise IOError(f"Could not open video writer: {out_video}")

    # One tracker + zone + smoother for the entire video.  Earlier bug
    # in the interim code: zone was reconstructed every frame (which
    # did nothing harmful, just wasteful), and tracker IDs were correct
    # but never surfaced in the scoring. We fix both here.
    tracker = WorkerTracker()
    zone = ZoneFilter(frame_shape=(h, w, 3))
    smoother = SmoothingState(alpha=alpha, alert_after=alert_after)

    csv_rows: list[dict] = []
    all_alerts: list[tuple[int, str]] = []  # (frame_idx, msg)

    t_start = time.perf_counter()
    frame_idx = 0
    last_print = t_start

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        annotated, raw_score, violations, details = process_frame(
            frame, detector, tracker, zone,
            use_tiled=use_tiled, show_visibility=show_visibility,
        )

        # Temporal smoothing & per-track alerting
        smoothed, new_alerts = smoother.update(raw_score, details)
        for a in new_alerts:
            all_alerts.append((frame_idx, a))

        # FPS estimate (recent average)
        now = time.perf_counter()
        inst_fps = frame_idx / (now - t_start)

        # Redraw HUD with smoothed value + FPS (cheap: overwrites prev)
        annotated = _draw_hud(
            annotated, raw_score, violations,
            smoothed=smoothed, fps=inst_fps,
        )

        writer.write(annotated)

        csv_rows.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "raw_score": raw_score,
            "smoothed_score": smoothed,
            "n_workers": details.get("n_workers", 0),
            "missing_helmets": details.get("missing_helmets", 0),
            "missing_vests": details.get("missing_vests", 0),
            "violations": "; ".join(violations),
        })

        # progress
        if progress_cb:
            progress_cb(frame_idx, total)
        elif now - last_print > 2.0:
            pct = (100.0 * frame_idx / total) if total > 0 else 0.0
            print(
                f"  [{pct:5.1f}%]  frame {frame_idx}/{total}  "
                f"raw={raw_score:5.1f}%  smoothed={smoothed:5.1f}%  "
                f"{inst_fps:.1f} FPS"
            )
            last_print = now

    elapsed = time.perf_counter() - t_start
    cap.release()
    writer.release()

    # Write CSV timeline
    csv_path = out_path.with_name(out_path.stem + "_timeline.csv")
    if write_csv and csv_rows:
        with open(csv_path, "w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w_csv.writeheader()
            w_csv.writerows(csv_rows)

    # Write summary JSON
    summary = smoother.summary()
    summary.update({
        "input_video": str(vid_path),
        "output_video": str(out_video),
        "timeline_csv": str(csv_path),
        "duration_s": round(frame_idx / fps, 2),
        "processing_time_s": round(elapsed, 2),
        "avg_processing_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        "playback_fps": fps,
        "resolution": [w, h],
        "alerts": [{"frame": fr, "message": msg} for fr, msg in all_alerts],
    })
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"Done. Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} FPS avg)")
    print(f"  Annotated video : {out_video}")
    print(f"  Timeline CSV    : {csv_path}")
    print(f"  Summary JSON    : {summary_path}")
    print(f"  Final score     : {summary['final_smoothed_score']}%")
    print(f"  Total alerts    : {len(summary['alerts'])}")
    return summary


def run_on_folder(
    folder: Path,
    out_folder: Path,
    detector: PPEDetector,
    **kw,
) -> list[dict]:
    """Process every image in a folder. Shared tracker (batch images
    are independent so we make a fresh one per image)."""
    results: list[dict] = []
    images = sorted(
        [p for p in folder.iterdir()
         if p.suffix.lower() in IMAGE_EXTS]
    )
    if not images:
        print(f"No images in {folder}", file=sys.stderr)
        return results

    for p in images:
        out_path = out_folder / p.name
        r = run_on_image(p, out_path, detector, **kw)
        results.append(r)
    return results


# ======================================================================
# CLI
# ======================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="PPE Compliance Monitoring — image / video / batch inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", "-m", required=True, type=Path,
        help="Path to YOLO .pt weights (e.g. models/ppe_detector/weights/best.pt)",
    )
    ap.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input image / video file, or folder of images",
    )
    ap.add_argument(
        "--output", "-o", type=Path, default=Path("output"),
        help="Output path (file for single image/video, folder for batch)",
    )
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (raw)")
    ap.add_argument("--iou", type=float, default=0.50,
                    help="NMS IoU threshold (§2.2 of report: default 0.50, lowered from 0.70)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference input size (pixels)")
    ap.add_argument("--device", default=None,
                    help="Device: 'cpu', '0', 'cuda:0', etc. (None = auto)")
    ap.add_argument("--tile", action="store_true",
                    help="Enable SAHI-style tiled inference (§2.4: helps distant workers, slower)")
    ap.add_argument("--visibility", action="store_true",
                    help="Also compute hi-vis colour score for each worker")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="EMA smoothing alpha for video (lower = smoother)")
    ap.add_argument("--alert-after", type=int, default=10,
                    help="Frames of consecutive violation before alert fires (video only)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-image/per-frame log output")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        print("       Train first with train_model.py, or download a pretrained .pt", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    print(f"Loading model: {args.model}")
    detector = PPEDetector(
        str(args.model),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    kwargs = dict(use_tiled=args.tile, show_visibility=args.visibility)

    if args.input.is_dir():
        out_folder = args.output
        out_folder.mkdir(parents=True, exist_ok=True)
        run_on_folder(args.input, out_folder, detector,
                      quiet=args.quiet, **kwargs)
    elif args.input.suffix.lower() in IMAGE_EXTS:
        out = args.output
        if out.is_dir() or str(out).endswith(os.sep):
            out = out / args.input.name
        run_on_image(args.input, out, detector, quiet=args.quiet, **kwargs)
    elif args.input.suffix.lower() in VIDEO_EXTS:
        out = args.output
        if out.is_dir() or str(out).endswith(os.sep):
            out = out / (args.input.stem + "_annotated.mp4")
        run_on_video(
            args.input, out, detector,
            alpha=args.alpha, alert_after=args.alert_after,
            **kwargs,
        )
    else:
        print(f"ERROR: unsupported input type: {args.input.suffix}", file=sys.stderr)
        print(f"       Images: {sorted(IMAGE_EXTS)}", file=sys.stderr)
        print(f"       Videos: {sorted(VIDEO_EXTS)}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
