"""
Flask app for the PPE Compliance Monitoring System.

Endpoints
---------
GET  /                  → upload page
POST /upload            → accepts an image or a video, starts inference
GET  /result/<job_id>   → results page for a job
GET  /status/<job_id>   → JSON polling endpoint (frame count, % complete)
GET  /static/results/…  → served by Flask static handler

Implementation notes
--------------------
* Video processing runs on a background thread.  Image processing is
  synchronous because it finishes in <1 second.
* We keep jobs in an in-memory dict.  This is fine for a demo /
  single-node deployment; swap in Redis / a DB for anything real.
* The model is loaded once at app startup (slow), not per request.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import (
    Flask, jsonify, render_template, request, send_from_directory, url_for,
)

# Make the parent project importable no matter where the Flask app is started from
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_inference import (
    run_on_image, run_on_video,
    IMAGE_EXTS, VIDEO_EXTS,
)
from modules.detector import PPEDetector


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get(
    "PPE_MODEL_PATH",
    str(ROOT / "models" / "ppe_detector" / "weights" / "best.pt"),
)
DEVICE = os.environ.get("PPE_DEVICE", None)        # 'cpu', '0', None→auto
UPLOAD_DIR = APP_DIR / "static" / "uploads"
RESULTS_DIR = APP_DIR / "static" / "results"
MAX_UPLOAD_MB = int(os.environ.get("PPE_MAX_UPLOAD_MB", "200"))


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Flask app + detector (loaded once)
# ----------------------------------------------------------------------------

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_DETECTOR: PPEDetector | None = None
_DETECTOR_LOCK = threading.Lock()


def get_detector() -> PPEDetector:
    """Lazy-load the detector on first request.  Reloading requires a restart."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            model_path = Path(DEFAULT_MODEL)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLO weights not found at {model_path}. "
                    f"Set PPE_MODEL_PATH or train first."
                )
            app.logger.info(f"Loading YOLO model from {model_path} …")
            _DETECTOR = PPEDetector(str(model_path), device=DEVICE)
        return _DETECTOR


# ----------------------------------------------------------------------------
# In-memory job registry
# ----------------------------------------------------------------------------

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        return JOBS.get(job_id, {}).copy() if job_id in JOBS else None


# ----------------------------------------------------------------------------
# Background worker for video
# ----------------------------------------------------------------------------

def _video_worker(job_id: str, in_path: Path, out_path: Path,
                  use_tiled: bool, alpha: float, alert_after: int):
    try:
        detector = get_detector()

        def progress(frame: int, total: int):
            pct = 100.0 * frame / total if total > 0 else 0.0
            _set_job(job_id, frames_processed=frame, total_frames=total,
                     percent=round(pct, 1))

        summary = run_on_video(
            in_path, out_path, detector,
            use_tiled=use_tiled, alpha=alpha, alert_after=alert_after,
            progress_cb=progress,
        )
        # Make paths web-relative for the template
        summary["output_video_web"] = _static_rel(Path(summary["output_video"]))
        summary["timeline_csv_web"] = _static_rel(Path(summary["timeline_csv"]))
        _set_job(job_id, status="done", summary=summary, percent=100.0)
    except Exception as e:
        app.logger.exception("Video job failed")
        _set_job(job_id, status="error", error=f"{type(e).__name__}: {e}",
                 traceback=traceback.format_exc())


def _static_rel(p: Path) -> str:
    """Turn an absolute path under app/static/ into a URL like /static/results/foo.mp4.

    We build the URL as a plain string instead of using url_for() because
    this function is called from a background thread (the video worker),
    where Flask's application context isn't available. Since the /static/
    prefix is fixed, url_for would only help if we mounted the app under
    a non-root path — which we don't.
    """
    try:
        rel = Path(p).resolve().relative_to((APP_DIR / "static").resolve())
        return "/static/" + str(rel).replace(os.sep, "/")
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    model_ok = Path(DEFAULT_MODEL).exists()
    return render_template(
        "index.html",
        model_path=DEFAULT_MODEL,
        model_ok=model_ok,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({
            "error": f"Unsupported file type: {ext}. "
                     f"Accepted: {sorted(IMAGE_EXTS | VIDEO_EXTS)}"
        }), 400

    # CLI options (optional)
    use_tiled = request.form.get("tile") == "1"
    alpha = float(request.form.get("alpha", 0.1))
    alert_after = int(request.form.get("alert_after", 10))

    # Save upload with a unique prefix
    job_id = secrets.token_hex(6)
    safe_name = f"{job_id}_{secrets.token_hex(2)}{ext}"
    in_path = UPLOAD_DIR / safe_name
    f.save(str(in_path))

    out_stem = in_path.stem + "_out"
    out_path = RESULTS_DIR / (out_stem + ext)

    _set_job(
        job_id,
        status="running",
        kind="video" if ext in VIDEO_EXTS else "image",
        filename=f.filename,
        in_path=str(in_path),
        out_path=str(out_path),
        started_at=datetime.utcnow().isoformat() + "Z",
        percent=0.0,
    )

    if ext in IMAGE_EXTS:
        try:
            detector = get_detector()
            result = run_on_image(in_path, out_path, detector, quiet=True)
            result["output_web"] = _static_rel(out_path)
            result["input_web"] = _static_rel(in_path)
            _set_job(job_id, status="done", result=result, percent=100.0)
        except Exception as e:
            app.logger.exception("Image job failed")
            _set_job(job_id, status="error", error=str(e))
            return jsonify({"error": str(e)}), 500
    else:
        # Fire-and-forget the video worker
        t = threading.Thread(
            target=_video_worker,
            args=(job_id, in_path, out_path, use_tiled, alpha, alert_after),
            daemon=True,
        )
        t.start()

    return jsonify({
        "job_id": job_id,
        "kind": "video" if ext in VIDEO_EXTS else "image",
        "result_url": url_for("result", job_id=job_id),
        "status_url": url_for("status", job_id=job_id),
    })


@app.route("/result/<job_id>")
def result(job_id: str):
    job = _get_job(job_id)
    if not job:
        return f"Job {job_id} not found", 404
    job["in_web"] = _static_rel(Path(job["in_path"])) if job.get("in_path") else None
    job["out_web"] = _static_rel(Path(job["out_path"])) if job.get("out_path") else None
    return render_template("result.html", job=job, job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    # Copy only the lightweight fields — the full summary blob lives
    # on the result page.
    return jsonify({
        "status": job.get("status"),
        "kind": job.get("kind"),
        "percent": job.get("percent", 0.0),
        "frames_processed": job.get("frames_processed", 0),
        "total_frames": job.get("total_frames", 0),
        "error": job.get("error"),
    })


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("PPE_HOST", "127.0.0.1")
    port = int(os.environ.get("PPE_PORT", "5000"))
    debug = os.environ.get("PPE_DEBUG", "").lower() in ("1", "true", "yes")
    print(f" * Model         : {DEFAULT_MODEL}")
    print(f" * Upload limit  : {MAX_UPLOAD_MB} MB")
    print(f" * Results saved : {RESULTS_DIR}")
    app.run(host=host, port=port, debug=debug, threaded=True)
