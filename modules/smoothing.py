"""
Temporal smoothing & per-track history for video mode.

Why this exists
---------------
Running ``score_frame`` independently on every video frame produces a
noisy timeseries: scores bounce because one worker momentarily turns
their back (losing the vest detection), or because the detector
hiccups on a single frame.  The human eye sees a stable scene; the
raw score graph looks like static.

We fix this two ways:

1. **Exponential moving average** over the per-frame score.  A short
   window (~1 second at 25 fps = alpha around 0.1) removes jitter
   without lagging the real signal noticeably.  A sudden violation
   that persists for more than ~5 frames will still show up.

2. **Per-track violation debouncing.**  A worker only counts as "in
   violation" if they've been missing helmet/vest for N consecutive
   frames they were visible.  This matches how an actual compliance
   monitor would behave — we don't want to issue an alert for a
   single-frame occlusion blip.

Both of these need per-video state, so they live in a ``SmoothingState``
object that the caller constructs once and feeds each frame's raw
``score_frame`` output.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackRecord:
    """History for one tracked worker."""
    frames_seen: int = 0
    frames_missing_helmet: int = 0
    frames_missing_vest: int = 0
    consecutive_no_helmet: int = 0
    consecutive_no_vest: int = 0
    alerted_helmet: bool = False   # have we already raised an alert for this?
    alerted_vest: bool = False
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=30))


class SmoothingState:
    """
    Holds running state across frames for one video stream.

    Usage
    -----
    >>> state = SmoothingState(alpha=0.1, alert_after=10)
    >>> for frame in video:
    ...     raw_score, violations, details = score_frame(dets)
    ...     smooth, alerts = state.update(raw_score, details)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        alert_after: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float in (0, 1]
            EMA smoothing factor. 1.0 = no smoothing, lower = smoother.
            At 25 fps, alpha=0.1 gives a roughly 1 s time constant.
        alert_after : int
            A track must be missing helmet/vest for this many
            consecutive observed frames before an alert fires.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.alert_after = alert_after

        self._ema: float | None = None
        self._tracks: dict[Any, TrackRecord] = defaultdict(TrackRecord)
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        raw_score: float,
        details: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """
        Feed one frame's raw output into the smoother.

        Returns
        -------
        smoothed_score : float
        alerts : list[str]
            Newly-triggered violation alerts for this frame (empty if
            none newly crossed the threshold).
        """
        self.frame_count += 1

        # ---- EMA on the overall score ------------------------------
        if self._ema is None:
            self._ema = raw_score
        else:
            self._ema = self.alpha * raw_score + (1.0 - self.alpha) * self._ema

        # ---- Per-track debouncing ----------------------------------
        alerts: list[str] = []
        seen_ids: set = set()

        for w in details.get("workers", []):
            tid = w["worker_id"]
            seen_ids.add(tid)
            rec = self._tracks[tid]
            rec.frames_seen += 1
            rec.recent_scores.append(
                (1 if w["has_helmet"] else 0) + (1 if w["has_vest"] else 0)
            )

            # helmet
            if not w["has_helmet"]:
                rec.consecutive_no_helmet += 1
                rec.frames_missing_helmet += 1
                if (
                    rec.consecutive_no_helmet >= self.alert_after
                    and not rec.alerted_helmet
                ):
                    alerts.append(f"[ALERT] Worker {tid} has been without a helmet for {rec.consecutive_no_helmet} frames")
                    rec.alerted_helmet = True
            else:
                rec.consecutive_no_helmet = 0
                rec.alerted_helmet = False

            # vest
            if not w["has_vest"]:
                rec.consecutive_no_vest += 1
                rec.frames_missing_vest += 1
                if (
                    rec.consecutive_no_vest >= self.alert_after
                    and not rec.alerted_vest
                ):
                    alerts.append(f"[ALERT] Worker {tid} has been without a vest for {rec.consecutive_no_vest} frames")
                    rec.alerted_vest = True
            else:
                rec.consecutive_no_vest = 0
                rec.alerted_vest = False

        # Tracks that weren't seen this frame don't reset — they might
        # come back.  But we do reset their consecutive counters when
        # they've been gone long enough that the SORT tracker drops
        # them.  That's handled implicitly: the tracker hands us a
        # fresh ID on re-entry.

        return round(self._ema, 1), alerts

    # ------------------------------------------------------------------
    # Summary — for the "done" page of the Flask UI / CSV export
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """
        Return a stats dict aggregating across all frames seen so far.
        """
        tracks_summary = []
        for tid, rec in self._tracks.items():
            if rec.frames_seen == 0:
                continue
            tracks_summary.append({
                "worker_id": tid,
                "frames_seen": rec.frames_seen,
                "pct_no_helmet": round(
                    100 * rec.frames_missing_helmet / rec.frames_seen, 1
                ),
                "pct_no_vest": round(
                    100 * rec.frames_missing_vest / rec.frames_seen, 1
                ),
                "alerted_helmet": rec.alerted_helmet or rec.frames_missing_helmet >= self.alert_after,
                "alerted_vest": rec.alerted_vest or rec.frames_missing_vest >= self.alert_after,
            })

        # Sort by worst offender first
        tracks_summary.sort(
            key=lambda t: -(t["pct_no_helmet"] + t["pct_no_vest"])
        )

        return {
            "total_frames": self.frame_count,
            "final_smoothed_score": round(self._ema, 1) if self._ema is not None else 100.0,
            "tracks": tracks_summary,
        }
