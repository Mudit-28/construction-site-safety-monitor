"""
Compliance scoring module.

Major revision for the final submission:

The interim version compared *frame-level totals* (n_helmets vs n_workers,
n_vests vs n_workers). Section 2.3 of the interim report flagged this as
the primary failure mode: at distance the vest box and person box don't
overlap nicely so the system can't tell which worker owns which vest,
producing the reported 63.9% vs. ~80% discrepancy on test9.jpg.

This rewrite does proper *per-worker spatial assignment*:

  1. For each tracked person box, search for helmet boxes whose centre
     falls in the upper ~45% of the person box (the head region).
  2. For each person box, search for vest boxes whose centre falls in
     the middle ~70% of the person box (the torso region).
  3. Each PPE box gets assigned to at most one worker — the one whose
     region contains it most centrally (min centre-distance tiebreak).
     This prevents one vest from being credited to multiple workers.
  4. Score is then computed per-worker with fractional deductions.

The "explicit" no-helmet / no-vest classes are now also assigned
per-worker the same way, and we only apply their deduction if the
positive class (helmet / vest) is *absent* for that worker — so if the
model is uncertain and fires both "vest" and "no-vest" on the same
torso (exactly the test9.jpg case), the positive detection wins.

Returns
-------
score : float in [0, 100]
violations : list of human-readable strings
details : dict with per-worker breakdown (useful for logs & the UI)
"""

from typing import Any

# Relative weight of each category.  `helmet + vest` sum to 0.75; the
# remaining 0.25 is reserved for zone / future sensors.  We scale the
# active-category weights so they always sum to 100 when zone check
# isn't active — otherwise removing a category silently caps the score.
WEIGHTS = {
    "helmet": 0.40,
    "vest": 0.35,
    "zone": 0.25,
}

# Confidence floors.  The interim report §2.1 (turban false positive)
# recommends tightening the helmet threshold; we do that here.  The
# detector still reports lower-confidence boxes for visual feedback,
# but compliance only trusts them above these thresholds.
HELMET_CONF_MIN = 0.45     # was 0.25 implicitly — raised to reduce turban FPs
VEST_CONF_MIN = 0.35
NO_PPE_CONF_MIN = 0.55     # explicit violation classes must be confident


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _centre(b: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _contains_point(box: list[float], pt: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = pt
    return x1 <= x <= x2 and y1 <= y <= y2


def _head_region(person_box: list[float]) -> list[float]:
    """Top 45% of the person box — where a helmet should be."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1, x2, y1 + 0.45 * h]


def _torso_region(person_box: list[float]) -> list[float]:
    """Rows 20%-85% of the person box — where a vest should be.

    A vest sits on the torso, which starts roughly at the neck and
    ends at the waist, not including head or legs.  This window is
    deliberately generous to handle small/distant worker boxes where
    the exact proportions break down.
    """
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1 + 0.20 * h, x2, y1 + 0.85 * h]


def _box_dist(a: list[float], b: list[float]) -> float:
    """Centre-to-centre distance, L2."""
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Assignment: match PPE boxes to worker boxes
# ---------------------------------------------------------------------------

def _assign(
    ppe_boxes: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    region_fn,
) -> dict[int, dict[str, Any] | None]:
    """Greedily assign each PPE box to the nearest worker whose
    region contains the PPE centre.

    Returns a dict mapping worker_index -> assigned PPE detection (or None).
    """
    # Pre-compute worker regions.
    regions = [region_fn(w["bbox"]) for w in workers]

    # Sort PPE boxes by confidence (descending) — a confident box gets
    # to claim its worker before a shaky one does.
    ppe_sorted = sorted(ppe_boxes, key=lambda d: -d["conf"])

    assignment: dict[int, dict[str, Any] | None] = {i: None for i in range(len(workers))}
    claimed_ppe: set[int] = set()  # ids of PPE boxes already assigned

    for ppe_idx, ppe in enumerate(ppe_sorted):
        if ppe_idx in claimed_ppe:
            continue
        ppe_centre = _centre(ppe["bbox"])
        best_w = -1
        best_dist = float("inf")
        for wi, region in enumerate(regions):
            if assignment[wi] is not None:
                continue            # that worker is already taken
            if not _contains_point(region, ppe_centre):
                continue
            d = _box_dist(ppe["bbox"], workers[wi]["bbox"])
            if d < best_dist:
                best_dist = d
                best_w = wi
        if best_w >= 0:
            assignment[best_w] = ppe
            claimed_ppe.add(ppe_idx)

    return assignment


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_frame(
    detections: list[dict[str, Any]],
    zone_active: bool = True,
) -> tuple[float, list[str], dict[str, Any]]:
    """Compute a compliance score for a single frame.

    Parameters
    ----------
    detections : list of {'class', 'conf', 'bbox'} dicts.
        'bbox' is [x1, y1, x2, y2] in pixel coords.
    zone_active : whether the zone-active weight contributes. Unused
        for now but kept for future expansion (e.g. "worker outside
        construction zone" deduction).

    Returns
    -------
    score : float in [0, 100]
    violations : list[str] human-readable
    details : dict with per-worker breakdown for the UI / CSV log
    """
    # ------------------------------------------------------------------
    # Bucket detections by class + confidence filter.
    # ------------------------------------------------------------------
    persons = [d for d in detections if d["class"] == "person"]
    helmets = [d for d in detections if d["class"] == "helmet" and d["conf"] >= HELMET_CONF_MIN]
    vests = [d for d in detections if d["class"] == "vest" and d["conf"] >= VEST_CONF_MIN]
    no_helm = [d for d in detections if d["class"] == "no-helmet" and d["conf"] >= NO_PPE_CONF_MIN]
    no_vest = [d for d in detections if d["class"] == "no-vest" and d["conf"] >= NO_PPE_CONF_MIN]

    n_workers = len(persons)

    if n_workers == 0:
        # No workers in frame → no work being done → full marks.
        return 100.0, [], {"workers": [], "n_workers": 0}

    # ------------------------------------------------------------------
    # Per-worker assignment.
    # ------------------------------------------------------------------
    helmet_map = _assign(helmets, persons, _head_region)
    vest_map = _assign(vests, persons, _torso_region)
    no_helm_map = _assign(no_helm, persons, _head_region)
    no_vest_map = _assign(no_vest, persons, _torso_region)

    # ------------------------------------------------------------------
    # Compute per-worker compliance.
    # ------------------------------------------------------------------
    worker_details: list[dict[str, Any]] = []
    missing_helmets = 0
    missing_vests = 0
    violation_strs: list[str] = []

    for i, p in enumerate(persons):
        has_helmet = helmet_map[i] is not None
        has_vest = vest_map[i] is not None
        # Only count an explicit violation if the positive class *wasn't*
        # also detected for this worker.  This fixes the test9.jpg case
        # where vest and no-vest fired on the same torso.
        explicit_no_helm = (no_helm_map[i] is not None) and not has_helmet
        explicit_no_vest = (no_vest_map[i] is not None) and not has_vest

        wid = p.get("track_id", f"p{i}")
        worker_details.append({
            "worker_id": wid,
            "bbox": p["bbox"],
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "explicit_no_helmet": explicit_no_helm,
            "explicit_no_vest": explicit_no_vest,
            "helmet_conf": helmet_map[i]["conf"] if has_helmet else None,
            "vest_conf": vest_map[i]["conf"] if has_vest else None,
        })

        if not has_helmet:
            missing_helmets += 1
            reason = "no-helmet (explicit)" if explicit_no_helm else "no helmet detected"
            violation_strs.append(f"Worker {wid}: {reason}")
        if not has_vest:
            missing_vests += 1
            reason = "no-vest (explicit)" if explicit_no_vest else "no vest detected"
            violation_strs.append(f"Worker {wid}: {reason}")

    # ------------------------------------------------------------------
    # Score calculation.
    # ------------------------------------------------------------------
    if zone_active:
        # The zone weight (0.25) doesn't produce a deduction yet — it's
        # reserved — so the effective penalty pool is 0.75 of 100.
        helmet_weight = WEIGHTS["helmet"] * 100
        vest_weight = WEIGHTS["vest"] * 100
    else:
        # Renormalise so helmet+vest span the full 100.
        total = WEIGHTS["helmet"] + WEIGHTS["vest"]
        helmet_weight = (WEIGHTS["helmet"] / total) * 100
        vest_weight = (WEIGHTS["vest"] / total) * 100

    helmet_deduction = helmet_weight * (missing_helmets / n_workers)
    vest_deduction = vest_weight * (missing_vests / n_workers)

    score = 100.0 - helmet_deduction - vest_deduction

    # Floor at 0, ceiling at 100. Round to one decimal for display.
    score = max(0.0, min(100.0, score))

    details = {
        "n_workers": n_workers,
        "missing_helmets": missing_helmets,
        "missing_vests": missing_vests,
        "helmet_deduction": round(helmet_deduction, 1),
        "vest_deduction": round(vest_deduction, 1),
        "workers": worker_details,
    }

    return round(score, 1), violation_strs, details
