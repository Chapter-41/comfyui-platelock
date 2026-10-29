"""How exactly a result sits on the plate, frame by frame.

A restyled generation never matches the plate pixel for pixel, so both are
compared as edge maps (outlines survive restyling). Per plate frame:

- shift: sub-pixel spatial offset of the result against the plate (phase correlation), in plate pixels
- dt: which plate frame (within +-2) this result frame actually shows (0 = in sync)
- sure: False where the shot barely moves, so neighbouring frames look alike and dt can't be told
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import inspect_geometry

MAX_DT = 2
MIN_RESPONSE = 0.1


def edge_maps(rgb: np.ndarray, width: int = 320) -> np.ndarray:
    n, h, w, _ = rgb.shape
    size = (width, max(2, round(h * width / w)))
    out = np.empty((n, size[1], size[0]), np.float32)
    for i, f in enumerate(rgb):
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                             .astype(np.float32), (0, 0), 1.2)
        m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        out[i] = (m - m.mean()) / (m.std() + 1e-6)
    return out


def measure(plate: np.ndarray, result: np.ndarray, valid: list[bool], plate_width: int,
            mask: np.ndarray | None = None) -> list[dict | None]:
    """Independent timing/position evidence. Scores are heuristics, not probabilities.

    Timing uses a five-frame neighbourhood and motion-weighted edges. A large
    score *range* is not confidence: the winner must beat the runner-up. Spatial
    offsets are removed before comparison; stationary frames remain unmeasurable.
    """
    ep, er = edge_maps(plate), edge_maps(result)
    if er.shape[1:] != ep.shape[1:]:
        er = np.stack([cv2.resize(f, ep.shape[:0:-1]) for f in er])
    scale = plate_width / ep.shape[2]
    h, w = ep.shape[1:]
    roi = np.ones((h, w), np.float32) if mask is None else cv2.resize(mask.astype(np.float32), (w, h)).clip(0, 1)
    if roi.mean() < .01:
        raise ValueError("Matching mask covers too little of the plate (minimum 1%).")
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    n = min(len(ep), len(er), len(valid))
    shifts, aligned = {}, er.copy()
    for t in range(n):
        if not valid[t]:
            continue
        (dx, dy), response = cv2.phaseCorrelate((ep[t]*roi).copy(), (er[t]*roi).copy(), window)
        reliable = np.isfinite([dx, dy, response]).all() and response >= MIN_RESPONSE and abs(dx) < w*.1 and abs(dy) < h*.1
        shifts[t] = (dx, dy, bool(reliable))
        if reliable:
            aligned[t] = cv2.warpAffine(er[t], np.float32([[1,0,-dx],[0,1,-dy]]), (w,h), borderMode=cv2.BORDER_REFLECT)
    # Preserve isolated timing errors: removing each frame's own translation
    # would absorb a one-frame jump during a pan into a spatial correction.
    trusted = [(dx,dy) for dx,dy,ok in shifts.values() if ok]
    ox,oy = np.median(trusted,axis=0) if trusted else (0.,0.)
    aligned = np.stack([cv2.warpAffine(f,np.float32([[1,0,-ox],[0,1,-oy]]),(w,h),borderMode=cv2.BORDER_REFLECT) for f in er])
    def corr(a, b, weights):
        den = weights.sum() + 1e-8
        aa, bb = a - (a*weights).sum()/den, b - (b*weights).sum()/den
        return float((aa*bb*weights).sum() / (np.sqrt((aa*aa*weights).sum()*(bb*bb*weights).sum())+1e-8))
    out = []
    for t in range(n):
        if not valid[t]:
            out.append(None)
            continue
        lo, hi = max(0,t-2), min(n,t+3)
        motion = np.std(ep[lo:hi], axis=0)
        activity = float((motion*roi).sum()/(roi.sum()+1e-8))
        weights = roi * (.05 + motion)
        scores = {}
        for k in range(-MAX_DT, MAX_DT+1):
            samples = [corr(aligned[j], ep[j+k], weights) for j in range(lo,hi)
                       if valid[j] and 0 <= j+k < len(ep)]
            if samples and 0 <= t+k < len(ep):
                scores[k] = .8*corr(aligned[t],ep[t+k],weights) + .2*float(np.mean(samples))
        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        margin = scores[best]-scores[ranked[1]] if len(ranked)>1 else 0
        exact_matches = [k for k in scores if corr(er[t],ep[t+k],roi) > .9999]
        reason = ("Not enough motion to determine timing" if activity < .025 else
                  "Too little shared detail" if scores[best] < .35 else
                  "Two frame matches are equally plausible" if margin < .008 else "")
        # Direct equality verifies image correspondence, not unique timing in a hold.
        if len(exact_matches) == 1 and activity >= .025:
            reason, best = "", exact_matches[0]
        sure = not reason
        dx, dy, reliable = shifts[t]
        out.append({"dt": int(best) if sure else 0, "sure": sure,
                    "timing_reason": reason, "timing_status": "measured" if sure else "not_measurable",
                    "match": round(scores.get(0,0),3), "margin": round(margin,4),
                    "position_reason": "" if reliable else "Too little shared detail for position",
                    "shift": round(float(np.hypot(dx,dy))*scale,2) if reliable else None,
                    "dx": round(dx*scale,2) if reliable else None,
                    "dy": round(dy*scale,2) if reliable else None,
                    "geometry": inspect_geometry(ep[t], er[t], roi, scale) if reliable else None})
    return out + [None]*(len(valid)-n)


def summary(drift: list[dict | None]) -> dict:
    m = [(i, d) for i, d in enumerate(drift) if d]
    if not m:
        return {"measured": 0}
    placed = [(i, d) for i, d in m if d["shift"] is not None]
    out = {"measured": len(m), "in_sync": sum(1 for _, d in m if d["sure"] and d["dt"] == 0),
           "off_in_time": [i for i, d in m if d["dt"]], "unsure": sum(1 for _, d in m if not d["sure"])}
    if placed:
        shifts = np.array([d["shift"] for _, d in placed])
        worst = max(placed, key=lambda x: x[1]["shift"])
        out.update(median_shift_px=round(float(np.median(shifts)), 1), p95_shift_px=round(float(np.percentile(shifts, 95)), 1),
                   worst_shift_px=worst[1]["shift"], worst_frame=worst[0],
                   offset_px=[round(float(np.median([d["dx"] for _, d in placed])), 1),
                              round(float(np.median([d["dy"] for _, d in placed])), 1)])
    return out


def report_lines(s: dict) -> list[str]:
    if not s.get("measured"):
        return []
    lines = [f"timing: {s['in_sync']}/{s['measured']} frames are consistent with their plate timing (estimate)"]
    if s["off_in_time"]:
        lines.append(f"CHECK {len(s['off_in_time'])} frames look like a neighbouring plate frame: {s['off_in_time'][:30]}")
    if s["unsure"]:
        lines.append(f"{s['unsure']} frames have insufficient evidence to determine timing (not detected errors)")
    if "median_shift_px" in s:
        lines.append(f"position: median {s['median_shift_px']:g} px off the plate, 95% within {s['p95_shift_px']:g} px "
                     f"(worst {s['worst_shift_px']:g} px at frame {s['worst_frame']})")
        if s["median_shift_px"] > 4:
            dx, dy = s["offset_px"]
            lines.append(f"CHECK the whole result sits {dx:+g} px x, {dy:+g} px y off the plate (the generator reframed it)")
    return lines
