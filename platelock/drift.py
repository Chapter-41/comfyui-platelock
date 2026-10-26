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


def measure(plate: np.ndarray, result: np.ndarray, valid: list[bool], plate_width: int) -> list[dict | None]:
    """One entry per plate frame (None where the result has no frame)."""
    ep, er = edge_maps(plate), edge_maps(result)
    scale = plate_width / ep.shape[2]
    window = cv2.createHanningWindow(ep.shape[1:][::-1], cv2.CV_32F)
    n = min(len(ep), len(er))
    shifts = {}
    for t in range(n):
        if valid[t]:
            (dx, dy), response = cv2.phaseCorrelate(ep[t], er[t], window)
            shifts[t] = (dx, dy, response)
    sure_shifts = [(dx, dy) for dx, dy, r in shifts.values() if r >= MIN_RESPONSE]
    ox, oy = np.median(sure_shifts, axis=0) if sure_shifts else (0.0, 0.0)
    undo = np.float32([[1, 0, -ox], [0, 1, -oy]])
    er = np.stack([cv2.warpAffine(f, undo, f.shape[::-1], borderMode=cv2.BORDER_REFLECT) for f in er])
    out: list[dict | None] = []
    for t in range(n):
        if not valid[t]:
            out.append(None)
            continue
        ncc = {k: float((er[t] * ep[t + k]).mean()) for k in range(-MAX_DT, MAX_DT + 1) if 0 <= t + k < len(ep)}
        best = max(ncc, key=ncc.get)
        sure = max(ncc.values()) - min(ncc.values()) > 0.04
        dx, dy, response = shifts[t]
        reliable = response >= MIN_RESPONSE
        out.append({"dt": int(best) if sure and ncc[best] - ncc[0] > 0.01 else 0, "sure": bool(sure),
                    "match": round(ncc[0], 3),
                    "shift": round(float(np.hypot(dx, dy)) * scale, 2) if reliable else None,
                    "dx": round(dx * scale, 2) if reliable else None, "dy": round(dy * scale, 2) if reliable else None})
    return out + [None] * (len(valid) - n)


def summary(drift: list[dict | None]) -> dict:
    m = [(i, d) for i, d in enumerate(drift) if d]
    if not m:
        return {"measured": 0}
    placed = [(i, d) for i, d in m if d["shift"] is not None]
    out = {"measured": len(m), "in_sync": sum(1 for _, d in m if d["dt"] == 0),
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
    lines = [f"timing: {s['in_sync']}/{s['measured']} frames show their own plate frame"]
    if s["off_in_time"]:
        lines.append(f"CHECK {len(s['off_in_time'])} frames look like a neighbouring plate frame: {s['off_in_time'][:30]}")
    if s["unsure"]:
        lines.append(f"{s['unsure']} frames barely move, so their timing can't be verified")
    if "median_shift_px" in s:
        lines.append(f"position: median {s['median_shift_px']:g} px off the plate, 95% within {s['p95_shift_px']:g} px "
                     f"(worst {s['worst_shift_px']:g} px at frame {s['worst_frame']})")
        if s["median_shift_px"] > 4:
            dx, dy = s["offset_px"]
            lines.append(f"CHECK the whole result sits {dx:+g} px x, {dy:+g} px y off the plate (the generator reframed it)")
    return lines
