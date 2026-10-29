"""Array-level entry point shared by the CLI and the ComfyUI nodes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features
from .align import MAX_STEP, Alignment, align, gap_cost
from .plan import Op, build_plan, summarize


@dataclass
class Result:
    alignment: Alignment
    duplicates: np.ndarray
    ops: list[Op]
    summary: dict
    weights: np.ndarray


def _align(gs, ps, w, motion_weight, stiffness, dup=None, gap=None):
    gd, pd = features.descriptors(gs, w), features.descriptors(ps, w)
    c = features.cost_matrix(gd, pd)
    gap = gap_cost(c, stiffness) if gap is None else gap
    if dup is not None:
        # A repeated frame is never the best witness for a plate frame if the original is available.
        c[dup] += 0.5 * gap
    trans = features.MotionTransition(gd, pd, motion_weight, MAX_STEP)
    return align(c, gap=gap, transition=trans), gap


def gate(al: Alignment, dup: np.ndarray, z_min: float) -> Alignment:
    """Keep a change of offset only where the data clearly prefers it.

    Each run of constant offset is compared with the offset in force before it,
    over the run's own frames. Without clear evidence (z < z_min) the run keeps
    the previous offset: frames are left as delivered instead of being edited
    on a guess. The first run is kept as is, since a head offset has no
    previous timing to fall back to.
    """
    if z_min <= 0 or len(al.pairs) < 2:
        return al
    c = al.cost
    runs: list[list[tuple[int, int]]] = []
    for g, p in al.pairs:
        if runs and g - p == runs[-1][-1][0] - runs[-1][-1][1]:
            runs[-1].append((g, p))
        else:
            runs.append([(g, p)])
    offsets = [runs[0][0][0] - runs[0][0][1]]
    for run in runs[1:]:
        new = run[0][0] - run[0][1]
        old = offsets[-1]
        d = [c[p + old, p] - c[g, p] for g, p in run if 0 <= p + old < al.gen_frames]
        z = np.mean(d) / (np.std(d) / np.sqrt(len(d)) + 1e-9) if len(d) >= 2 else 0.0
        offsets.append(new if z >= z_min or len(d) < len(run) else old)

    original = set(al.pairs)
    pairs, last_g, prev_end = [], -1, runs[0][0][1] - 1
    for run, off in zip(runs, offsets):
        own = run[0][0] - run[0][1]
        # A reverted run also takes back the plate frames skipped just before it.
        start = prev_end + 1 if off != own else run[0][1]
        for p in range(start, run[-1][1] + 1):
            g = p + off
            if g > last_g and 0 <= g < al.gen_frames and not (dup[g] and (g, p) not in original):
                pairs.append((g, p))
                last_g = g
        prev_end = run[-1][1]
    return Alignment(pairs=pairs, plate_frames=al.plate_frames, gen_frames=al.gen_frames, gap=al.gap, cost=al.cost)


def _edges(f: np.ndarray) -> np.ndarray:
    cv2 = features.cv2
    g = cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def estimate_reframe(gen: np.ndarray, plate: np.ndarray, pairs: list[tuple[int, int]], weights: np.ndarray,
                     samples: int = 12) -> np.ndarray | None:
    """One zoom + shift for the whole clip, mapping gen pixels onto the plate (2x3, for WARP_INVERSE_MAP).

    Fitted only where the gen follows the plate (weights, e.g. the actors; an invented background
    would pull it off) and on frames where the plate barely moves, so a pairing that is a frame off
    hardly biases it. None when the gen already sits on the plate or the fits disagree.
    """
    cv2 = features.cv2
    if len(pairs) < 8:
        return None
    h, w = plate.shape[1] // 2, plate.shape[2] // 2
    small = lambda f: _edges(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
    mask = (cv2.resize(weights, (w, h)) > 0.15).astype(np.uint8)
    if mask.mean() < 0.02:
        return None
    motion = features.frame_deltas(plate)
    chosen = sorted(pairs, key=lambda gp: motion[gp[1]])[:max(samples * 3, len(pairs) // 3)]
    chosen = sorted(chosen, key=lambda gp: gp[1])[::max(1, len(chosen) // samples)][:samples]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
    found = []
    for g, p in chosen:
        m = np.eye(2, 3, dtype=np.float32)
        try:
            _, m = cv2.findTransformECC(small(plate[p]), small(gen[g]), m, cv2.MOTION_AFFINE, criteria, mask, 5)
        except cv2.error:
            continue
        found.append(m)
    if len(found) < samples // 2:
        return None
    found = np.stack(found)
    zooms = np.sqrt(np.abs(np.linalg.det(found[:, :, :2])))
    shifts = found[:, :, 2]
    zoom, shift = float(np.median(zooms)), np.median(shifts, axis=0)
    if np.median(np.abs(shifts - shift)) > 1.0 or np.median(np.abs(zooms - zoom)) > 0.015:
        return None
    m = np.float32([[zoom, 0, 2 * shift[0]], [0, zoom, 2 * shift[1]]])
    if abs(zoom - 1) < 0.004 and np.hypot(*m[:, 2]) < 0.5:
        return None
    return m


def estimate_feature_reframe(gen, plate, pairs, weights, samples=16):
    """Fallback when edge ECC cannot converge: robust shared-detail scale/translation.

    Initial pairs can be temporally imperfect. Require a consistent transform
    across the shot, then re-run correspondence; never reuse these pairs as truth.
    """
    from .framing import correspondences
    if len(pairs) < 12:
        return None
    selected=[pairs[i] for i in np.unique(np.linspace(0,len(pairs)-1,min(samples,len(pairs))).astype(int))]
    found=[]
    roi=features.cv2.resize(weights.astype(np.float32),plate.shape[1:][::-1])
    rgb=lambda f: np.repeat(np.rint(np.clip(f,0,1)*255).astype(np.uint8)[...,None],3,axis=2)
    for g,p in selected:
        fit=correspondences(rgb(plate[p]),rgb(gen[g]),roi)
        if fit is not None: found.append((p,fit[2],fit[3]))
    if len(found)<8 or found[-1][0]-found[0][0]<len(plate)*.65:
        return None
    z=float(np.median([x[1] for x in found]));t=np.median([x[2] for x in found],axis=0)
    if np.median(np.abs(np.array([x[1] for x in found])-z))>.01:
        return None
    if np.percentile(np.linalg.norm(np.array([x[2] for x in found])-t,axis=1),75)*plate.shape[2]>1.5:
        return None
    if abs(z-1)<.004 and np.linalg.norm(t)*plate.shape[2]<.5:
        return None
    return np.float32([[z,0,t[0]*plate.shape[2]],[0,z,t[1]*plate.shape[2]]])


def _match_cost(r: "Result") -> float:
    return float(np.mean([r.alignment.cost[g, p] for g, p in r.alignment.pairs])) if r.alignment.pairs else 1.0


def _warp(frames: np.ndarray, m: np.ndarray) -> np.ndarray:
    cv2 = features.cv2
    h, w = frames.shape[1:3]
    return np.stack([cv2.warpAffine(f, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REFLECT) for f in frames])


def analyze_arrays(
    gen: np.ndarray,
    plate: np.ndarray,
    thumb: tuple[int, int] = features.THUMB,
    motion_weight: float = 0.5,
    max_interp: int = 4,
    mask: np.ndarray | None = None,
    auto_mask: bool = True,
    stiffness: float = 2.0,
    min_evidence: float = 0.5,
    reframe: bool = True,
) -> Result:
    """gen, plate: float32 grayscale stacks (frames, H, W) in [0, 1].

    mask: optional (H, W) weight in plate space, e.g. an actor matte; it
    replaces the automatic plate-agreement weighting.
    reframe: undo a zoom or shift the generator applied to the whole shot before matching;
    otherwise a reframed frame can look more like a neighbouring plate frame than its own.
    """
    kw = dict(thumb=thumb, motion_weight=motion_weight, max_interp=max_interp, mask=mask, auto_mask=auto_mask,
              stiffness=stiffness, min_evidence=min_evidence)
    if gen.shape[1:] != plate.shape[1:]:
        gen = np.stack([features.cv2.resize(f, plate.shape[1:][::-1], interpolation=features.cv2.INTER_AREA) for f in gen])
    r = analyze_arrays(gen, plate, reframe=False, **kw) if reframe else None
    if r is None:
        return _analyze(gen, plate, **kw)
    m = estimate_reframe(gen, plate, r.alignment.pairs, r.weights)
    if m is None:
        m = estimate_feature_reframe(gen, plate, r.alignment.pairs, r.weights)
    if m is None:
        return r
    warped = _analyze(_warp(gen, m), plate, **kw)
    if _match_cost(warped) > 1.05 * _match_cost(r):
        return r
    w = plate.shape[2]
    warped.summary["reframe"] = {"zoom": round(float(np.sqrt(abs(np.linalg.det(m[:, :2])))), 4),
                                 "shift_x": round(float(m[0, 2]) / w, 5), "shift_y": round(float(m[1, 2]) / w, 5)}
    return warped


def _analyze(gen, plate, thumb, motion_weight, max_interp, mask, auto_mask, stiffness, min_evidence) -> Result:
    gs, ps = features.structure_maps(gen, thumb), features.structure_maps(plate, thumb)
    if mask is not None:
        w = np.clip(features.cv2.resize(mask.astype(np.float32), thumb, interpolation=features.cv2.INTER_AREA), 0, 1)
    elif auto_mask:
        w = features.agreement_weights(gs, ps)
    else:
        w = np.ones(gs.shape[1:], np.float32)

    first, gap = _align(gs, ps, w, motion_weight, stiffness)
    if mask is None and auto_mask:
        w = features.agreement_weights(gs, ps, first.pairs)
        first, gap = _align(gs, ps, w, motion_weight, stiffness)

    dup = features.detect_duplicates(gen)
    # A gen pause where the plate pauses too is faithful, not a repeat.
    plate_hold = features.holds(plate)
    plate_repeat = features.detect_duplicates(plate)
    for g, p in first.pairs:
        if dup[g] and (plate_hold[max(0, p - 1):p + 2].any() or plate_repeat[p]):
            dup[g] = False
    al, _ = _align(gs, ps, w, motion_weight, stiffness, dup=dup, gap=gap)
    al = gate(al, dup, min_evidence)
    ops = build_plan(al, dup, max_interp=max_interp)
    return Result(al, dup, ops, summarize(al, dup, ops), w)
