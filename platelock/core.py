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


def analyze_arrays(
    gen: np.ndarray,
    plate: np.ndarray,
    thumb: tuple[int, int] = features.THUMB,
    motion_weight: float = 0.5,
    max_interp: int = 4,
    mask: np.ndarray | None = None,
    auto_mask: bool = True,
    stiffness: float = 2.0,
    min_evidence: float = 2.0,
) -> Result:
    """gen, plate: float32 grayscale stacks (frames, H, W) in [0, 1].

    mask: optional (H, W) weight in plate space, e.g. an actor matte; it
    replaces the automatic plate-agreement weighting.
    """
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
    for g, p in first.pairs:
        if dup[g] and plate_hold[max(0, p - 1):p + 2].any():
            dup[g] = False
    al, _ = _align(gs, ps, w, motion_weight, stiffness, dup=dup, gap=gap)
    al = gate(al, dup, min_evidence)
    ops = build_plan(al, dup, max_interp=max_interp)
    return Result(al, dup, ops, summarize(al, dup, ops), w)
