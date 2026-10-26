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


def analyze_arrays(
    gen: np.ndarray,
    plate: np.ndarray,
    thumb: tuple[int, int] = features.THUMB,
    motion_weight: float = 0.5,
    max_interp: int = 4,
    mask: np.ndarray | None = None,
    auto_mask: bool = True,
    stiffness: float = 2.0,
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
    ops = build_plan(al, dup, max_interp=max_interp)
    return Result(al, dup, ops, summarize(al, dup, ops), w)
