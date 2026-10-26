"""Monotonic alignment of generated frames to plate frames.

Dynamic time warping over a cost matrix C[g, p]. Each step advances the
generated clip by 1..MAX_STEP frames and the plate by 1..MAX_STEP frames:
- (1, 1): normal frame
- (1, k): plate frames skipped by the generation (to be interpolated)
- (k, 1): extra generated frames (dropped)
Unmatched frames cost `gap` each, so skipping is never free. The path may start
and end anywhere, which absorbs global offsets and missing head/tail frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_STEP = 3


@dataclass
class Alignment:
    pairs: list[tuple[int, int]]  # (gen frame, plate frame), strictly increasing in both
    plate_frames: int
    gen_frames: int
    gap: float
    cost: np.ndarray = field(repr=False)

    def gen_for_plate(self) -> list[int | None]:
        out: list[int | None] = [None] * self.plate_frames
        for g, p in self.pairs:
            out[p] = g
        return out


def gap_cost(c: np.ndarray, stiffness: float = 2.0, k: float = 0.6) -> float:
    """Per-frame penalty for an unmatched frame.

    The base sits between a typical good match and a typical random pairing.
    It must stay well above the cost of a merely bad match (shimmer, occlusion),
    or the path drops those frames and zigzags; a real skip shifts every later
    frame and accumulates far more cost than any single gap, so stiffness
    removes noise without hiding real retiming.
    """
    good = float(np.median(c.min(axis=1)))
    rand = float(np.median(c))
    return stiffness * (good + k * max(rand - good, 1e-3))


def align(c: np.ndarray, gap: float | None = None, max_edge: int | None = None, transition=None) -> Alignment:
    """transition(g, dg, dp) -> per-p extra cost for stepping into (g, p); see features.MotionTransition."""
    m, n = c.shape
    gap = gap_cost(c) if gap is None else gap
    max_edge = max(m, n) if max_edge is None else max_edge
    inf = np.float64(np.inf)

    d = np.full((m, n), inf)
    back_g = np.zeros((m, n), np.int16)
    back_p = np.zeros((m, n), np.int16)

    # Start cell (g0, p0) leaves g0 gen frames and p0 plate frames unmatched.
    p_idx = np.arange(n)
    start_p = np.where(p_idx <= max_edge, gap * p_idx, inf)

    for g in range(m):
        best = np.where(g <= max_edge, gap * g + start_p, inf)  # start here
        bg = np.zeros(n, np.int16)
        bp = np.zeros(n, np.int16)
        for dg in range(1, MAX_STEP + 1):
            if g - dg < 0:
                break
            prev = d[g - dg]
            for dp in range(1, MAX_STEP + 1):
                cand = np.full(n, inf)
                cand[dp:] = prev[:-dp] + gap * ((dg - 1) + (dp - 1))
                if transition is not None:
                    cand += transition(g, dg, dp)
                better = cand < best
                best = np.where(better, cand, best)
                bg = np.where(better, dg, bg)
                bp = np.where(better, dp, bp)
        d[g] = best + c[g]
        back_g[g] = bg
        back_p[g] = bp

    # End cell leaves the remaining frames on both sides unmatched.
    tail_g = gap * (m - 1 - np.arange(m))[:, None]
    tail_p = gap * (n - 1 - np.arange(n))[None, :]
    end = d + tail_g + tail_p
    g, p = np.unravel_index(int(np.argmin(end)), end.shape)

    pairs = []
    while True:
        pairs.append((int(g), int(p)))
        dg, dp = int(back_g[g, p]), int(back_p[g, p])
        if dg == 0:
            break
        g, p = g - dg, p - dp
    pairs.reverse()
    return Alignment(pairs=pairs, plate_frames=n, gen_frames=m, gap=gap, cost=c)
