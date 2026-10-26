"""Render the repaired clip and a diagnostic strip chart."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .interp import interpolate
from .plan import Op
from .video import FrameReader, FrameWriter, probe


def flow_fill(a: np.ndarray, b: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
    """Default fill: weights-free optical flow. A fill takes two uint8 RGB frames and returns one frame per alpha."""
    return [interpolate(a, b, t) for t in alphas]


class Filler:
    """Fills every gap once, with all of its in-between frames in a single call."""

    def __init__(self, ops: list[Op], fill=None):
        self.fill = fill or flow_fill
        self.alphas: dict[tuple[int, int], list[float]] = {}
        for o in ops:
            if o.op == "interp":
                self.alphas.setdefault((o.gen, o.gen_b), []).append(o.alpha)
        self._done: dict[tuple[int, int], dict[float, np.ndarray]] = {}

    def frame(self, o: Op, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        key = (o.gen, o.gen_b)
        if key not in self._done:
            alphas = self.alphas[key]
            self._done = {key: dict(zip(alphas, self.fill(a, b, alphas)))}
        return self._done[key][o.alpha]


def render(gen_path: str | Path, ops: list[Op], out_path: str | Path, fps: Fraction, codec: str = "prores",
           fill=None) -> int:
    info = probe(gen_path)
    reader = FrameReader(gen_path)
    writer = FrameWriter(out_path, info.width, info.height, fps, codec)
    filler = Filler(ops, fill)
    cache: dict[int, np.ndarray] = {}
    black = np.zeros((info.height, info.width, 3), np.uint8)

    def get(i: int) -> np.ndarray:
        while i not in cache:
            f = reader.read()
            if f is None:
                raise RuntimeError(f"generated clip ended before frame {i}")
            cache[reader.index] = f
        return cache[i]

    try:
        for o in ops:
            if o.op == "copy":
                frame = get(o.gen)
            elif o.op == "interp":
                frame = filler.frame(o, get(o.gen), get(o.gen_b))
            else:
                frame = black
            writer.write(frame)
            keep = min((x.gen for x in ops[o.plate + 1:] if x.gen is not None), default=None)
            if keep is not None:
                for k in [k for k in cache if k < keep]:
                    del cache[k]
    finally:
        reader.close()
        writer.close()
    return writer.count


COLORS = {"copy": (90, 170, 90), "interp": (40, 170, 255), "missing": (60, 60, 230)}


def strip_chart(ops: list[Op], dup: np.ndarray, path: str | Path, width: int = 1600) -> None:
    """Top: gen-minus-plate offset per plate frame. Bottom bars: repair op per plate frame and gen duplicates."""
    n = len(ops)
    h_plot, h_bar = 220, 26
    img = np.full((h_plot + 3 * h_bar + 30, width, 3), 24, np.uint8)
    xs = lambda p: int(p * (width - 1) / max(n - 1, 1))

    offs = [(o.plate, o.gen - o.plate) for o in ops if o.op == "copy"]
    if offs:
        vals = [v for _, v in offs]
        lo, hi = min(vals) - 1, max(vals) + 1
        ys = lambda v: int(h_plot - 10 - (v - lo) * (h_plot - 20) / max(hi - lo, 1))
        for v in range(lo, hi + 1):
            cv2.line(img, (0, ys(v)), (width, ys(v)), (45, 45, 45), 1)
            cv2.putText(img, f"{v:+d}", (4, ys(v) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        for (p0, v0), (p1, v1) in zip(offs, offs[1:]):
            cv2.line(img, (xs(p0), ys(v0)), (xs(p1), ys(v1)), (230, 230, 230), 1)

    y = h_plot
    for o in ops:
        cv2.rectangle(img, (xs(o.plate), y), (xs(o.plate + 1), y + h_bar - 4), COLORS[o.op], -1)
    y += h_bar
    m = len(dup)
    for g in np.flatnonzero(dup):
        x0 = int(g * (width - 1) / max(m - 1, 1))
        cv2.rectangle(img, (x0, y), (x0 + 2, y + h_bar - 4), (200, 80, 200), -1)
    cv2.putText(img, "offset gen-plate (frames) | ops: green copy, orange interp, red missing | magenta: gen duplicates",
                (4, y + h_bar + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
