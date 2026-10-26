"""Turn an alignment into one repair operation per plate frame."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .align import Alignment


@dataclass
class Op:
    plate: int
    op: str  # copy | interp | missing
    gen: int | None = None
    gen_b: int | None = None
    alpha: float = 0.0
    reason: str = ""
    cost: float | None = None


def build_plan(al: Alignment, dup: np.ndarray, max_interp: int = 4) -> list[Op]:
    n = al.plate_frames
    mapped = al.gen_for_plate()
    good: list[int | None] = [g if g is not None and not dup[g] else None for g in mapped]
    ops: list[Op] = []
    anchors = [p for p in range(n) if good[p] is not None]

    for p in range(n):
        if good[p] is not None:
            ops.append(Op(p, "copy", gen=good[p], cost=float(al.cost[good[p], p])))
            continue
        reason = "duplicate" if mapped[p] is not None else "skipped"
        prev = max((a for a in anchors if a < p), default=None)
        nxt = min((a for a in anchors if a > p), default=None)
        if prev is None or nxt is None:
            ops.append(Op(p, "missing", reason="head" if prev is None else "tail"))
        elif nxt - prev - 1 > max_interp:
            ops.append(Op(p, "missing", reason=f"gap>{max_interp}"))
        else:
            ops.append(Op(p, "interp", gen=good[prev], gen_b=good[nxt],
                          alpha=(p - prev) / (nxt - prev), reason=reason))
    return ops


def segments(al: Alignment, ops: list[Op]) -> list[dict]:
    """Runs of constant gen-plate offset, each with how strongly the data prefers it over offset +/-1.

    confidence is a z-score (mean cost advantage over its standard error);
    below ~2 the offset is a guess, typically in near-static stretches.
    """
    out: list[dict] = []
    copies = [o for o in ops if o.op == "copy"]
    run: list[Op] = []
    for o in copies + [None]:
        if run and (o is None or o.gen - o.plate != run[-1].gen - run[-1].plate):
            off = run[0].gen - run[0].plate
            zs = []
            for step in (-1, 1):
                d = [al.cost[r.gen + step, r.plate] - al.cost[r.gen, r.plate]
                     for r in run if 0 <= r.gen + step < al.gen_frames]
                if len(d) >= 2:
                    zs.append(float(np.mean(d) / (np.std(d) / np.sqrt(len(d)) + 1e-9)))
            out.append({"plate": [run[0].plate, run[-1].plate], "offset": off,
                        "confidence": round(min(zs), 1) if zs else None})
            run = []
        if o is not None:
            run.append(o)
    return out


def summarize(al: Alignment, dup: np.ndarray, ops: list[Op]) -> dict:
    used = {g for g, _ in al.pairs}
    dropped = [g for g in range(al.gen_frames) if g not in used]
    copies = [o for o in ops if o.op == "copy"]
    offsets = [o.gen - o.plate for o in copies]
    missing = [o.plate for o in ops if o.op == "missing"]
    return {
        "plate_frames": al.plate_frames,
        "gen_frames": al.gen_frames,
        "offset_median": float(np.median(offsets)) if offsets else None,
        "offset_range": [int(min(offsets)), int(max(offsets))] if offsets else None,
        "copied": len(copies),
        "interpolated": [o.plate for o in ops if o.op == "interp"],
        "duplicates_in_gen": [int(i) for i in np.flatnonzero(dup)],
        "dropped_gen_frames": dropped,
        "missing_plate_frames": missing,
        "segments": segments(al, ops),
        "complete": not missing,
        "gap_cost": al.gap,
    }


def plan_to_json(ops: list[Op]) -> list[dict]:
    return [{k: v for k, v in asdict(o).items() if v is not None and v != ""} for o in ops]
