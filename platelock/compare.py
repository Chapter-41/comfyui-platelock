"""Human-readable previews: a one-image summary and a labelled side-by-side video."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .plan import Op
from .render import Filler
from .video import FrameWriter

GREY, ORANGE, RED, BLUE, PURPLE = (120, 120, 120), (255, 160, 40), (230, 60, 60), (80, 160, 255), (190, 110, 255)
BG, FG, DIM = (22, 22, 22), (235, 235, 235), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_DUPLEX


def _text(img, s, xy, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def locked_frames(gen: np.ndarray, ops: list[Op], fill=None, all_ops: list[Op] | None = None) -> list[np.ndarray | None]:
    """all_ops: the full plan when `ops` is a subset, so a gap is filled with all its frames at once."""
    filler = Filler(all_ops or ops, fill)
    out = []
    for o in ops:
        if o.op == "copy":
            out.append(gen[o.gen])
        elif o.op == "interp":
            out.append(filler.frame(o, gen[o.gen], gen[o.gen_b]))
        else:
            out.append(None)
    return out


def describe(o: Op, generated=()) -> tuple[str, tuple]:
    if o.plate in generated:
        return "GENERATED - video model filled the gap", PURPLE
    if o.op == "copy":
        return f"source: AI frame {o.gen}", FG
    if o.op == "interp":
        why = "generator repeated a frame here" if o.reason == "duplicate" else "generator skipped this frame"
        return f"FILLED - {why}", ORANGE
    return "MISSING - never generated", RED


def headline(summary: dict, generated=()) -> list[tuple[str, tuple]]:
    s = summary
    dup = set(s["duplicates_in_gen"])
    removed = s["dropped_gen_frames"]
    lines = [(f"plate {s['plate_frames']} frames   generated {s['gen_frames']} frames   ->   output {s['plate_frames']} frames", FG)]
    facts = []
    filled = [p for p in s["interpolated"] if p not in generated]
    if filled:
        facts.append((f"{len(filled)} filled", ORANGE))
    if generated:
        facts.append((f"{len(generated)} generated", PURPLE))
    missing = [p for p in s["missing_plate_frames"] if p not in generated]
    if missing:
        facts.append((f"{len(missing)} missing", RED))
    extra = [g for g in removed if g not in dup]
    if extra:
        facts.append((f"{len(extra)} extra removed", BLUE))
    dups = [g for g in removed if g in dup]
    if dups:
        facts.append((f"{len(dups)} duplicates removed", BLUE))
    if not facts:
        facts.append(("already in sync, nothing changed", FG))
    lines.append(facts)
    return lines


def _removed_positions(ops: list[Op], removed: list[int]) -> list[int]:
    """Plate position at which each removed generated frame would have appeared."""
    copies = [(o.gen, o.plate) for o in ops if o.op == "copy"]
    pos = []
    for g in removed:
        before = [p for gg, p in copies if gg < g]
        pos.append(before[-1] if before else 0)
    return pos


def timeline(ops: list[Op], summary: dict, width: int, cursor: int | None = None, generated=()) -> np.ndarray:
    n = len(ops)
    h = 46
    img = np.full((h, width, 3), BG, np.uint8)
    x = lambda p: int(round(p * (width - 1) / max(n, 1)))
    for o in ops:
        color = PURPLE if o.plate in generated else GREY if o.op == "copy" else ORANGE if o.op == "interp" else RED
        cv2.rectangle(img, (x(o.plate), 14), (max(x(o.plate + 1) - 1, x(o.plate)), 34), color, -1)
    for p in _removed_positions(ops, summary["dropped_gen_frames"]):
        cv2.line(img, (x(p), 2), (x(p), 12), BLUE, 2)
    if cursor is not None:
        cv2.line(img, (x(cursor), 10), (x(cursor), 40), FG, 2)
    _text(img, "0", (2, 45), 0.35, DIM)
    _text(img, str(n - 1), (width - 30, 45), 0.35, DIM)
    return img


def legend(width: int, generated=()) -> np.ndarray:
    img = np.full((24, width, 3), BG, np.uint8)
    x = 4
    items = [("source frame", GREY), ("filled", ORANGE)] + ([("generated", PURPLE)] if generated else []) \
        + [("missing", RED), ("removed generated frame", BLUE)]
    for label, color in items:
        cv2.rectangle(img, (x, 7), (x + 12, 19), color, -1)
        _text(img, label, (x + 18, 18), 0.42, DIM)
        x += 30 + int(cv2.getTextSize(label, FONT, 0.42, 1)[0][0])
    return img


def _panel(frame: np.ndarray | None, size: tuple[int, int], title: str, status: str, status_color) -> np.ndarray:
    w, h = size
    if frame is None:
        img = np.full((h, w, 3), (40, 0, 0), np.uint8)
    else:
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) if frame.shape[:2] != (h, w) else frame.copy()
    band = np.full((26, w, 3), BG, np.uint8)
    _text(band, title, (6, 18), 0.5, FG)
    foot = np.full((24, w, 3), BG, np.uint8)
    _text(foot, status, (6, 17), 0.42, status_color)
    return np.vstack([band, img, foot])


def _row(plate, delivered, locked, o: Op, gen_count: int, size, generated=(), result_label="OUTPUT") -> np.ndarray:
    d_status = f"original AI frame {o.plate}" if o.plate < gen_count else "clip already ended"
    r_status, r_color = describe(o, generated)
    r_status = f"output frame {o.plate} | {r_status}"
    return np.hstack([
        _panel(plate, size, f"PLATE  frame {o.plate}", "reference", DIM),
        _panel(delivered, size, "AS DELIVERED", d_status, DIM),
        _panel(locked, size, result_label, r_status, r_color),
    ])


def pick_examples(ops: list[Op], k: int = 3, spacing: int = 15) -> list[int]:
    """Frames that show the change best: largest timing shift, first filled, first missing; never near-neighbours."""
    picks: list[int] = []
    far = lambda p: all(abs(p - q) >= spacing for q in picks)
    for kind in ("missing", "interp"):
        first = next((o.plate for o in ops if o.op == kind and far(o.plate)), None)
        if first is not None:
            picks.append(first)
    shifted = sorted((o for o in ops if o.op == "copy" and o.gen != o.plate), key=lambda o: -abs(o.gen - o.plate))
    worst = next((o.plate for o in shifted if far(o.plate)), None)
    if worst is not None:
        picks.append(worst)
    return sorted(picks[:k])


def summary_image(plate: np.ndarray, gen: np.ndarray, ops: list[Op], summary: dict, panel_width: int = 320,
                  fill=None, final: np.ndarray | None = None, generated=()) -> np.ndarray:
    """One picture: headline, timeline, and plate / as delivered / locked at the frames that changed most.

    final: the finished clip; when given, the locked panel shows its frames (e.g. after a video model filled gaps).
    """
    ph = int(round(plate.shape[1] * panel_width / plate.shape[2]))
    size = (panel_width, ph)
    width = 3 * panel_width
    head = np.full((62, width, 3), BG, np.uint8)
    lines = headline(summary, generated)
    _text(head, lines[0][0], (8, 24), 0.55, FG)
    x = 8
    for label, color in lines[1]:
        _text(head, label, (x, 50), 0.6, color)
        x += 24 + int(cv2.getTextSize(label, FONT, 0.6, 1)[0][0])
    parts = [head, timeline(ops, summary, width, generated=generated), legend(width, generated)]
    picks = pick_examples(ops)
    if picks:
        small_gen = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in gen])
        for p in picks:
            o = ops[p]
            r = final[p] if final is not None else locked_frames(small_gen, [o], fill, ops)[0]
            delivered = small_gen[p] if p < len(small_gen) else None
            parts.append(np.full((8, width, 3), BG, np.uint8))
            parts.append(_row(plate[p], delivered, r, o, len(gen), size, generated))
    return np.vstack(parts)


def compare_video(plate: np.ndarray, gen: np.ndarray, ops: list[Op], summary: dict, out_path: str | Path,
                  fps: Fraction, panel_width: int = 480, start: int = 0, end: int | None = None, fill=None,
                  final: np.ndarray | None = None, generated=(), result_label="OUTPUT") -> int:
    """Plate | as delivered (frame for frame) | locked, with status and a timeline cursor."""
    filler = Filler(ops, fill)
    ph = int(round(plate.shape[1] * panel_width / plate.shape[2]))
    ph -= ph % 2
    size = (panel_width, ph)
    small_gen = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in gen])
    small_plate = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in plate])
    width = 3 * panel_width
    end = len(ops) if end is None else min(end, len(ops))
    first = None
    for p in range(start, end):
        o = ops[p]
        if final is not None:
            r = cv2.resize(final[p], size, interpolation=cv2.INTER_AREA)
        elif o.op == "copy":
            r = small_gen[o.gen]
        elif o.op == "interp":
            r = filler.frame(o, small_gen[o.gen], small_gen[o.gen_b])
        else:
            r = None
        delivered = small_gen[p] if p < len(small_gen) else None
        frame = np.vstack([_row(small_plate[p], delivered, r, o, len(gen), size, generated, result_label),
                           timeline(ops, summary, width, cursor=p, generated=generated), legend(width, generated)])
        if first is None:
            h = frame.shape[0] + frame.shape[0] % 2
            writer = FrameWriter(out_path, width, h, fps, codec="h264")
            first = h
        if frame.shape[0] != first:
            frame = np.vstack([frame, np.full((first - frame.shape[0], width, 3), BG, np.uint8)])
        writer.write(frame)
    writer.close()
    return writer.count
