"""Ground-truth test set for detection: does PlateLock pick the right generated frame for every plate frame?

    python -m platelock.benchmark make  OUT [plate1.mp4 ...]   (synthetic scenes, plus real plates if given)
    python -m platelock.benchmark score OUT

`make` renders procedural scenes with exactly known motion (pans, a very slow pan, a static shot that
starts moving, moving objects, a talking head, a spinner that repeats every 12 frames, flicker) and
optionally cuts segments from real plates. From each it builds fake "generated" clips
from it with known damage (late or early start, short or long tail, skipped and
repeated frames, fps slips, speed drift), restyled so they no longer match
pixel for pixel. Every case writes <case>_plate.mp4, <case>_gen.mp4 and the
truth: for each generated frame, the source frame it shows.

`score` runs the alignment only (no frame filling) and checks each plate frame:

- shown by a generated frame -> must be a copy of one of those frames
- skipped inside the clip      -> must be filled from the generated frames on either side
- before / after the clip      -> must be reported missing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .core import analyze_arrays
from .video import FrameWriter, decode_gray, decode_small_rgb, probe

LENGTH = 240
LEAD = 24


def cases(n: int, rng: np.random.Generator) -> dict[str, list[int]]:
    """Source frame per generated frame, relative to the plate segment start (may be <0 or >=n: extra frames)."""
    base = list(range(n))
    out = {
        "identity": base,
        "starts_6_late": list(range(6, n)),
        "starts_10_early": list(range(-10, n)),
        "ends_13_short": list(range(n - 13)),
        "runs_20_long": list(range(n + 20)),
        "skips_every_17": [i for i in base if i % 17 != 16],
        "repeats_every_23": [x for i in base for x in ([i, i] if i % 23 == 22 else [i])],
        "hold_then_skip_every_29": [i - 1 if i % 29 == 28 else i for i in base],
        "fps_25_to_24": [round(j * 25 / 24) for j in range(n) if round(j * 25 / 24) < n],
        "fps_24_to_25": [int(j * 24 / 25) for j in range(round(n * 25 / 24))],
        "slow_drift_3pct": [int(j * 0.97) for j in range(round(n / 0.97))],
    }
    mix, i = [], -4
    while i < n - 9:
        r = rng.random()
        if r < 0.05:
            i += 2
        elif r < 0.10:
            mix.append(i)
        mix.append(i)
        i += 1
    out["random_mix"] = mix
    return out


def restyle(frames: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """Mimic a generator re-rendering the shot. graded: colour only; heavy: + softness and grain;
    reframed: + a 3% zoom and a few pixels of shift, which in a pan looks like a neighbouring frame."""
    if level == "clean":
        return frames
    f = frames.astype(np.float32) / 255.0
    f = np.clip((f ** 0.85 - 0.5) * 1.1 + 0.5, 0, 1)
    gray = f.mean(axis=3, keepdims=True)
    f = np.clip(gray + (f - gray) * 1.3 + np.float32([0.04, 0.0, -0.03]), 0, 1)
    out = (f * 255).astype(np.uint8)
    if level in ("heavy", "reframed"):
        h, w = out.shape[1:3]
        zoom = np.float32([[1.03, 0, -0.015 * w + 4], [0, 1.03, -0.015 * h - 3]])
        out = np.stack([np.clip(cv2.GaussianBlur(cv2.warpAffine(x, zoom, (w, h), borderMode=cv2.BORDER_REFLECT)
                                                 if level == "reframed" else x, (0, 0), 1.0).astype(np.float32)
                                + rng.normal(0, 6, x.shape).astype(np.float32), 0, 255).astype(np.uint8)
                        for x in out])
    return out


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    layers = [cv2.resize(rng.random((max(2, h // s), max(2, w // s), 3)).astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC) / (i + 1) for i, s in enumerate((64, 16, 4))]
    t = sum(layers)
    return ((t - t.min()) / (t.max() - t.min()) * 200 + 25).astype(np.uint8)


def synthetic(name: str, n: int, size=(480, 270), seed: int = 0) -> np.ndarray:
    """A procedural shot with exactly known motion, n frames of uint8 RGB."""
    rng = np.random.default_rng(seed)
    w, h = size
    world = _texture(rng, h * 2, w * 4)
    frames = np.empty((n, h, w, 3), np.uint8)
    blobs = [(rng.uniform(0.1, 0.9, 2), rng.uniform(-1, 1, 2), rng.integers(15, 45), tuple(int(c) for c in rng.integers(40, 255, 3)))
             for _ in range(6)]
    for t in range(n):
        if name == "pan":
            x = 3.0 * t
        elif name == "slow_pan":
            x = 0.35 * t
        elif name == "static_then_pan":
            x = max(0, t - n // 2) * 2.5
        else:
            x = 0
        bob = 6 * np.sin(t / 30) if name in ("pan", "slow_pan") else 0
        m = np.float32([[1, 0, -x - 20], [0, 1, -20 - bob]])
        f = cv2.warpAffine(world, m, (w, h), flags=cv2.INTER_LINEAR)
        if name in ("objects", "talking_head", "spinner", "flicker"):
            if name == "objects":
                for (p, v, r, c) in blobs:
                    q = (p + v * t * 0.004) % 1
                    cv2.circle(f, (int(q[0] * w), int(q[1] * h)), int(r), c, -1, cv2.LINE_AA)
            elif name == "talking_head":
                cx, cy = w // 2 + int(4 * np.sin(t / 11)), h // 2 + int(3 * np.sin(t / 7))
                cv2.ellipse(f, (cx, cy), (60, 80), 3 * np.sin(t / 13), 0, 360, (190, 150, 130), -1, cv2.LINE_AA)
                mouth = 4 + int(6 * abs(np.sin(t / 3)))
                cv2.ellipse(f, (cx, cy + 35), (18, mouth), 0, 0, 360, (90, 40, 40), -1, cv2.LINE_AA)
                for ex in (-22, 22):
                    cv2.circle(f, (cx + ex, cy - 15 + (2 if t % 47 < 3 else 0)), 6, (30, 30, 30), -1, cv2.LINE_AA)
            elif name == "spinner":
                a = t * 2 * np.pi / 12
                for k in range(3):
                    b = a + k * 2 * np.pi / 3
                    cv2.line(f, (w // 2, h // 2), (int(w // 2 + 90 * np.cos(b)), int(h // 2 + 90 * np.sin(b))),
                             (240, 240, 240), 9, cv2.LINE_AA)
                cv2.circle(f, (int(w * 0.2 + t * 0.8) % w, int(h * 0.8)), 12, (255, 60, 60), -1, cv2.LINE_AA)
            elif name == "flicker":
                cv2.circle(f, (int(w * 0.1 + t * 1.5) % w, h // 2), 25, (60, 200, 255), -1, cv2.LINE_AA)
                f = np.clip(f.astype(np.float32) * (1 + 0.25 * np.sin(t * 1.7) * (rng.random() > 0.5)), 0, 255).astype(np.uint8)
        frames[t] = f
    return frames


SCENES = ["pan", "slow_pan", "static_then_pan", "objects", "talking_head", "spinner", "flicker"]


def _write(path: Path, frames: np.ndarray, fps: Fraction) -> None:
    h, w = frames.shape[1:3]
    writer = FrameWriter(path, w, h, fps, "h264")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.close()


def _sources(plates: list[str], width: int):
    for name in SCENES:
        yield name, synthetic(name, LENGTH + 2 * LEAD, seed=SCENES.index(name)), Fraction(24)
    for plate_path in plates:
        stem = Path(plate_path).stem.split("_")
        yield "_".join(stem[:2]), decode_small_rgb(plate_path, width), probe(plate_path).fps


def make(out: Path, plates: list[str] = (), levels=("graded", "heavy", "reframed"), width: int = 480) -> None:
    """Synthetic scenes always; real plates too when given."""
    rng = np.random.default_rng(41)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for shot, src, fps in _sources(list(plates), width):
        a = min(max(LEAD, (len(src) - LENGTH) // 2), max(0, len(src) - LENGTH - LEAD))
        n = min(LENGTH, len(src) - a - LEAD)
        _write(out / f"{shot}_plate.mp4", src[a:a + n], fps)
        for name, mapping in cases(n, rng).items():
            mapping = [m for m in mapping if -a <= m < len(src) - a]
            for level in levels:
                case = f"{shot}__{name}__{level}"
                _write(out / f"{case}_gen.mp4", restyle(src[[a + m for m in mapping]], level, rng), fps)
                truth[case] = {"plate": f"{shot}_plate.mp4", "gen": f"{case}_gen.mp4", "shows": mapping,
                               "case": name, "level": level, "shot": shot}
        print(f"{shot}: {n} plate frames, {len(truth)} cases so far", flush=True)
    (out / "truth.json").write_text(json.dumps(truth))


def judge(ops: list, shows: list[int], n: int, plate: np.ndarray | None = None) -> dict:
    """Per plate frame: ok, or what went wrong. With the plate given, picking a frame that is pixel-identical
    to the right one (a static shot) counts as correct."""
    def same(a: int, b: int) -> bool:
        return plate is not None and 0 <= a < n and 0 <= b < n and float(np.abs(plate[a] - plate[b]).mean()) < 1e-4

    by_plate: dict[int, set[int]] = {}
    for j, p in enumerate(shows):
        by_plate.setdefault(p, set()).add(j)
    lo = min((p for p in shows if 0 <= p < n), default=n)
    hi = max((p for p in shows if 0 <= p < n), default=-1)
    errors: dict[str, list[int]] = {}
    for p in range(n):
        o = ops[p]
        if p in by_plate:
            if o.op == "copy" and (o.gen in by_plate[p] or same(shows[o.gen], p)):
                continue
            if o.op == "copy":
                kind = "wrong frame"
            else:
                kind = "filled although a matching frame exists" if o.op == "interp" else "reported missing although it exists"
        elif lo < p < hi:
            if o.op == "interp" and shows[o.gen] < p < shows[o.gen_b]:
                continue
            kind = "skip not detected" if o.op == "copy" else "skip filled from the wrong frames" if o.op == "interp" else "skip reported missing"
        else:
            if o.op == "missing":
                continue
            kind = "missing frame not reported"
        errors.setdefault(kind, []).append(p)
    wrong = sum(len(v) for v in errors.values())
    offsets = [abs(shows[ops[p].gen] - p) for p in errors.get("wrong frame", [])]
    return {"frames": n, "correct": n - wrong, "within_1": n - wrong + sum(1 for o in offsets if o <= 1),
            "errors": errors, "max_off_by": max(offsets, default=0)}


def score(out: Path, only: str = "") -> dict:
    truth = json.loads((out / "truth.json").read_text())
    plates: dict[str, np.ndarray] = {}
    results = {}
    for case, t in truth.items():
        if only and only not in case:
            continue
        plate = plates.setdefault(t["plate"], decode_gray(out / t["plate"]))
        started = time.time()
        r = analyze_arrays(decode_gray(out / t["gen"]), plate)
        res = judge(r.ops, t["shows"], len(plate), plate)
        res["seconds"] = round(time.time() - started, 1)
        results[case] = {**res, "case": t["case"], "level": t["level"], "shot": t["shot"]}
        bad = ", ".join(f"{k}: {v[:6]}{'…' if len(v) > 6 else ''}" for k, v in res["errors"].items())
        print(f"{case:48s} {res['correct']:4d}/{res['frames']}  {bad}", flush=True)
    (out / "scores.json").write_text(json.dumps(results, indent=1))
    _table(results)
    return results


def _table(results: dict) -> None:
    def pct(rs, key="correct"):
        return f"{100 * sum(r[key] for r in rs) / max(1, sum(r['frames'] for r in rs)):5.1f}%"
    levels = sorted({r["level"] for r in results.values()}, key=["graded", "heavy", "reframed"].index)
    for group in ("case", "shot"):
        print(f"\n{group:24s}" + "".join(f"{lv:>20s}" for lv in levels) + "     (exact / within 1 frame)")
        for key in sorted({r[group] for r in results.values()}):
            cells = [[r for r in results.values() if r[group] == key and r["level"] == lv] for lv in levels]
            print(f"{key:24s}" + "".join(f"{pct(c):>12s} /{pct(c, 'within_1'):>6s}" for c in cells))
    perfect = sum(r["correct"] == r["frames"] for r in results.values())
    print(f"\nALL {pct(results.values())} exact, {pct(results.values(), 'within_1')} within 1 frame, "
          f"{perfect}/{len(results)} clips perfect")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="platelock.benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out", type=Path)
    m.add_argument("plates", nargs="*")
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    s.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a.out, a.plates)
    else:
        score(a.out, a.only)


if __name__ == "__main__":
    main(sys.argv[1:])
