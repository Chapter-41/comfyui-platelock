"""realign PLATE GEN --out DIR [--render]"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import cv2

from .compare import compare_video, summary_image
from .core import analyze_arrays
from .plan import plan_to_json
from .render import render, strip_chart
from .video import decode_small_rgb, load_gray_cached, probe


def analyze(plate_path: str, gen_path: str, cache_dir: str | None = None, max_interp: int = 4,
            motion_weight: float = 0.5, stiffness: float = 2.0):
    plate_info, gen_info = probe(plate_path), probe(gen_path)
    plate = load_gray_cached(plate_path, cache_dir)
    gen = load_gray_cached(gen_path, cache_dir)
    r = analyze_arrays(gen, plate, motion_weight=motion_weight, max_interp=max_interp, stiffness=stiffness)
    report = {"plate": plate_info.to_dict(), "gen": gen_info.to_dict(), "summary": r.summary}
    return report, r.ops, r.duplicates, plate_info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="realign", description="Retime a generated video onto its source plate frame by frame.")
    ap.add_argument("plate")
    ap.add_argument("gen")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--render", action="store_true", help="write the repaired clip at the plate frame rate")
    ap.add_argument("--codec", choices=["prores", "h264"], default="prores")
    ap.add_argument("--preview", action="store_true", help="write a summary image and a side-by-side comparison video")
    ap.add_argument("--fps", help="output timebase (e.g. 25 when the plate was a 24 fps reinterpretation); frames are unchanged")
    ap.add_argument("--max-interp", type=int, default=4, help="longest gap to interpolate; longer gaps are reported missing")
    ap.add_argument("--motion-weight", type=float, default=0.5)
    ap.add_argument("--stiffness", type=float, default=2.0, help="higher = fewer retiming events; lower = more sensitive")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "realign"))
    a = ap.parse_args(argv)

    t0 = time.time()
    report, ops, dup, plate_info = analyze(a.plate, a.gen, a.cache, a.max_interp, a.motion_weight, a.stiffness)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.gen).stem
    strip_chart(ops, dup, a.out / f"{stem}_realign.png")
    if a.render:
        ext = "mov" if a.codec == "prores" else "mp4"
        out_clip = a.out / f"{stem}_realigned.{ext}"
        fps = Fraction(a.fps) if a.fps else plate_info.fps
        written = render(a.gen, ops, out_clip, fps, a.codec)
        report["output"] = {"path": str(out_clip), "frames": written}
    if a.preview:
        plate_rgb, gen_rgb = decode_small_rgb(a.plate, 480), decode_small_rgb(a.gen, 480)
        img = summary_image(plate_rgb, gen_rgb, ops, report["summary"])
        cv2.imwrite(str(a.out / f"{stem}_summary.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        fps = Fraction(a.fps) if a.fps else plate_info.fps
        compare_video(plate_rgb, gen_rgb, ops, report["summary"], a.out / f"{stem}_compare.mp4", fps)
    report["seconds"] = round(time.time() - t0, 2)
    (a.out / f"{stem}_realign.json").write_text(json.dumps({**report, "plan": plan_to_json(ops)}, indent=1))
    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
