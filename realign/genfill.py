"""Generative gap filling with any video model that can inpaint in time (Wan VACE, LTX first/last frame).

The realigned clip is cut into fixed-length windows around the frames to
generate. Each window becomes a control video (known frames kept, frames to
generate set to neutral gray) and a mask (1 = generate). The model's output is
merged back frame-exactly: only masked frames are taken, colour-matched to the
model's own rendition of the known frames, so every real frame stays untouched.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .video import FrameReader, FrameWriter, probe


def windows(targets: list[int], frames: int, length: int, context: int = 8) -> list[tuple[int, list[int]]]:
    """Windows of `length` frames covering every target.

    Each starts up to `context` known frames before its first target; the
    context shrinks (down to one frame) when that lets one window take more
    targets, e.g. a whole missing tail.
    """
    out = []
    todo = sorted(set(targets))
    i = 0
    while i < len(todo):
        best = None
        for ctx in range(context, 0, -1):
            start = max(0, min(todo[i] - ctx, frames - length))
            covered = [t for t in todo[i:] if start <= t < start + length]
            if best is None or len(covered) > len(best[1]):
                best = (start, covered)
        start, covered = best
        out.append((start, covered))
        i += len(covered)
    return out


def read_range(path: str | Path, start: int, count: int, max_width: int = 0) -> np.ndarray:
    """Frames [start, start+count) as uint8 RGB, optionally downscaled to max_width."""
    info = probe(path)
    w, h = info.width, info.height
    if max_width and w > max_width:
        h = max(2, round(h * max_width / w / 2) * 2)
        w = max_width
    vf = f"select='between(n\\,{start}\\,{start + count - 1})',scale={w}:{h}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def control(frames: np.ndarray, local_targets: list[int], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Control video (float 0-1, gray where generating) and mask (1 = generate), both `length` frames."""
    n, h, w, _ = frames.shape
    video = np.full((length, h, w, 3), 0.5, np.float32)
    video[:n] = frames.astype(np.float32) / 255.0
    mask = np.zeros((length, h, w), np.float32)
    mask[n:] = 1.0
    for t in local_targets:
        video[t] = 0.5
        mask[t] = 1.0
    return video, mask


def colour_match(generated: np.ndarray, actual: np.ndarray, known: list[int]) -> np.ndarray:
    """Per-channel linear map from the model's rendition of known frames to the real ones, applied to all frames."""
    g = generated.astype(np.float32)
    if not known:
        return generated
    a = actual[known].astype(np.float32)
    gk = g[known]
    if gk.shape[1:3] != a.shape[1:3]:
        gk = np.stack([cv2.resize(f, (a.shape[2], a.shape[1]), interpolation=cv2.INTER_AREA) for f in gk])
    out = np.empty_like(g)
    for c in range(3):
        x, y = gk[..., c].ravel(), a[..., c].ravel()
        slope, intercept = np.polyfit(x, y, 1) if x.std() > 1e-3 else (1.0, float(y.mean() - x.mean()))
        out[..., c] = g[..., c] * slope + intercept
    return np.clip(out, 0, 255).astype(np.uint8)


def save_patches(work: str | Path, start: int, generated: np.ndarray, actual: np.ndarray,
                 local_targets: list[int], size: tuple[int, int]) -> list[int]:
    """Store colour-matched target frames at clip resolution. Returns the plate frames written."""
    n = len(actual)
    known = [i for i in range(n) if i not in set(local_targets) and actual[i].any()]
    matched = colour_match(generated[:max(n, max(local_targets) + 1)], actual, [k for k in known if k < len(generated)])
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    written = []
    for t in local_targets:
        if t >= len(matched):
            continue
        frame = cv2.resize(matched[t], size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(work / f"patch_{start + t:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        written.append(start + t)
    return written


def patched_frames(work: str | Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in Path(work).glob("patch_*.png"))


def render_patched(clip: str | Path, work: str | Path, out_path: str | Path, fps: Fraction, codec: str) -> int:
    """Re-encode the realigned clip with stored patches in place of their frames."""
    info = probe(clip)
    patches = set(patched_frames(work))
    reader = FrameReader(clip)
    writer = FrameWriter(out_path, info.width, info.height, fps, codec)
    try:
        while (frame := reader.read()) is not None:
            i = reader.index
            if i in patches:
                frame = cv2.cvtColor(cv2.imread(str(Path(work) / f"patch_{i:06d}.png")), cv2.COLOR_BGR2RGB)
            writer.write(frame)
    finally:
        reader.close()
        writer.close()
    return writer.count
