"""ffmpeg-based probing and decoding."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    frames: int
    codec: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "frames": self.frames,
            "codec": self.codec,
        }


def probe(path: str | Path) -> VideoInfo:
    """Probe with a real packet count; container frame counts are often wrong."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,codec_name",
            "-of", "json", str(path),
        ],
        text=True,
    )
    s = json.loads(out)["streams"][0]
    return VideoInfo(
        path=str(path),
        width=int(s["width"]),
        height=int(s["height"]),
        fps=Fraction(s["r_frame_rate"]),
        frames=int(s.get("nb_read_packets") or 0),
        codec=s.get("codec_name", ""),
    )


def decode_gray(path: str | Path, width: int = 256) -> np.ndarray:
    """Decode every frame (no fps conversion) as small float32 grayscale, shape (N, H, W)."""
    info = probe(path)
    height = max(2, round(info.height * width / info.width / 2) * 2)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-vf", f"scale={width}:{height}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width)
    return frames.astype(np.float32) / 255.0


def load_gray_cached(path: str | Path, cache_dir: str | Path | None, width: int = 256) -> np.ndarray:
    if cache_dir is None:
        return decode_gray(path, width)
    p = Path(path)
    st = p.stat()
    key = f"{p.stem}_{st.st_size}_{int(st.st_mtime)}_{width}.npy"
    cache = Path(cache_dir) / key
    if cache.exists():
        return np.load(cache)
    frames = decode_gray(path, width)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, frames)
    return frames


class FrameReader:
    """Sequential full-resolution RGB reader."""

    def __init__(self, path: str | Path, width: int | None = None, height: int | None = None):
        info = probe(path)
        self.width = width or info.width
        self.height = height or info.height
        vf = [] if (self.width, self.height) == (info.width, info.height) else [f"scale={self.width}:{self.height}"]
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._size = self.width * self.height * 3
        self.index = -1

    def read(self) -> np.ndarray | None:
        buf = self._proc.stdout.read(self._size)
        if len(buf) < self._size:
            return None
        self.index += 1
        return np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> None:
        self._proc.stdout.close()
        self._proc.wait()


class FrameWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction, codec: str = "prores"):
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        ]
        if codec == "prores":
            cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
        else:
            cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p"]
        cmd.append(str(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")
