"""Files for the PlateLock Viewer: exact per-frame JPEGs plus a manifest.

Frames are stills rather than video so the browser shows exactly frame N,
never the nearest keyframe. Layout of the viewer folder:

    p/000000.jpg   plate
    g/000000.jpg   generation as delivered (its own frame numbers)
    l/000000.jpg   result on the plate's timing
    manifest.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

WIDTH = 960


def extract(video: str | Path, folder: Path, width: int = WIDTH) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf", f"scale={width}:-2",
                    "-fps_mode", "passthrough", "-q:v", "3", "-start_number", "0", str(folder / "%06d.jpg")], check=True)


def export(folder: Path, plate: str | Path, generated: str | Path, result: str | Path, manifest: dict,
           sources: bool = True) -> None:
    """Write the viewer folder. sources=False keeps existing plate/generated stills and only refreshes the result."""
    folder.mkdir(parents=True, exist_ok=True)
    if sources or not (folder / "p").is_dir():
        extract(plate, folder / "p")
        extract(generated, folder / "g")
    if result is not None:
        extract(result, folder / "l")
    manifest = {**manifest, "version": int(time.time() * 1000),
                "counts": {k: len(list((folder / k).glob("*.jpg"))) for k in ("p", "g", "l")}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
