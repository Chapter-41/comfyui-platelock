"""ComfyUI nodes. Realign streams from video files; Realign (images) is for short IMAGE batches."""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .realign.compare import compare_video, realigned_frames, summary_image
from .realign.core import analyze_arrays
from .realign.plan import Op, plan_to_json
from .realign.render import render
from .realign.video import decode_small_rgb, load_gray_cached, probe

CATEGORY = "Realign"


def _video_path(video) -> str:
    try:
        from comfy_api.input_impl import VideoFromFile
    except ImportError:  # pragma: no cover - outside ComfyUI
        VideoFromFile = ()
    if VideoFromFile and not isinstance(video, VideoFromFile):
        raise ValueError("Realign needs a file-backed VIDEO (Load Video). For frame batches use Realign (images).")
    path = video.get_stream_source()
    start, duration = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0, 0)
    if start or duration > 0:
        raise ValueError("Trimmed VIDEO inputs are not supported; export the trimmed range first.")
    if not isinstance(path, (str, Path)) or not Path(path).is_file():
        raise ValueError("Video file is unavailable.")
    return str(path)


def _dirs() -> tuple[Path, Path]:
    try:
        import folder_paths
        out, tmp = Path(folder_paths.get_output_directory()), Path(folder_paths.get_temp_directory())
    except ImportError:  # pragma: no cover
        out = tmp = Path.home() / ".cache" / "realign"
    return out / "realign", tmp / "realign_cache"


def _image(rgb: np.ndarray):
    import torch
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]


def _report(summary: dict) -> str:
    s = summary
    lines = [f"plate {s['plate_frames']} frames, generated {s['gen_frames']} frames -> output {s['plate_frames']} frames"]
    if s["interpolated"]:
        lines.append(f"filled {len(s['interpolated'])} frames the generator skipped or repeated: {s['interpolated']}")
    if s["dropped_gen_frames"]:
        lines.append(f"removed {len(s['dropped_gen_frames'])} extra generated frames: {s['dropped_gen_frames']}")
    if s["missing_plate_frames"]:
        lines.append(f"MISSING {len(s['missing_plate_frames'])} plate frames never generated (black): {s['missing_plate_frames']}")
    if len(lines) == 1:
        lines.append("already in sync, nothing changed")
    return "\n".join(lines)


def _time_remap(ops: list[Op], gen_fps: Fraction) -> list[float | None]:
    """Per plate frame, the generated-clip time in seconds (null where missing), for AE/Nuke time remapping."""
    out = []
    for o in ops:
        if o.op == "copy":
            out.append(round(float(o.gen / gen_fps), 6))
        elif o.op == "interp":
            out.append(round(float((o.gen + o.alpha * (o.gen_b - o.gen)) / gen_fps), 6))
        else:
            out.append(None)
    return out


def _mask_array(mask) -> np.ndarray | None:
    if mask is None:
        return None
    m = mask.cpu().numpy()
    return m[0] if m.ndim == 3 else m


class Realign:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("realigned", "side_by_side", "preview", "report")
    OUTPUT_NODE = True
    DESCRIPTION = ("Puts a generated video back on the plate's timing: one output frame per plate frame. "
                   "Skipped or repeated frames are filled, extra frames removed, never-generated frames left black and reported.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("VIDEO", {"tooltip": "The source plate the generation was made from."}),
                "generated": ("VIDEO", {"tooltip": "The AI result, as delivered."}),
                "output_fps": ("STRING", {"default": "plate",
                                          "tooltip": "'plate', or e.g. 25. Only the timebase changes, never the frames."}),
                "format": (["mp4", "prores"], {"default": "mp4"}),
            },
            "optional": {"mask": ("MASK", {"tooltip": "Optional: area to match on (e.g. the actors). By default this is found automatically."})},
        }

    def run(self, plate, generated, output_fps, format, mask=None):
        from comfy_api.input_impl import VideoFromFile
        plate_path, gen_path = _video_path(plate), _video_path(generated)
        plate_info, gen_info = probe(plate_path), probe(gen_path)
        out_dir, cache = _dirs()
        r = analyze_arrays(load_gray_cached(gen_path, cache), load_gray_cached(plate_path, cache), mask=_mask_array(mask))
        fps = plate_info.fps if output_fps.strip() in ("", "plate") else Fraction(output_fps.strip())

        stem = f"{Path(gen_path).stem}_realign_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = out_dir / f"{stem}.{'mov' if format == 'prores' else 'mp4'}"
        render(gen_path, r.ops, clip, fps, "prores" if format == "prores" else "h264")
        plate_rgb, gen_rgb = decode_small_rgb(plate_path, 480), decode_small_rgb(gen_path, 480)
        side = out_dir / f"{stem}_side_by_side.mp4"
        compare_video(plate_rgb, gen_rgb, r.ops, r.summary, side, fps)
        (out_dir / f"{stem}.json").write_text(json.dumps({
            "plate": plate_info.to_dict(), "generated": gen_info.to_dict(), "output": str(clip),
            "summary": r.summary, "ops": plan_to_json(r.ops), "time_remap_seconds": _time_remap(r.ops, gen_info.fps),
        }, indent=1))
        preview = summary_image(plate_rgb, gen_rgb, r.ops, r.summary)
        return (VideoFromFile(str(clip)), VideoFromFile(str(side)), _image(preview), _report(r.summary))


def _gray(images) -> np.ndarray:
    import cv2
    a = images.cpu().numpy()
    g = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).astype(np.float32)
    h = max(2, round(g.shape[1] * 256 / g.shape[2] / 2) * 2)
    return np.stack([cv2.resize(f, (256, h), interpolation=cv2.INTER_AREA) for f in g])


class RealignImages:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("realigned", "preview", "report")
    DESCRIPTION = "Same as Realign, for short IMAGE batches. Use Realign for long or 4K clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plate": ("IMAGE",), "generated": ("IMAGE",)}, "optional": {"mask": ("MASK",)}}

    def run(self, plate, generated, mask=None):
        import torch
        r = analyze_arrays(_gray(generated), _gray(plate), mask=_mask_array(mask))
        gen8 = (generated.cpu().numpy() * 255).astype(np.uint8)
        plate8 = (plate.cpu().numpy() * 255).astype(np.uint8)
        frames = [f if f is not None else np.zeros_like(gen8[0]) for f in realigned_frames(gen8, r.ops)]
        preview = summary_image(plate8, gen8, r.ops, r.summary)
        return (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0), _image(preview), _report(r.summary))


NODE_CLASS_MAPPINGS = {"Realign": Realign, "RealignImages": RealignImages}
NODE_DISPLAY_NAME_MAPPINGS = {"Realign": "Realign", "RealignImages": "Realign (images)"}
