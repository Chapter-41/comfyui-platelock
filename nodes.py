"""ComfyUI nodes. VIDEO nodes stream from disk; the IMAGE node is for short clips and interop."""
from __future__ import annotations

import json
import time
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .realign.core import analyze_arrays
from .realign.plan import Op, plan_to_json
from .realign.render import render, strip_chart
from .realign.interp import interpolate
from .realign.video import load_gray_cached, probe

CATEGORY = "Realign"


def _video_path(video) -> str:
    try:
        from comfy_api.input_impl import VideoFromFile
    except ImportError:  # pragma: no cover - outside ComfyUI
        VideoFromFile = ()
    if VideoFromFile and not isinstance(video, VideoFromFile):
        raise ValueError("Realign needs a file-backed VIDEO (Load Video), not a decoded frame batch.")
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


def _to_image(bgr_or_gray: np.ndarray):
    import torch
    img = bgr_or_gray
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def _chart_image(ops, dup) -> np.ndarray:
    path = _dirs()[1] / f"chart_{uuid.uuid4().hex}.png"
    strip_chart(ops, dup, path)
    img = cv2.imread(str(path))
    path.unlink(missing_ok=True)
    return img


def _mask_image(w: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return (cv2.resize(np.clip(w, 0, 1), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)


def _report_text(plan: dict) -> str:
    s = plan["summary"]
    lines = [
        f"plate {s['plate_frames']} frames, generated {s['gen_frames']} frames",
        f"offset median {s['offset_median']:+.0f}, range {s['offset_range']}" if s["offset_range"] else "no matched frames",
        f"copied {s['copied']}, interpolated {len(s['interpolated'])}, missing {len(s['missing_plate_frames'])}",
        f"duplicates in generated: {s['duplicates_in_gen'] or 'none'}",
        f"dropped generated frames: {s['dropped_gen_frames'] or 'none'}",
        "segments (plate range, offset, confidence z):",
        *[f"  {g['plate'][0]}-{g['plate'][1]}  {g['offset']:+d}  {g['confidence']}" for g in s["segments"]],
        "COMPLETE" if s["complete"] else "INCOMPLETE: plate frames without generated coverage are rendered black",
    ]
    return "\n".join(lines)


def _mask_array(mask) -> np.ndarray | None:
    if mask is None:
        return None
    m = mask.cpu().numpy()
    return m[0] if m.ndim == 3 else m


_ANALYSIS_OPTIONS = {
    "motion_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05,
                                "tooltip": "Weight of frame-to-frame motion agreement vs. per-frame structure."}),
    "stiffness": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 8.0, "step": 0.25,
                            "tooltip": "Higher: fewer retiming events. Lower: more sensitive to small drifts."}),
    "max_interp": ("INT", {"default": 4, "min": 0, "max": 24,
                           "tooltip": "Longest gap to fill by interpolation; longer gaps are reported missing."}),
    "auto_mask": ("BOOLEAN", {"default": True,
                              "tooltip": "Weight regions where the generation follows the plate (e.g. actors over a replaced background)."}),
}


class RealignAnalyze:
    CATEGORY = CATEGORY
    FUNCTION = "analyze"
    RETURN_TYPES = ("REALIGN_PLAN", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("plan", "report", "chart", "agreement_mask")
    DESCRIPTION = "Match every plate frame to the generated frame that shows it: offsets, drift, duplicates, skips, short or long clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"plate": ("VIDEO",), "generated": ("VIDEO",), **_ANALYSIS_OPTIONS},
            "optional": {"mask": ("MASK", {"tooltip": "Plate-space weight (e.g. actor matte); replaces auto_mask."})},
        }

    def analyze(self, plate, generated, motion_weight, stiffness, max_interp, auto_mask, mask=None):
        t0 = time.time()
        plate_path, gen_path = _video_path(plate), _video_path(generated)
        cache = _dirs()[1]
        p = load_gray_cached(plate_path, cache)
        g = load_gray_cached(gen_path, cache)
        r = analyze_arrays(g, p, motion_weight=motion_weight, max_interp=max_interp,
                           mask=_mask_array(mask), auto_mask=auto_mask, stiffness=stiffness)
        plan = {
            "version": 1,
            "plate": probe(plate_path).to_dict(),
            "gen": probe(gen_path).to_dict(),
            "summary": r.summary,
            "ops": plan_to_json(r.ops),
            "seconds": round(time.time() - t0, 2),
        }
        return (plan, _report_text(plan), _to_image(_chart_image(r.ops, r.duplicates)),
                _to_image(_mask_image(r.weights, p.shape[1:])))


def _ops(plan: dict) -> list[Op]:
    return [Op(**o) for o in plan["ops"]]


class RealignRender:
    CATEGORY = CATEGORY
    FUNCTION = "render"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    OUTPUT_NODE = True
    DESCRIPTION = "Write one output frame per plate frame: copy, interpolate (optical flow) or black for uncovered frames."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("VIDEO",),
                "plan": ("REALIGN_PLAN",),
                "fps": ("STRING", {"default": "plate",
                                   "tooltip": "'plate' or a rate such as 25 or 24000/1001. Frames are unchanged; only the timebase."}),
                "codec": (["prores", "h264"], {"default": "prores"}),
                "filename_prefix": ("STRING", {"default": "realign"}),
            }
        }

    def render(self, generated, plan, fps, codec, filename_prefix):
        from comfy_api.input_impl import VideoFromFile
        gen_path = _video_path(generated)
        if probe(gen_path).frames != plan["gen"]["frames"]:
            raise ValueError("Generated video does not match the analyzed plan.")
        rate = Fraction(plan["plate"]["fps"]) if fps.strip() == "plate" else Fraction(fps.strip())
        out = _dirs()[0] / f"{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{'mov' if codec == 'prores' else 'mp4'}"
        written = render(gen_path, _ops(plan), out, rate, codec)
        out.with_suffix(".json").write_text(json.dumps({**plan, "output": {"path": str(out), "frames": written,
                                                                            "fps": f"{rate.numerator}/{rate.denominator}"}}, indent=1))
        return (VideoFromFile(str(out)), str(out))


class RealignTimeRemap:
    CATEGORY = CATEGORY
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("time_remap_json",)
    DESCRIPTION = "Per plate frame, the generated-clip time to show (seconds); null where uncovered. For After Effects or Nuke time remapping."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("REALIGN_PLAN",)}}

    def export(self, plan):
        gen_fps = Fraction(plan["gen"]["fps"])
        keys = []
        for o in _ops(plan):
            if o.op == "copy":
                t = o.gen / gen_fps
            elif o.op == "interp":
                t = (o.gen + o.alpha * (o.gen_b - o.gen)) / gen_fps
            else:
                t = None
            keys.append({"plate_frame": o.plate, "gen_time": None if t is None else round(float(t), 6), "op": o.op})
        return (json.dumps({"plate_fps": plan["plate"]["fps"], "gen_fps": plan["gen"]["fps"], "keys": keys}),)


def _gray_batch(images) -> np.ndarray:
    a = images.cpu().numpy()
    g = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).astype(np.float32)
    h = max(2, round(g.shape[1] * 256 / g.shape[2] / 2) * 2)
    return np.stack([cv2.resize(f, (256, h), interpolation=cv2.INTER_AREA) for f in g])


class RealignImages:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "report", "chart")
    DESCRIPTION = "IMAGE-batch version for short clips or chaining with other frame nodes. Prefer the VIDEO nodes for long or 4K clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"plate": ("IMAGE",), "generated": ("IMAGE",), **_ANALYSIS_OPTIONS},
            "optional": {"mask": ("MASK",)},
        }

    def run(self, plate, generated, motion_weight, stiffness, max_interp, auto_mask, mask=None):
        import torch
        r = analyze_arrays(_gray_batch(generated), _gray_batch(plate), motion_weight=motion_weight,
                           max_interp=max_interp, mask=_mask_array(mask), auto_mask=auto_mask, stiffness=stiffness)
        gen = generated.cpu().numpy()
        out = []
        for o in r.ops:
            if o.op == "copy":
                out.append(gen[o.gen])
            elif o.op == "interp":
                a = (gen[o.gen] * 255).astype(np.uint8)
                b = (gen[o.gen_b] * 255).astype(np.uint8)
                out.append(interpolate(a, b, o.alpha).astype(np.float32) / 255.0)
            else:
                out.append(np.zeros_like(gen[0]))
        plan = {"summary": r.summary}
        return (torch.from_numpy(np.stack(out)), _report_text(plan), _to_image(_chart_image(r.ops, r.duplicates)))


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (RealignAnalyze, RealignRender, RealignTimeRemap, RealignImages)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RealignAnalyze": "Realign · Analyze (plate vs generated)",
    "RealignRender": "Realign · Render to plate timing",
    "RealignTimeRemap": "Realign · Time-remap JSON",
    "RealignImages": "Realign · Images (short clips)",
}
