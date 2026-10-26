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


def model_fill(interp_model):
    """Fill gaps with a native ComfyUI frame interpolation model (RIFE or FILM from Load Frame Interpolation Model)."""
    import torch
    from comfy_extras.nodes_frame_interpolation import FrameInterpolate

    def fill(a: np.ndarray, b: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
        k = len(alphas)
        expected = [round((i + 1) / (k + 1), 6) for i in range(k)]
        if sorted(round(t, 6) for t in alphas) != expected:
            raise ValueError("Frame interpolation models need evenly spaced in-betweens.")
        pair = torch.from_numpy(np.stack([a, b]).astype(np.float32) / 255.0)
        out = FrameInterpolate.execute(interp_model, pair, k + 1)
        images = (getattr(out, "args", None) or getattr(out, "result", None) or out)[0]
        mids = (images[1:-1].float().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        by_alpha = dict(zip(expected, mids))
        return [by_alpha[round(t, 6)] for t in alphas]

    return fill


class Realign:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "VIDEO", "IMAGE", "STRING", "REALIGN_FILL")
    RETURN_NAMES = ("realigned", "side_by_side", "preview", "report", "fill")
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
            "optional": {
                "interp_model": ("INTERP_MODEL", {"tooltip": "RIFE or FILM from 'Load Frame Interpolation Model'. "
                                                             "Without it, gaps are filled with basic optical flow."}),
                "mask": ("MASK", {"tooltip": "Optional: area to match on (e.g. the actors). By default this is found automatically."}),
            },
        }

    def run(self, plate, generated, output_fps, format, interp_model=None, mask=None):
        from comfy_api.input_impl import VideoFromFile
        plate_path, gen_path = _video_path(plate), _video_path(generated)
        plate_info, gen_info = probe(plate_path), probe(gen_path)
        out_dir, cache = _dirs()
        r = analyze_arrays(load_gray_cached(gen_path, cache), load_gray_cached(plate_path, cache), mask=_mask_array(mask))
        fps = plate_info.fps if output_fps.strip() in ("", "plate") else Fraction(output_fps.strip())
        fill = model_fill(interp_model) if interp_model is not None else None

        stem = f"{Path(gen_path).stem}_realign_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        codec = "prores" if format == "prores" else "h264"
        clip = out_dir / f"{stem}.{'mov' if format == 'prores' else 'mp4'}"
        render(gen_path, r.ops, clip, fps, codec, fill=fill)
        plate_rgb, gen_rgb = decode_small_rgb(plate_path, 480), decode_small_rgb(gen_path, 480)
        side = out_dir / f"{stem}_side_by_side.mp4"
        compare_video(plate_rgb, gen_rgb, r.ops, r.summary, side, fps, fill=fill)
        sidecar = out_dir / f"{stem}.json"
        sidecar.write_text(json.dumps({
            "plate": plate_info.to_dict(), "generated": gen_info.to_dict(), "output": str(clip),
            "fill_method": "frame interpolation model" if fill else "optical flow",
            "summary": r.summary, "ops": plan_to_json(r.ops), "time_remap_seconds": _time_remap(r.ops, gen_info.fps),
        }, indent=1))
        preview = summary_image(plate_rgb, gen_rgb, r.ops, r.summary, fill=fill)
        fill_state = {
            "clip": str(clip), "sidecar": str(sidecar), "work": str(out_dir / stem), "codec": codec,
            "fps": f"{fps.numerator}/{fps.denominator}", "frames": len(r.ops),
            "missing": r.summary["missing_plate_frames"], "filled": r.summary["interpolated"],
        }
        return (VideoFromFile(str(clip)), VideoFromFile(str(side)), _image(preview), _report(r.summary), fill_state)


def _targets(fill: dict, frames_to_generate: str) -> list[int]:
    return sorted(set(fill["missing"]) | (set(fill["filled"]) if frames_to_generate == "missing and filled" else set()))


class RealignFillWindow:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "REALIGN_WINDOW", "STRING")
    RETURN_NAMES = ("control_video", "control_masks", "first_frame", "last_frame", "window", "status")
    DESCRIPTION = ("Cuts one window around frames to generate. Wan VACE: control_video + control_masks into WanVaceToVideo "
                   "(length = this window's length). LTX: first_frame / last_frame as guides. Queue once per window.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill": ("REALIGN_FILL",),
                "window": ("INT", {"default": 0, "min": 0, "max": 9999, "control_after_generate": "increment",
                                   "tooltip": "Which window to prepare; increments after each queue."}),
                "length": ("INT", {"default": 81, "min": 9, "max": 1025, "step": 8,
                                   "tooltip": "81 suits Wan (4n+1) and LTX (8n+1)."}),
                "frames_to_generate": (["missing", "missing and filled"], {"default": "missing",
                                       "tooltip": "'missing and filled' also regenerates frames that interpolation filled."}),
                "max_width": ("INT", {"default": 1280, "min": 256, "max": 8192, "step": 16}),
            }
        }

    def run(self, fill, window, length, frames_to_generate, max_width):
        import torch
        from .realign.genfill import control, read_range, windows
        targets = _targets(fill, frames_to_generate)
        wins = windows(targets, fill["frames"], length)
        if not wins:
            raise ValueError("Nothing to generate: every plate frame is covered.")
        if window >= len(wins):
            raise ValueError(f"Only {len(wins)} window(s); all done. Reset window to 0 to start over.")
        start, covered = wins[window]
        frames = read_range(fill["clip"], start, min(length, fill["frames"] - start), max_width)
        local = [t - start for t in covered]
        video, mask = control(frames, local, length)
        before = max([i for i in range(len(frames)) if i not in local and i < min(local)], default=None)
        after = min([i for i in range(len(frames)) if i not in local and i > max(local)], default=None)
        first = frames[before] if before is not None else frames[0]
        last = frames[after] if after is not None else first
        info = {**fill, "start": start, "targets": local, "length": length, "frames_to_generate": frames_to_generate}
        status = (f"window {window + 1}/{len(wins)}: plate frames {start}-{start + length - 1}, "
                  f"generating {len(covered)} frame(s) {covered[0]}-{covered[-1]}")
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return (to_t(video), to_t(mask), _image(first), _image(last), info, status)


class RealignFillMerge:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "status")
    OUTPUT_NODE = True
    DESCRIPTION = "Takes the model's frames for this window, keeps only the generated ones (colour-matched), and re-renders the clip."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"window": ("REALIGN_WINDOW",), "images": ("IMAGE", {"tooltip": "Decoded model output for this window."})}}

    def run(self, window, images):
        from comfy_api.input_impl import VideoFromFile
        from .realign.genfill import patched_frames, read_range, render_patched, save_patches
        gen = (images.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        if len(gen) < max(window["targets"]) + 1:
            raise ValueError(f"Model returned {len(gen)} frames; this window needs {window['length']} "
                             "(trim any reference frames first).")
        info = probe(window["clip"])
        count = min(window["length"], window["frames"] - window["start"])
        actual = read_range(window["clip"], window["start"], count, max_width=gen.shape[2])
        if actual.shape[1:3] != gen.shape[1:3]:
            import cv2
            gen = np.stack([cv2.resize(f, (actual.shape[2], actual.shape[1]), interpolation=cv2.INTER_AREA) for f in gen])
        save_patches(window["work"], window["start"], gen, actual, window["targets"], (info.width, info.height))
        out = Path(window["clip"]).with_name(Path(window["clip"]).stem + "_generated" + Path(window["clip"]).suffix)
        render_patched(window["clip"], window["work"], out, Fraction(window["fps"]), window["codec"])
        done = set(patched_frames(window["work"]))
        todo = [t for t in _targets(window, window["frames_to_generate"]) if t not in done]
        status = "all frames generated" if not todo else f"{len(todo)} frame(s) still to generate: {todo[:12]}{' ...' if len(todo) > 12 else ''}"
        return (VideoFromFile(str(out)), status)


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


NODE_CLASS_MAPPINGS = {"Realign": Realign, "RealignImages": RealignImages,
                       "RealignFillWindow": RealignFillWindow, "RealignFillMerge": RealignFillMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"Realign": "Realign", "RealignImages": "Realign (images)",
                              "RealignFillWindow": "Realign Fill · Window (for Wan VACE / LTX)",
                              "RealignFillMerge": "Realign Fill · Merge"}
