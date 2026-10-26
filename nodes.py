"""ComfyUI nodes. PlateLock streams from video files; PlateLock (images) is for short IMAGE batches."""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .platelock.compare import compare_video, locked_frames, summary_image
from .platelock.core import analyze_arrays
from .platelock.drift import measure, report_lines, summary as drift_summary
from .platelock.plan import Op, plan_to_json
from .platelock.render import render
from .platelock.video import decode_small_rgb, load_gray_cached, probe

CATEGORY = "PlateLock"
FILL_MODES = ["interpolate (RIFE)", "generate (LTX)", "none (debug)"]
_continue = [0]


def _video_path(video) -> str:
    try:
        from comfy_api.input_impl import VideoFromFile
    except ImportError:  # pragma: no cover - outside ComfyUI
        VideoFromFile = ()
    if VideoFromFile and not isinstance(video, VideoFromFile):
        raise ValueError("PlateLock needs a file-backed VIDEO (Load Video). For frame batches use PlateLock (images).")
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
        out = tmp = Path.home() / ".cache" / "platelock"
    return out / "platelock", tmp / "platelock_cache"


def _image(rgb: np.ndarray):
    import torch
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]


def _report(summary: dict, generated=()) -> str:
    s = summary
    lines = [f"plate {s['plate_frames']} frames, generated {s['gen_frames']} frames -> output {s['plate_frames']} frames"]
    filled = [f for f in s["interpolated"] if f not in generated]
    if filled:
        lines.append(f"filled {len(filled)} frames the generator skipped or repeated: {filled}")
    if s["dropped_gen_frames"]:
        lines.append(f"removed {len(s['dropped_gen_frames'])} extra generated frames: {s['dropped_gen_frames']}")
    if generated:
        lines.append(f"generated {len(generated)} frames with the video model: {sorted(generated)}")
    missing = [f for f in s["missing_plate_frames"] if f not in generated]
    if missing:
        lines.append(f"MISSING {len(missing)} plate frames never generated (black): {missing}")
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


def hold_fill(a: np.ndarray, b: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
    """No new pixels: each in-between repeats the nearer generated frame."""
    return [a if t < 0.5 else b for t in alphas]


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


class PlateLock:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "VIDEO", "IMAGE", "STRING", "PLATELOCK_FILL")
    RETURN_NAMES = ("locked", "side_by_side", "preview", "report", "fill")
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
                "fill_frames": (FILL_MODES, {"default": FILL_MODES[0], "tooltip":
                    "interpolate: RIFE makes the frames the generator skipped or repeated. "
                    "generate: also has the video model (LTX) make frames that were never generated; needs the Fill nodes. "
                    "none: debug, no new frames: skipped frames hold the nearest generated frame, missing ones stay black."}),
            },
            "optional": {
                "interp_model": ("INTERP_MODEL", {"tooltip": "RIFE or FILM from 'Load Frame Interpolation Model'. "
                                                             "Without it, gaps are filled with basic optical flow."}),
                "mask": ("MASK", {"tooltip": "Optional: area to match on (e.g. the actors). By default this is found automatically."}),
            },
        }

    def run(self, plate, generated, output_fps, format, fill_frames=FILL_MODES[0], interp_model=None, mask=None):
        from comfy_api.input_impl import VideoFromFile
        plate_path, gen_path = _video_path(plate), _video_path(generated)
        plate_info, gen_info = probe(plate_path), probe(gen_path)
        out_dir, cache = _dirs()
        r = analyze_arrays(load_gray_cached(gen_path, cache), load_gray_cached(plate_path, cache), mask=_mask_array(mask))
        fps = plate_info.fps if output_fps.strip() in ("", "plate") else Fraction(output_fps.strip())
        debug = fill_frames.startswith("none")
        fill = hold_fill if debug else model_fill(interp_model) if interp_model is not None else None

        stem = f"{Path(gen_path).stem}_platelock_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        codec = "prores" if format == "prores" else "h264"
        clip = out_dir / f"{stem}.{'mov' if format == 'prores' else 'mp4'}"
        render(gen_path, r.ops, clip, fps, codec, fill=fill)
        plate_rgb, gen_rgb = decode_small_rgb(plate_path, 480), decode_small_rgb(gen_path, 480)
        side = out_dir / f"{stem}_side_by_side.mp4"
        compare_video(plate_rgb, gen_rgb, r.ops, r.summary, side, fps, fill=fill)
        drift, exact = _exactness(plate_rgb, clip, r.ops, set(), plate_info.width)
        sidecar = out_dir / f"{stem}.json"
        sidecar.write_text(json.dumps({
            "plate": plate_info.to_dict(), "generated": gen_info.to_dict(), "output": str(clip),
            "fill_method": "none (hold)" if debug else "frame interpolation model" if fill else "optical flow",
            "summary": r.summary, "exactness": exact, "ops": plan_to_json(r.ops),
            "time_remap_seconds": _time_remap(r.ops, gen_info.fps),
        }, indent=1))
        preview = summary_image(plate_rgb, gen_rgb, r.ops, r.summary, fill=fill)
        viewer = out_dir / f"{stem}_viewer"
        fill_state = {
            "clip": str(clip), "sidecar": str(sidecar), "work": str(out_dir / stem), "codec": codec,
            "fps": f"{fps.numerator}/{fps.denominator}", "frames": len(r.ops), "viewer": str(viewer),
            "generate": fill_frames.startswith("generate"),
            "missing": r.summary["missing_plate_frames"], "filled": r.summary["interpolated"],
        }
        _export_viewer(viewer, plate_path, gen_path, clip, r.ops, set(), drift, exact, fps, "LOCKED")
        lines = [_report(r.summary)] + report_lines(exact)
        if debug and r.summary["interpolated"]:
            lines.insert(1, "debug: no frames were made; 'filled' frames hold the nearest generated frame")
        final = not (fill_state["generate"] and fill_state["missing"])
        return _with_viewer(viewer, (VideoFromFile(str(clip)), VideoFromFile(str(side)), _image(preview),
                                     "\n".join(lines), fill_state), final)


def _exactness(plate_rgb: np.ndarray, result_clip, ops: list[Op], generated: set[int], width: int):
    result_rgb = decode_small_rgb(str(result_clip), 480)
    drift = measure(plate_rgb, result_rgb, [o.op != "missing" or o.plate in generated for o in ops], width)
    return drift, drift_summary(drift)


def _export_viewer(folder: Path, plate, generated, result, ops, done: set[int], drift, exact, fps: Fraction,
                   label: str, sources: bool = True) -> None:
    from .platelock.viewer import export
    export(folder, plate, generated, result, {
        "name": Path(str(generated)).stem, "fps": f"{fps.numerator}/{fps.denominator}", "label": label,
        "ops": plan_to_json(ops), "generated": sorted(done), "drift": drift, "exactness": exact,
    }, sources=sources)


def _with_viewer(folder: Path, result: tuple, final: bool = True) -> dict:
    return {"ui": {"platelock_viewer": [{"subfolder": f"platelock/{folder.name}", "final": final}]}, "result": result}


class PlateLockFillWindow:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "FLOAT", "PLATELOCK_WINDOW", "STRING")
    RETURN_NAMES = ("control_video", "control_masks", "context_frames", "first_frame", "last_frame",
                    "width", "height", "length", "fps", "window", "status")
    DESCRIPTION = ("Picks the next missing gap and cuts a window around it, sized automatically. LTX: context_frames as a "
                   "guide at frame 0, width/height/length into the empty latent. Wan VACE: control_video + control_masks. "
                   "Only runs when PlateLock's fill_frames is 'generate' and frames are missing; otherwise the model "
                   "nodes are skipped. Fill Merge queues the workflow again until every gap is generated.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"fill": ("PLATELOCK_FILL",)}}

    @classmethod
    def IS_CHANGED(cls, **_):
        return _continue[0]

    def run(self, fill):
        import torch
        from .platelock.genfill import control, model_size, next_window, overlay_patches, patched_frames, read_range
        if not (fill.get("generate") and fill["missing"]):
            return _blocked(len(self.RETURN_TYPES))
        work = Path(fill["work"])
        remaining = sorted(set(fill["missing"]) - set(patched_frames(work)))
        if not remaining:
            return self._nothing(fill)
        remaining = sorted(set(fill["missing"]) - set(patched_frames(work)))
        w = next_window(remaining, fill["frames"])
        clip_info = probe(fill["clip"])
        size = model_size(clip_info.width, clip_info.height, 1280)
        frames = overlay_patches(read_range(fill["clip"], w["start"], w["count"], size=size), w["start"], work)
        local = [t - w["start"] for t in w["targets"]]
        if w["reverse"]:
            frames = frames[::-1]
            local = sorted(w["count"] - 1 - t for t in local)
        video, mask = control(frames, local, w["length"])
        context = frames[:w["context"]]
        info = {**fill, **w}
        status = (f"generating {len(w['targets'])} of {len(remaining)} missing frame(s): "
                  f"{w['targets'][0]}-{w['targets'][-1]}{' (backwards from the frames after)' if w['reverse'] else ''}")
        to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        return (to_t(video), to_t(mask), to_t(context.astype(np.float32) / 255.0), _image(context[-1]),
                _image(frames[min(local[-1] + 1, len(frames) - 1)]), size[0], size[1], w["length"],
                float(Fraction(fill["fps"])), info, status)

    @staticmethod
    def _nothing(fill):
        """Every gap is already generated: a tiny window so the model step is instant and Merge shows the result."""
        import torch
        frames = np.zeros((9, 64, 64, 3), np.float32)
        info = {**fill, "start": 0, "count": 0, "length": 9, "context": 1, "targets": [], "reverse": False}
        return (torch.from_numpy(frames), torch.from_numpy(np.zeros((9, 64, 64), np.float32)), torch.from_numpy(frames[:1]), torch.from_numpy(frames[:1]),
                torch.from_numpy(frames[:1]), 64, 64, 9, float(Fraction(fill["fps"])), info, "nothing missing")




def _blocked(n: int):
    """Skip every node behind this one (the video model and Merge)."""
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:  # pragma: no cover - outside ComfyUI
        return (None,) * n
    return tuple(ExecutionBlocker(None) for _ in range(n))


def _queue_again(prompt: dict, extra_pnginfo) -> None:
    """Queue the same workflow once more so the next gap gets generated."""
    import threading
    import urllib.request
    from server import PromptServer
    server = PromptServer.instance
    body = {"prompt": prompt, "client_id": server.client_id, "extra_data": {"extra_pnginfo": extra_pnginfo or {}}}
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}/prompt", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    threading.Thread(target=lambda: urllib.request.urlopen(request, timeout=30).read(), daemon=True).start()


class PlateLockFillMerge:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("locked", "side_by_side", "preview", "report")
    OUTPUT_NODE = True
    DESCRIPTION = ("Takes the model's frames for this window, keeps only the missing ones (colour-matched), re-renders the "
                   "clip and shows what changed, like PlateLock. Runs the workflow again until no gap is left.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"window": ("PLATELOCK_WINDOW",), "images": ("IMAGE", {"tooltip": "Decoded model output for this window."})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    def run(self, window, images, prompt=None, extra_pnginfo=None):
        from comfy_api.input_impl import VideoFromFile
        from .platelock.genfill import patched_frames, read_range, render_patched, save_patches
        fps = Fraction(window["fps"])
        if not window["targets"]:
            side_by_side, preview, report = _overview(window, Path(window["clip"]), set(), fps)
            return _with_viewer(Path(window["viewer"]),
                                (VideoFromFile(window["clip"]), VideoFromFile(str(side_by_side)), _image(preview), report))
        gen = (images.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        start, count = window["start"], window["count"]
        if len(gen) < count:
            raise ValueError(f"Model returned {len(gen)} frames; this window needs {window['length']} "
                             "(trim any reference frames first).")
        gen = gen[:count][::-1] if window["reverse"] else gen[:count]
        info = probe(window["clip"])
        actual = read_range(window["clip"], start, count, size=(gen.shape[2], gen.shape[1]))
        written = save_patches(window["work"], start, gen, actual, [t - start for t in window["targets"]],
                               (info.width, info.height))
        out = Path(window["clip"]).with_name(Path(window["clip"]).stem + "_generated" + Path(window["clip"]).suffix)
        render_patched(window["clip"], window["work"], out, fps, window["codec"])
        done = set(patched_frames(window["work"]))
        todo = [t for t in window["missing"] if t not in done]
        side_by_side, preview, report = _overview(window, out, done, fps)
        if todo and written and prompt:
            _continue[0] += 1
            _queue_again(prompt, extra_pnginfo)
            report += f"\n{len(todo)} frame(s) left: running again automatically"
        return _with_viewer(Path(window["viewer"]),
                            (VideoFromFile(str(out)), VideoFromFile(str(side_by_side)), _image(preview), report))


def _overview(window: dict, final_clip: Path, generated: set[int], fps: Fraction):
    """The same side-by-side, preview and report as PlateLock, showing the finished clip."""
    side = json.loads(Path(window["sidecar"]).read_text())
    ops = [Op(**o) for o in side["ops"]]
    plate_rgb = decode_small_rgb(side["plate"]["path"], 480)
    gen_rgb = decode_small_rgb(side["generated"]["path"], 480)
    final_rgb = decode_small_rgb(str(final_clip), 480)
    out = final_clip.with_name(final_clip.stem + "_side_by_side.mp4")
    compare_video(plate_rgb, gen_rgb, ops, side["summary"], out, fps, final=final_rgb, generated=generated)
    preview = summary_image(plate_rgb, gen_rgb, ops, side["summary"], final=final_rgb, generated=generated)
    drift, exact = _exactness(plate_rgb, final_clip, ops, generated, side["plate"]["width"])
    _export_viewer(Path(window["viewer"]), side["plate"]["path"], side["generated"]["path"], final_clip, ops, generated,
                   drift, exact, fps, "RESULT", sources=False)
    return out, preview, "\n".join([_report(side["summary"], generated)] + report_lines(exact))


def _gray(images) -> np.ndarray:
    import cv2
    a = images.cpu().numpy()
    g = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).astype(np.float32)
    h = max(2, round(g.shape[1] * 256 / g.shape[2] / 2) * 2)
    return np.stack([cv2.resize(f, (256, h), interpolation=cv2.INTER_AREA) for f in g])


class PlateLockImages:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("locked", "preview", "report")
    DESCRIPTION = "Same as PlateLock, for short IMAGE batches. Use PlateLock for long or 4K clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plate": ("IMAGE",), "generated": ("IMAGE",)}, "optional": {"mask": ("MASK",)}}

    def run(self, plate, generated, mask=None):
        import torch
        r = analyze_arrays(_gray(generated), _gray(plate), mask=_mask_array(mask))
        gen8 = (generated.cpu().numpy() * 255).astype(np.uint8)
        plate8 = (plate.cpu().numpy() * 255).astype(np.uint8)
        frames = [f if f is not None else np.zeros_like(gen8[0]) for f in locked_frames(gen8, r.ops)]
        preview = summary_image(plate8, gen8, r.ops, r.summary)
        return (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0), _image(preview), _report(r.summary))


NODE_CLASS_MAPPINGS = {"PlateLock": PlateLock, "PlateLockImages": PlateLockImages,
                       "PlateLockFillWindow": PlateLockFillWindow, "PlateLockFillMerge": PlateLockFillMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"PlateLock": "PlateLock", "PlateLockImages": "PlateLock (images)",
                              "PlateLockFillWindow": "PlateLock Fill · Window (for Wan VACE / LTX)",
                              "PlateLockFillMerge": "PlateLock Fill · Merge"}
