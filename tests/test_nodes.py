"""Load the pack the way ComfyUI does, with torch, comfy_api and folder_paths stubbed."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_platelock import restyle, synthetic_plate  # noqa: E402


class Tensor(np.ndarray):
    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return np.asarray(self)


class VideoFromFile:
    def __init__(self, path):
        self.path = path

    def get_stream_source(self):
        return self.path

    def get_active_trim_window(self):
        return 0, 0


def write_video(path, frames, fps):
    raw = (np.repeat(frames[..., None], 3, axis=3) * 255).astype(np.uint8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x144", "-r", fps,
                    "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)],
                   input=raw.tobytes(), check=True)


class NodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        run = Path(cls.tmp.name)
        cls.out = run / "out"
        class FrameInterpolate:
            calls = []

            @classmethod
            def execute(cls, model, images, multiplier):
                cls.calls.append(multiplier)
                a, b = np.asarray(images[0]), np.asarray(images[1])
                frames = [a + (b - a) * i / multiplier for i in range(multiplier + 1)]
                return types.SimpleNamespace(args=(np.stack(frames).view(Tensor),))

        cls.FrameInterpolate = FrameInterpolate
        stubs = {
            "comfy_extras": types.ModuleType("comfy_extras"),
            "comfy_extras.nodes_frame_interpolation": types.SimpleNamespace(FrameInterpolate=FrameInterpolate),
            "torch": types.SimpleNamespace(from_numpy=lambda a: np.asarray(a).view(Tensor)),
            "comfy_api": types.ModuleType("comfy_api"),
            "comfy_api.input_impl": types.SimpleNamespace(VideoFromFile=VideoFromFile),
            "folder_paths": types.SimpleNamespace(get_output_directory=lambda: str(run / "out"),
                                                  get_temp_directory=lambda: str(run / "temp")),
        }
        cls._saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location("comfyui_platelock", ROOT / "__init__.py",
                                                      submodule_search_locations=[str(ROOT)])
        cls.pack = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_platelock"] = cls.pack
        spec.loader.exec_module(cls.pack)

        cls.plate = synthetic_plate(n=80)
        truth = list(range(80))
        truth[20] = 19
        truth.remove(50)
        cls.gen = restyle(cls.plate[truth])
        write_video(run / "plate.mp4", cls.plate, "24")
        write_video(run / "gen.mp4", cls.gen, "24")
        cls.pv, cls.gv = VideoFromFile(str(run / "plate.mp4")), VideoFromFile(str(run / "gen.mp4"))
        cls.short_truth = list(range(70))
        write_video(run / "short.mp4", restyle(cls.plate[cls.short_truth]), "24")
        cls.sv = VideoFromFile(str(run / "short.mp4"))

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("comfyui_platelock", None)
        cls.tmp.cleanup()

    def test_platelock_video(self):
        from platelock.video import probe
        m = self.pack.NODE_CLASS_MAPPINGS
        self.assertEqual(sorted(m), ["PlateLock", "PlateLockFillMerge", "PlateLockFillWindow", "PlateLockImages"])
        for cls in m.values():
            cls.INPUT_TYPES()
        clip, side, preview, report, fill = m["PlateLock"]().run(self.pv, self.gv, "25", "mp4")
        info = probe(clip.get_stream_source())
        self.assertEqual((info.frames, info.fps), (80, 25))
        self.assertEqual(probe(side.get_stream_source()).frames, 80)
        self.assertEqual(preview.ndim, 4)
        self.assertIn("filled 2", report)
        sidecar = json.loads(Path(fill["sidecar"]).read_text())
        self.assertEqual(len(sidecar["time_remap_seconds"]), 80)
        self.assertEqual(set(sidecar["summary"]["interpolated"]), {20, 50})

    def test_interpolation_model_fills_each_gap_once(self):
        self.FrameInterpolate.calls.clear()
        m = self.pack.NODE_CLASS_MAPPINGS
        _, _, _, report, _ = m["PlateLock"]().run(self.pv, self.gv, "plate", "mp4", interp_model=object())
        self.assertIn("filled 2", report)
        self.assertTrue(self.FrameInterpolate.calls)
        self.assertEqual(set(self.FrameInterpolate.calls), {2})

    def test_generative_fill_of_missing_tail(self):
        from platelock.video import decode_small_rgb
        m = self.pack.NODE_CLASS_MAPPINGS
        clip, _, _, report, fill = m["PlateLock"]().run(self.pv, self.sv, "plate", "mp4")
        self.assertIn("MISSING 10", report)
        out = m["PlateLockFillWindow"]().run(fill, 0, 25, 9, "missing", 1280)
        control, masks, context, first, last, width, height, length, window, status = out
        self.assertEqual((control.shape[0], length, width % 32, height % 32), (25, 25, 0, 0))
        self.assertEqual(window["start"], 61)
        self.assertEqual(context.shape[0], 9)
        generate = np.asarray(masks).reshape(25, -1).max(axis=1) > 0
        self.assertEqual(int(generate[:19].sum()), 10)
        self.assertTrue(np.allclose(control[generate], 0.5))
        # A perfect "model": the restyled plate for this window (the plate ends 6 frames before the window does).
        start = window["start"]
        model_out = np.zeros((25, height, width, 3), np.float32)
        plate = restyle(self.plate[start:start + 25])
        import cv2
        model_out[:len(plate)] = np.stack([cv2.resize(f, (width, height)) for f in plate])[..., None]
        video, status = m["PlateLockFillMerge"]().run(window, model_out.view(Tensor))
        self.assertEqual(status, "all frames generated")
        frames = decode_small_rgb(video.get_stream_source(), 256)
        self.assertEqual(len(frames), 80)
        self.assertTrue(all(frames[i].mean() > 10 for i in range(70, 80)))
        with self.assertRaises(ValueError):
            m["PlateLockFillWindow"]().run(fill, 1, 25, 9, "missing", 1280)

    def test_platelock_images(self):
        frames = lambda a: np.repeat(a[..., None], 3, axis=3).astype(np.float32).view(Tensor)
        imgs, preview, report = self.pack.NODE_CLASS_MAPPINGS["PlateLockImages"]().run(frames(self.plate), frames(self.gen))
        self.assertEqual(imgs.shape, (80, 144, 256, 3))
        self.assertIn("filled 2", report)


if __name__ == "__main__":
    unittest.main()
