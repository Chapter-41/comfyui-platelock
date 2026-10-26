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
from test_realign import restyle, synthetic_plate  # noqa: E402


class Tensor(np.ndarray):
    def cpu(self):
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
        stubs = {
            "torch": types.SimpleNamespace(from_numpy=lambda a: np.asarray(a).view(Tensor)),
            "comfy_api": types.ModuleType("comfy_api"),
            "comfy_api.input_impl": types.SimpleNamespace(VideoFromFile=VideoFromFile),
            "folder_paths": types.SimpleNamespace(get_output_directory=lambda: str(run / "out"),
                                                  get_temp_directory=lambda: str(run / "temp")),
        }
        cls._saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location("comfyui_realign", ROOT / "__init__.py",
                                                      submodule_search_locations=[str(ROOT)])
        cls.pack = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_realign"] = cls.pack
        spec.loader.exec_module(cls.pack)

        cls.plate = synthetic_plate(n=80)
        truth = list(range(80))
        truth[20] = 19
        truth.remove(50)
        cls.gen = restyle(cls.plate[truth])
        write_video(run / "plate.mp4", cls.plate, "24")
        write_video(run / "gen.mp4", cls.gen, "24")
        cls.pv, cls.gv = VideoFromFile(str(run / "plate.mp4")), VideoFromFile(str(run / "gen.mp4"))

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("comfyui_realign", None)
        cls.tmp.cleanup()

    def test_realign_video(self):
        from realign.video import probe
        m = self.pack.NODE_CLASS_MAPPINGS
        self.assertEqual(sorted(m), ["Realign", "RealignImages"])
        for cls in m.values():
            cls.INPUT_TYPES()
        clip, side, preview, report = m["Realign"]().run(self.pv, self.gv, "25", "mp4")
        info = probe(clip.get_stream_source())
        self.assertEqual((info.frames, info.fps), (80, 25))
        self.assertEqual(probe(side.get_stream_source()).frames, 80)
        self.assertEqual(preview.ndim, 4)
        self.assertIn("filled 2", report)
        sidecar = json.loads(next(self.out.glob("realign/*.json")).read_text())
        self.assertEqual(len(sidecar["time_remap_seconds"]), 80)
        self.assertEqual(set(sidecar["summary"]["interpolated"]), {20, 50})

    def test_realign_images(self):
        frames = lambda a: np.repeat(a[..., None], 3, axis=3).astype(np.float32).view(Tensor)
        imgs, preview, report = self.pack.NODE_CLASS_MAPPINGS["RealignImages"]().run(frames(self.plate), frames(self.gen))
        self.assertEqual(imgs.shape, (80, 144, 256, 3))
        self.assertIn("filled 2", report)


if __name__ == "__main__":
    unittest.main()
