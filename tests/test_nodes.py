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
        self.assertEqual(sorted(m), ["PlateLock", "PlateLockExport", "PlateLockFillMerge", "PlateLockFillWindow", "PlateLockImages", "PlateLockStudio"])
        for cls in m.values():
            cls.INPUT_TYPES()
        out = m["PlateLock"]().run(self.pv, self.gv, "25", "mp4")
        clip, side, preview, report, fill = out["result"]
        viewer = Path(fill["viewer"])
        self.assertEqual(out["ui"]["platelock_viewer"][0]["subfolder"], "platelock/" + viewer.name)
        manifest = json.loads((viewer / "manifest.json").read_text())
        self.assertEqual((len(manifest["ops"]), len(manifest["drift"]), manifest["counts"]["l"]), (80, 80, 80))
        self.assertIn("timing:", report)
        info = probe(clip.get_stream_source())
        self.assertEqual((info.frames, info.fps), (80, 25))
        self.assertEqual(probe(side.get_stream_source()).frames, 80)
        self.assertEqual(preview.ndim, 4)
        self.assertIn("filled 2", report)
        sidecar = json.loads(Path(fill["sidecar"]).read_text())
        self.assertEqual(len(sidecar["time_remap_seconds"]), 80)
        self.assertEqual(set(sidecar["summary"]["interpolated"]), {20, 50})
        window = m["PlateLockFillWindow"]().run(fill)[9]
        merge = m["PlateLockFillMerge"]()
        self.assertEqual(merge.check_lazy_status(window), [], "no video model run unless 'generate'")
        _, _, preview, merged_report = merge.run(window, None)["result"]
        self.assertEqual(merged_report, report)
        self.assertEqual(preview.shape[0], 1)

    def test_analysis_does_not_render_or_fill(self):
        from unittest.mock import patch
        module = sys.modules["comfyui_platelock.nodes"]
        with patch.object(module, "render", side_effect=AssertionError("analysis rendered")), \
             patch.object(module, "model_fill", side_effect=AssertionError("analysis interpolated")):
            out = self.pack.NODE_CLASS_MAPPINGS["PlateLock"]().run(self.pv, self.gv, "plate", "mp4", "Analyse only")
        clip, side, _, report, state = out["result"]
        self.assertEqual(clip.get_stream_source(), self.gv.path)
        self.assertEqual(side.get_stream_source(), self.gv.path)
        manifest = json.loads((Path(state["viewer"]) / "manifest.json").read_text())
        self.assertTrue(manifest["analysis_only"])
        self.assertEqual(manifest["counts"]["l"], 0)
        self.assertEqual(len(manifest["drift"]), 80)
        for i in (20, 50):
            self.assertIsNone(manifest["drift"][i])
        self.assertIn("no frames repaired", report)
        with self.assertRaisesRegex(ValueError, "Analyse only"):
            self.pack.NODE_CLASS_MAPPINGS["PlateLockFillWindow"]().run(state)

    def test_explicit_rife_requires_model(self):
        with self.assertRaisesRegex(ValueError, "Connect Load"):
            self.pack.NODE_CLASS_MAPPINGS["PlateLock"]().run(self.pv, self.gv, "plate", "mp4", "Repair timing · RIFE")

    def test_studio_analysis_has_no_video_and_export_rejects_it(self):
        m = self.pack.NODE_CLASS_MAPPINGS
        result = m["PlateLockStudio"]().run(self.pv, self.gv)
        review, fill, locked, comparison = result["result"]
        self.assertIsNone(review["video"])
        self.assertEqual(m["PlateLockStudio"].RETURN_TYPES, ("PLATELOCK_RESULT", "PLATELOCK_FILL", "VIDEO", "VIDEO"))
        self.assertTrue(fill["analysis_only"])
        with self.assertRaisesRegex(ValueError, "Analysis has no repaired video"):
            m["PlateLockExport"]().run(review)

    def test_framing_only_preserves_extra_frames_without_retiming(self):
        from platelock.demo import examples
        from platelock.video import FrameWriter, probe
        from unittest.mock import patch
        spatial = sys.modules.get('comfyui_platelock.platelock.spatial')
        if spatial is None:
            import importlib
            spatial = importlib.import_module('comfyui_platelock.platelock.spatial')
        plate, cases, _ = examples()
        ai = np.concatenate([cases['framing'], cases['framing'][-2:]])
        folder = Path(self.tmp.name) / 'framing_only'; folder.mkdir(exist_ok=True)
        for name, frames in [('plate',plate),('ai',ai)]:
            writer = FrameWriter(folder/f'{name}.mp4',320,180,24,'h264')
            for frame in frames: writer.write(frame)
            writer.close()
        # This deliberately contains an unsafe timing plan. It must not drive
        # the frame-preserving spatial operation, nor be reported as repaired.
        unsafe = {'summary': {'interpolated':[20], 'missing_plate_frames':[59],
                             'segments':[{'confidence':-2}]}}
        with patch('comfyui_platelock.platelock.render.Filler.frame',
                   side_effect=AssertionError('framing-only interpolated')):
            video, _, _, state = spatial.framing_only(
                str(folder/'plate.mp4'),str(folder/'ai.mp4'),folder/'out',unsafe)
        info = probe(video.get_stream_source())
        self.assertEqual((info.frames,info.fps),(len(ai),24))
        manifest = json.loads((Path(state['viewer'])/'manifest.json').read_text())
        self.assertTrue(manifest['timing_preserved'])
        self.assertEqual(manifest['timing_analysis'],unsafe['summary'])
        self.assertTrue(all(o['op']=='copy' and o['gen']==o['plate'] for o in manifest['ops']))
        self.assertLess(manifest['framing']['rendered_error_px'],1)
        # An unsupported timebase cannot quietly retime the generated video.
        pi,gi=probe(folder/'plate.mp4'),probe(folder/'ai.mp4')
        from dataclasses import replace
        with patch.object(spatial,'probe',side_effect=[pi,replace(gi,fps=25)]), \
             self.assertRaisesRegex(ValueError,'matching input frame rates'):
            spatial.framing_only(str(folder/'plate.mp4'),str(folder/'ai.mp4'),folder/'out',unsafe)

    def test_studio_repair_exports_and_pending_ltx_cannot(self):
        m = self.pack.NODE_CLASS_MAPPINGS
        review, _, locked, comparison = m["PlateLockStudio"]().run(self.pv, self.gv, "Repair", "RIFE", object(), allow_ambiguous_timing=True)["result"]
        video, = m["PlateLockExport"]().run(review)
        self.assertTrue(Path(video.get_stream_source()).is_file())
        manifest=json.loads((Path(review["viewer"])/"manifest.json").read_text())
        self.assertEqual(set(manifest["summary"]["interpolated"]), {20,50})
        with self.assertRaisesRegex(ValueError, "LTX footage is still missing"):
            m["PlateLockExport"]().run({**review, "pending_ltx": True})

    def test_studio_blocks_weak_candidate_edits_before_loading_models(self):
        from unittest.mock import patch
        module = sys.modules["comfyui_platelock.nodes"]
        analysis = module.PlateLock().run(self.pv,self.gv,"plate","mp4","Analyse only")
        analysis["result"][4]["summary"]["segments"][0]["confidence"] = 1.9
        with patch.object(module.PlateLock,"run",return_value=analysis), \
             patch("comfyui_platelock.platelock.backends.load_rife",side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError,"weak sequence evidence"):
                module.PlateLockStudio().run(self.pv,self.gv,"Repair")

    def test_constant_shift_repair_reduces_measured_displacement(self):
        import cv2
        shifted = np.stack([cv2.warpAffine(f,np.float32([[1,0,5],[0,1,-3]]),(256,144),
                                         borderMode=cv2.BORDER_REFLECT) for f in self.plate])
        path=Path(self.tmp.name)/"shifted.mp4"
        write_video(path,shifted,"24")
        out=self.pack.NODE_CLASS_MAPPINGS["PlateLockStudio"]().run(
            self.pv,VideoFromFile(str(path)),"Repair","None",position="Correct constant shift",
            allow_ambiguous_timing=True)
        review=out["result"][0]
        manifest=json.loads((Path(review["viewer"])/"manifest.json").read_text())
        before=np.median([d["shift"] for d in manifest["before_drift"] if d and d["shift"] is not None])
        self.assertGreater(before,4)
        self.assertLess(manifest["exactness"]["median_shift_px"],1)
        self.assertIn("Exposed borders are black",review["report"])

    def test_debug_mode_makes_no_frames(self):
        from platelock.video import probe
        m = self.pack.NODE_CLASS_MAPPINGS
        clip, _, _, report, _ = m["PlateLock"]().run(self.pv, self.gv, "plate", "mp4", "none (debug)")["result"]
        self.assertIn("debug: no frames were made", report)
        self.assertEqual(probe(clip.get_stream_source()).frames, 80)

    def test_interpolation_model_fills_each_gap_once(self):
        self.FrameInterpolate.calls.clear()
        m = self.pack.NODE_CLASS_MAPPINGS
        _, _, _, report, _ = m["PlateLock"]().run(self.pv, self.gv, "plate", "mp4", "Repair timing · RIFE", interp_model=object())["result"]
        self.assertIn("filled 2", report)
        self.assertTrue(self.FrameInterpolate.calls)
        self.assertEqual(set(self.FrameInterpolate.calls), {2})

    def test_generative_fill_of_missing_tail(self):
        from platelock.video import decode_small_rgb
        m = self.pack.NODE_CLASS_MAPPINGS
        out = m["PlateLock"]().run(self.pv, self.sv, "plate", "mp4", "Repair timing + missing footage · LTX", interp_model=object())
        clip, _, _, report, fill = out["result"]
        self.assertIn("MISSING 10", report)
        out = m["PlateLockFillWindow"]().run(fill)
        control, masks, context, first, last, width, height, length, fps, window, status = out
        self.assertEqual(m["PlateLockFillMerge"]().check_lazy_status(window), ["images"])
        self.assertEqual(fps, 24.0)
        self.assertEqual((control.shape[0], length, width % 32, height % 32), (33, 33, 0, 0))
        self.assertEqual((window["start"], window["count"]), (53, 27))
        self.assertEqual(context.shape[0], 17)
        generate = np.asarray(masks).reshape(33, -1).max(axis=1) > 0
        self.assertEqual(int(generate[:27].sum()), 10)
        self.assertTrue(np.allclose(control[generate], 0.5))
        # A perfect "model": the restyled plate for this window (the plate ends 6 frames before the window does).
        start = window["start"]
        model_out = np.zeros((33, height, width, 3), np.float32)
        plate = restyle(self.plate[start:start + 33])
        import cv2
        model_out[:len(plate)] = np.stack([cv2.resize(f, (width, height)) for f in plate])[..., None]
        video, side, preview, report = m["PlateLockFillMerge"]().run(window, model_out.view(Tensor))["result"]
        self.assertIn("generated 10 frames with the video model", report)
        self.assertNotIn("MISSING", report)
        self.assertNotIn("running again", report)
        self.assertTrue(Path(side.get_stream_source()).is_file())
        self.assertEqual(preview.shape[0], 1)
        frames = decode_small_rgb(video.get_stream_source(), 256)
        self.assertEqual(len(frames), 80)
        self.assertTrue(all(frames[i].mean() > 10 for i in range(70, 80)))

    def test_platelock_images(self):
        frames = lambda a: np.repeat(a[..., None], 3, axis=3).astype(np.float32).view(Tensor)
        imgs, preview, report = self.pack.NODE_CLASS_MAPPINGS["PlateLockImages"]().run(frames(self.plate), frames(self.gen))
        self.assertEqual(imgs.shape, (80, 144, 256, 3))
        self.assertIn("filled 2", report)


if __name__ == "__main__":
    unittest.main()
