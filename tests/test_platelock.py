import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from platelock.core import analyze_arrays  # noqa: E402
from platelock.features import detect_duplicates  # noqa: E402


def synthetic_plate(n=160, size=(256, 144), seed=1):
    """Textured background with two independently moving textured shapes; varying speed, including a pause."""
    rng = np.random.default_rng(seed)
    w, h = size
    bg = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3) * 0.4 + 0.3
    tex = cv2.GaussianBlur(rng.random((40, 40)).astype(np.float32), (0, 0), 1.5)
    speed = 1.0 + 0.8 * np.sin(np.linspace(0, 3 * np.pi, n))
    speed[70:76] = 0.0
    x = 20 + np.cumsum(speed)
    frames = []
    for i in range(n):
        f = bg.copy()
        for cx, cy in ((x[i], 40 + 20 * np.sin(i / 15)), (w - x[i] * 0.7, 100 + 10 * np.cos(i / 9))):
            x0, y0 = int(cx) % (w - 40), int(cy) % (h - 40)
            f[y0:y0 + 40, x0:x0 + 40] = 0.2 + 0.7 * tex
        frames.append(f)
    return np.stack(frames)


def restyle(frames, seed=2):
    rng = np.random.default_rng(seed)
    out = 1.0 - frames ** 0.7
    return np.clip(out + rng.normal(0, 0.02, out.shape), 0, 1).astype(np.float32)


class PlateLockTest(unittest.TestCase):
    def check(self, truth, plate, expect_interp=(), expect_missing=()):
        gen = restyle(plate[truth])
        r = analyze_arrays(gen, plate)
        wrong = [o.plate for o in r.ops if o.op == "copy" and truth[o.gen] != o.plate]
        self.assertEqual(wrong, [])
        self.assertTrue(set(expect_interp) <= set(r.summary["interpolated"]), r.summary["interpolated"])
        self.assertEqual(r.summary["missing_plate_frames"], list(expect_missing))
        self.assertEqual(len(r.ops), len(plate))
        return r

    def test_identity(self):
        plate = synthetic_plate()
        r = self.check(list(range(len(plate))), plate)
        self.assertEqual(r.summary["interpolated"], [])
        self.assertEqual(r.summary["dropped_gen_frames"], [])

    def test_offset_dupes_skips_and_short_tail(self):
        plate = synthetic_plate()
        n = len(plate)
        truth = [0, 0] + list(range(n))   # two padded head frames
        truth[42] = truth[41]               # repeat replacing plate 40
        truth.insert(101, truth[100])       # inserted repeat
        truth.remove(120)                   # single skip
        truth.remove(130)
        truth.remove(131)                   # double skip
        truth = truth[:-4]                  # last 4 plate frames never generated
        self.check(truth, plate, expect_interp={40, 120, 130, 131}, expect_missing=range(n - 4, n))

    def test_long_generation_is_trimmed(self):
        plate = synthetic_plate()
        n = len(plate)
        truth = list(range(n)) + [n - 1] * 6
        r = self.check(truth, plate)
        # Any of the identical trailing frames may be the one kept.
        self.assertEqual(len(r.summary["dropped_gen_frames"]), 6)
        self.assertTrue(all(truth[g] == n - 1 for g in r.summary["dropped_gen_frames"]))

    def test_natural_pause_is_not_a_duplicate(self):
        plate = synthetic_plate()
        dup = detect_duplicates(restyle(plate))
        self.assertFalse(dup[70:77].any())

    def test_exact_repeat_is_detected(self):
        plate = synthetic_plate()
        truth = list(range(len(plate)))
        truth[30] = 29
        dup = detect_duplicates(plate[truth])
        self.assertTrue(dup[30])


class RenderTest(unittest.TestCase):
    def test_cli_render_matches_plate_count(self):
        from platelock.video import probe
        plate = synthetic_plate(n=60)
        truth = list(range(60))
        truth.remove(30)
        gen = restyle(plate[truth])
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, frames, fps in (("plate.mp4", plate, "24"), ("gen.mp4", gen, "25")):
                raw = (np.repeat(frames[..., None], 3, axis=3) * 255).astype(np.uint8)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "256x144", "-r", fps, "-i", "-", "-c:v", "libx264", "-qp", "0",
                                "-pix_fmt", "yuv444p", str(d / name)], input=raw.tobytes(), check=True)
            from platelock.cli import main
            code = main([str(d / "plate.mp4"), str(d / "gen.mp4"), "--out", str(d / "out"),
                         "--render", "--codec", "h264", "--cache", str(d / "cache"), "--fps", "25"])
            self.assertEqual(code, 0)
            info = probe(d / "out" / "gen_locked.mp4")
            self.assertEqual(info.frames, 60)
            self.assertEqual(info.fps, Fraction(25))


class ColourMatchTest(unittest.TestCase):
    def test_misaligned_model_frames_keep_contrast(self):
        from platelock.genfill import colour_match
        rng = np.random.default_rng(1)
        real = rng.integers(0, 256, (3, 64, 96, 3)).astype(np.uint8)
        model = np.clip(np.roll(real, 7, axis=2) * 0.8 + 20, 0, 255).astype(np.uint8)
        matched = colour_match(model, real, [0, 1, 2])
        self.assertAlmostEqual(matched.std() / real.std(), 1.0, delta=0.05)
        self.assertAlmostEqual(matched.mean(), real.mean(), delta=2)


class NextWindowTest(unittest.TestCase):
    def test_tail_head_and_long_gaps(self):
        from platelock.genfill import next_window
        tail = next_window(list(range(362, 375)), 375)
        self.assertEqual((tail["start"], tail["context"], tail["length"], tail["reverse"]), (345, 17, 33, False))
        self.assertEqual(tail["targets"], list(range(362, 375)))
        head = next_window([0, 1, 2, 3, 4], 100)
        self.assertTrue(head["reverse"])
        self.assertEqual((head["start"], head["count"], head["length"]), (0, 22, 25))
        long = next_window(list(range(100, 400)), 400)
        self.assertEqual((len(long["targets"]), long["length"]), (104, 121))
        short_context = next_window([5, 6], 50)
        self.assertEqual((short_context["start"], short_context["context"]), (4, 1))
        self.assertIsNone(next_window([], 50))


class DriftTest(unittest.TestCase):
    def test_finds_known_shift_and_time_offset(self):
        import cv2
        from platelock import drift
        rng = np.random.default_rng(1)
        world = cv2.GaussianBlur((rng.random((400, 900)) * 255).astype(np.uint8), (0, 0), 2)
        plate = np.stack([cv2.cvtColor(world[50:230, 6 * t:6 * t + 320], cv2.COLOR_GRAY2RGB) for t in range(40)])
        result = np.roll(plate, 3, axis=2)
        result[20] = plate[21]
        d = drift.measure(plate, result, [True] * 40, plate_width=320)
        s = drift.summary(d)
        self.assertEqual(s["off_in_time"], [20])
        self.assertAlmostEqual(s["median_shift_px"], 3.0, delta=0.3)
        self.assertTrue(any("look like a neighbouring plate frame" in l for l in drift.report_lines(s)))


if __name__ == "__main__":
    unittest.main()
