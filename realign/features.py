"""Per-frame descriptors that survive restyling.

Generated videos change colour, texture and lighting, so raw pixels are a poor
match. Structure (edges of a blurred thumbnail) and how that structure changes
between frames are far more stable.
"""
from __future__ import annotations

import cv2
import numpy as np

THUMB = (64, 36)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.reshape(v.shape[0], -1)
    v = v - v.mean(axis=1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def thumbnails(frames: np.ndarray, size: tuple[int, int] = THUMB) -> np.ndarray:
    return np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])


def structure(thumbs: np.ndarray) -> np.ndarray:
    """Edge magnitude of a lightly blurred thumbnail, with per-frame contrast normalised away."""
    out = []
    for t in thumbs:
        t = cv2.GaussianBlur(t, (0, 0), 1.0)
        t = (t - t.mean()) / (t.std() + 1e-6)
        gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
        out.append(np.sqrt(gx * gx + gy * gy))
    return np.stack(out)


def frame_deltas(frames: np.ndarray) -> np.ndarray:
    """Mean absolute change from the previous frame (first frame gets the second's value)."""
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return np.concatenate([d[:1], d]) if len(d) else np.zeros(len(frames), np.float32)


def _sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)


def detect_duplicates(frames: np.ndarray, ratio: float = 0.25, min_motion: float = 0.3,
                      jump: float = 3.0, local: float = 0.75, window: int = 8) -> np.ndarray:
    """Flag frames that repeat the previous one, robust to grain and AI shimmer.

    With previous/current/next frames, A = |cur-prev|^2, B = |next-cur|^2 and
    C = |next-prev|^2. Per-frame noise adds the same constant to all three, so
    it cancels in C-B and C-A. A repeat has C ~ B (current adds nothing over
    previous) while C-A carries the two-step motion. Normal motion gives
    C-B ~ C-A. A sudden move after a normal frame also gives C ~ B, so a
    repeat must additionally sit far below the local motion (A) and be
    followed by the catch-up step (B >> A). Frames with no motion relative to
    the clip are left alone: a repeat there is invisible and harmless.
    """
    a, b, c, local_a = _triangle(frames, window)
    dup = np.zeros(len(frames), bool)
    if a is None:
        return dup
    ca, cb = c - a, c - b
    ref = np.median(np.abs(ca)) + 1e-12
    hit = (ca > min_motion * ref) & (cb < ratio * ca) & (b > jump * a) & (a < local * local_a)
    hit |= a < 1e-9  # bit-exact repeats
    dup[1:-1] = hit
    return dup


def holds(frames: np.ndarray, local: float = 0.5, window: int = 8) -> np.ndarray:
    """Frames that barely change from the previous one relative to the local motion (natural pauses)."""
    a, _, _, local_a = _triangle(frames, window)
    out = np.zeros(len(frames), bool)
    if a is not None:
        out[1:-1] = a < local * local_a
    return out


def _triangle(frames: np.ndarray, window: int):
    th = np.stack([cv2.GaussianBlur(t, (0, 0), 1.0) for t in thumbnails(frames, (128, 72))])
    if len(th) < 3:
        return None, None, None, None
    a = _sq(th[1:-1], th[:-2])
    b = _sq(th[2:], th[1:-1])
    c = _sq(th[2:], th[:-2])
    local_a = np.array([np.median(np.delete(a[max(0, i - window):i + window + 1], min(i, window)))
                        for i in range(len(a))])
    return a, b, c, local_a


def structure_maps(frames: np.ndarray, size: tuple[int, int] = THUMB) -> np.ndarray:
    return structure(thumbnails(frames, size))


def agreement_weights(gen_st: np.ndarray, plate_st: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Per-pixel weight: how strongly the gen follows the plate over time.

    Regions the model kept (actors) correlate with the plate across the clip;
    regions it invented (a replaced background) do not and would otherwise
    dominate the match with their own motion.
    """
    if pairs:
        gi = np.array([g for g, _ in pairs])
        pi = np.array([p for _, p in pairs])
    else:
        pi = np.arange(len(plate_st))
        gi = np.round(pi * (len(gen_st) - 1) / max(len(plate_st) - 1, 1)).astype(int)
    a, b = gen_st[gi], plate_st[pi]
    za = (a - a.mean(0)) / (a.std(0) + 1e-6)
    zb = (b - b.mean(0)) / (b.std(0) + 1e-6)
    corr = (za * zb).mean(0)
    w = cv2.GaussianBlur(np.clip(corr, 0, None) ** 2, (0, 0), 1.0)
    return w / (w.max() + 1e-8)


def descriptors(st: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Flattened (optionally weighted) structure maps, shape (frames, D)."""
    if weights is not None:
        st = st * weights[None]
    return st.reshape(len(st), -1)


def cost_matrix(gen: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Structure cost (M gen frames x N plate frames), lower is better, in [0, 2]."""
    return 1.0 - _normalize(gen) @ _normalize(plate).T


class MotionTransition:
    """Cost of a DTW step by comparing the gen change with the plate change it spans.

    Per-frame motion descriptors break right after a skip (the gen frame moved
    three plate frames at once), so motion is scored on the step itself.
    """

    def __init__(self, gen: np.ndarray, plate: np.ndarray, weight: float, max_step: int):
        self.gen, self.weight = gen, weight
        self.n = len(plate)
        self.plate_diff: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        energies = []
        for dp in range(1, max_step + 1):
            pd = plate[dp:] - plate[:-dp]
            pd = pd - pd.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(pd, axis=1)
            self.plate_diff[dp] = (pd / np.maximum(norm, 1e-8)[:, None], norm / dp)
            energies.append(norm / dp)
        gd1 = np.linalg.norm(gen[1:] - gen[:-1], axis=1)
        self.ref = float(np.median(np.concatenate([energies[0], gd1]))) + 1e-8

    def __call__(self, g: int, dg: int, dp: int) -> np.ndarray:
        """Cost for steps (g-dg, p-dp) -> (g, p) for every p; entries p < dp are unused."""
        out = np.zeros(self.n)
        if self.weight <= 0:
            return out
        v = self.gen[g] - self.gen[g - dg]
        v = v - v.mean()
        gn = float(np.linalg.norm(v))
        pdn, pe = self.plate_diff[dp]
        ncc = pdn @ (v / max(gn, 1e-8))
        # Correlating two near-static changes is noise; fade the term out there.
        conf = np.sqrt(np.clip(gn / dg / self.ref, 0, 1) * np.clip(pe / self.ref, 0, 1))
        out[dp:] = self.weight * conf * (1.0 - ncc)
        return out
