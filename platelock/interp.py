"""Weights-free optical-flow frame interpolation (OpenCV DIS).

A fallback so the core runs anywhere; the ComfyUI nodes can swap in RIFE/FILM.
"""
from __future__ import annotations

import cv2
import numpy as np

_DIS = None


def _flow(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        ga = cv2.resize(ga, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    f = _DIS.calc(ga, gb, None)
    if scale != 1.0:
        f = cv2.resize(f, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) / scale
    return f


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    scale = min(1.0, 960.0 / a.shape[1])
    f01 = _flow(a, b, scale)
    f10 = _flow(b, a, scale)
    # Super SloMo intermediate-flow approximation.
    ft0 = -(1 - t) * t * f01 + t * t * f10
    ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
    w0 = _warp(a.astype(np.float32), ft0)
    w1 = _warp(b.astype(np.float32), ft1)
    return np.clip((1 - t) * w0 + t * w1, 0, 255).astype(np.uint8)
