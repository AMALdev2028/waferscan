"""
Die map -> model input, plus the augmentations used in training.

Model input = 2 channels at S x S (default 64):
  ch0 = on-wafer mask (where dies exist), ch1 = failed-die mask.
Die-grid registration: crop to the wafer's bounding box, pad to a square
(keeps the die aspect 1:1), then resize with INTER_AREA, which averages when
shrinking and behaves like nearest-neighbour when enlarging - so a 26x26 and
a 200x200 wafer both become comparable 64x64 inputs.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_wafer(dm: np.ndarray) -> np.ndarray:
    """Crop a die map to the bounding box of on-wafer dies, pad to square."""
    ys, xs = np.nonzero(dm)
    if len(ys) == 0:
        raise ValueError("die map has no on-wafer dies")
    dm = dm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = dm.shape
    s = max(h, w)
    out = np.zeros((s, s), dm.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = dm
    return out


def to_input(dm: np.ndarray, size: int = 64) -> np.ndarray:
    """(H, W) die map with 0/1/2 -> (2, size, size) float32 in [0, 1]."""
    sq = crop_wafer(np.asarray(dm, np.uint8))
    chans = [(sq > 0).astype(np.float32), (sq == 2).astype(np.float32)]
    return np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) for c in chans])


def input_to_native(heat: np.ndarray, dm: np.ndarray) -> np.ndarray:
    """Map a (S, S) map in model-input space back onto the native die grid
    of `dm` (inverse of crop -> pad -> resize in to_input)."""
    ys, xs = np.nonzero(dm)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    s = max(h, w)
    sq = cv2.resize(heat.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
    out = np.zeros(dm.shape, np.float32)
    out[y0:y1, x0:x1] = sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w]
    return out * (dm > 0)


def rotate(x: np.ndarray, angle_deg: float, flip: bool = False) -> np.ndarray:
    """Rotate a (C, S, S) input about the wafer centre. All 9 WM-811K classes
    are defined by shape relative to the wafer centre/edge, never by absolute
    angle, so any rotation or mirror of a wafer keeps its label. (In polar
    coordinates a rotation is just a circular shift along the angle axis -
    that is the 'rotational invariance' being exploited.)"""
    s = x.shape[-1]
    M = cv2.getRotationMatrix2D(((s - 1) / 2, (s - 1) / 2), angle_deg, 1.0)
    out = np.stack([cv2.warpAffine(c, M, (s, s), flags=cv2.INTER_LINEAR, borderValue=0) for c in x])
    return out[:, :, ::-1].copy() if flip else out


def to_polar(fail: np.ndarray, wafer: np.ndarray, n_r: int = 16, n_theta: int = 64) -> np.ndarray:
    """Fail density on a (radius x angle) grid. Rotating the wafer = rolling
    this array along axis 1. Used by the morphology features."""
    s = fail.shape[-1]
    c = (s - 1) / 2
    yy, xx = np.mgrid[:s, :s]
    r = np.hypot(xx - c, yy - c) / (s / 2)
    th = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    ri = np.clip((r * n_r).astype(int), 0, n_r - 1)
    ti = np.clip((th * n_theta).astype(int), 0, n_theta - 1)
    on = (wafer > 0.5) & (r < 1.0)
    num = np.zeros((n_r, n_theta))
    den = np.zeros((n_r, n_theta))
    np.add.at(num, (ri[on], ti[on]), fail[on])
    np.add.at(den, (ri[on], ti[on]), 1.0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
