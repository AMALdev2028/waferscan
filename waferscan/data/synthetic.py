"""
Synthetic WM-811K-style wafers and a synthetic fab.

Two uses:
1. Tests and demos that must run without any dataset (make_wafer / make_dataset).
2. A fab *simulation* with root causes we planted on purpose
   (simulate_fab): chamber B's focus ring wears out -> Edge-Ring rises;
   CMP pad ages between replacements -> Scratch rises; a few deposition
   temperature excursions -> Center. The root-cause engine is tested by
   checking it recovers exactly these planted causes.
Nothing here is presented as real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from waferscan.classes import CLASSES, INDEX


def _grid(n: int):
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c) / (n / 2)
    th = np.arctan2(yy - c, xx - c)
    return r, th, xx, yy


def make_wafer(cls: str, rng: np.random.Generator, n: int = 52, noise: float | None = None) -> np.ndarray:
    """One die map (n x n, 0 off / 1 pass / 2 fail) of the given class."""
    r, th, xx, yy = _grid(n)
    on = r <= 1.0
    p = np.full((n, n), rng.uniform(0.005, 0.04) if noise is None else noise)   # background fail prob
    a0 = rng.uniform(-np.pi, np.pi)
    dang = np.abs(np.angle(np.exp(1j * (th - a0))))
    if cls == "Center":
        p[r < rng.uniform(0.18, 0.4)] = rng.uniform(0.7, 0.95)
    elif cls == "Donut":
        r0, w = rng.uniform(0.35, 0.6), rng.uniform(0.1, 0.2)
        p[np.abs(r - r0) < w] = rng.uniform(0.6, 0.9)
    elif cls == "Edge-Ring":
        p[(r > rng.uniform(0.82, 0.9)) & (dang < rng.uniform(0.6, 1.0) * np.pi)] = rng.uniform(0.6, 0.95)
    elif cls == "Edge-Loc":
        p[(r > rng.uniform(0.6, 0.8)) & (dang < rng.uniform(0.2, 0.45))] = rng.uniform(0.6, 0.9)
    elif cls == "Loc":
        rc, ac, s = rng.uniform(0.25, 0.6), rng.uniform(-np.pi, np.pi), rng.uniform(0.1, 0.2)
        cx, cy = (n - 1) / 2 * (1 + rc * np.cos(ac)), (n - 1) / 2 * (1 + rc * np.sin(ac))
        p[np.hypot(xx - cx, yy - cy) < s * n / 2] = rng.uniform(0.6, 0.9)
    elif cls == "Random":
        p[:] = rng.uniform(0.15, 0.45)
    elif cls == "Near-full":
        p[:] = rng.uniform(0.8, 0.97)
    elif cls == "Scratch":
        # a thin curved line: random walk with smooth heading changes
        x, y = rng.uniform(0.2, 0.8, 2) * n
        h = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.uniform(0.5, 1.1) * n)):
            h += rng.normal(0, 0.08)
            x, y = x + np.cos(h), y + np.sin(h)
            i, j = int(round(y)), int(round(x))
            if 0 <= i < n and 0 <= j < n:
                p[i, j] = 0.95
    elif cls != "None":
        raise KeyError(cls)
    dm = np.where(on, 1, 0).astype(np.uint8)
    dm[on & (rng.random((n, n)) < p)] = 2
    return dm


def make_dataset(per_class: int = 200, n: int = 52, seed: int = 0, wafers_per_lot: int = 25):
    """Balanced synthetic dataset -> (maps, y, groups); groups = synthetic lot ids."""
    rng = np.random.default_rng(seed)
    maps, y = [], []
    for c in CLASSES:
        for _ in range(per_class):
            maps.append(make_wafer(c, rng, n))
            y.append(INDEX[c])
    y = np.array(y)
    order = rng.permutation(len(y))
    maps = [maps[i] for i in order]
    y = y[order]
    groups = np.arange(len(y)) // wafers_per_lot
    return maps, y, groups


def simulate_fab(n_lots: int = 48, wafers_per_lot: int = 25, seed: int = 7, with_maps: bool = False):
    """A timeline of wafers through 3 etch chambers and one CMP tool.
    Planted causes (ground truth for the root-cause tests):
      * chamber B focus ring wears from wafer ~500: etch edge rate drifts ->
        Edge-Ring probability ramps up only in chamber B (etch_time also creeps).
      * CMP pad cycles climb until the pad is replaced every 400 wafers;
        Scratch probability grows with pad age.
      * rare deposition temperature excursions (+15 C) -> Center.
    Returns a DataFrame (one row per wafer) [+ die maps]."""
    rng = np.random.default_rng(seed)
    rows, maps = [], []
    n = n_lots * wafers_per_lot
    for i in range(n):
        chamber = "ABC"[i % 3]
        pad_cycles = (i % 400) * 2 + int(rng.integers(0, 3))
        wear = max(0.0, (i - 500) / (n - 500)) if chamber == "B" else 0.0
        etch_time = 62.0 + rng.normal(0, 0.8) + 4.0 * wear
        dep_temp = 400.0 + rng.normal(0, 2.0) + (15.0 if rng.random() < 0.03 else 0.0)
        pr_thick = 1200.0 + rng.normal(0, 12.0)
        dose = 1.0e15 * (1 + rng.normal(0, 0.01))
        probs = {"Edge-Ring": 0.02 + 0.55 * wear, "Scratch": 0.01 + 0.30 * (pad_cycles / 800) ** 2,
                 "Center": 0.01 + (0.75 if dep_temp > 410 else 0.0), "Loc": 0.02, "Edge-Loc": 0.02,
                 "Random": 0.01, "Donut": 0.005, "Near-full": 0.002}
        u, acc, cls = rng.random() * max(1.0, sum(probs.values())), 0.0, "None"
        for k, pk in probs.items():
            acc += pk
            if u < acc:
                cls = k
                break
        rows.append({"wafer_id": f"W{i:05d}", "lot_id": f"LOT{i // wafers_per_lot:03d}", "seq": i,
                     "chamber": chamber, "etch_time_s": round(etch_time, 3),
                     "pr_thickness_nm": round(pr_thick, 2), "cmp_pad_cycles": pad_cycles,
                     "dep_temp_c": round(dep_temp, 2), "implant_dose": dose, "true_class": cls})
        if with_maps:
            maps.append(make_wafer(cls, rng))
    df = pd.DataFrame(rows)
    return (df, maps) if with_maps else df


def synthetic_optical_scan(dm: np.ndarray, die: int = 24, tilt: float = 0.8, seed: int = 0) -> np.ndarray:
    """A full-wafer 'photo' of a die map, for tests and benchmarks: textured dies,
    dark scribe lines, a bright blob in every failed die, sensor noise, and an
    off-axis camera (tilt < 1 squashes the wafer into an ellipse)."""
    import cv2
    H, W = dm.shape
    tile = np.full((die, die), 150, np.uint8)
    tile[: max(2, die // 12), :] = 60
    tile[:, : max(2, die // 12)] = 60                               # scribe lines
    tile[die // 3:2 * die // 3, die * 5 // 12:die * 13 // 24] = 200  # die circuitry (golden die has structure)
    bad = tile.copy()
    bad[die // 5:die // 2, die // 6:die * 11 // 24] = 250            # the defect
    img = np.full((H * die + 160, W * die + 160), 15, np.uint8)
    for (r, c), v in np.ndenumerate(dm):
        if v:
            img[80 + r * die:80 + (r + 1) * die, 80 + c * die:80 + (c + 1) * die] = bad if v == 2 else tile
    img = np.clip(img + np.random.default_rng(seed).normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return cv2.warpAffine(img, np.float32([[1, 0, 0], [0, tilt, 0]]), (img.shape[1], int(img.shape[0] * tilt)))
