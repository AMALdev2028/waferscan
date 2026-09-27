"""
Handcrafted, physically meaningful wafer features (in the spirit of Wu et
al., IEEE TSM 2015, the paper that released WM-811K: density per region,
Radon-projection and geometry of the most salient defect region).

Used three ways:
1. a gradient-boosting base model that is diverse from the CNN (stacking),
2. the secondary *verification* stage (is "Edge-Ring" really a ring?),
3. plain-English explanations in API responses.

Input is the 2-channel model input (wafer mask, fail mask) at 64x64, so the
features are comparable across wafer products with different die counts.
"""
from __future__ import annotations

import cv2
import numpy as np

from waferscan.data.preprocess import to_polar

N_RADIAL = 8

# human-readable names for explanations
NAMES = {
    "fail_frac": "share of dies failed",
    "center_contrast": "centre fail density minus the rest",
    "donut_contrast": "mid-radius fail density minus centre/edge",
    "edge_contrast": "outer-ring fail density minus inner wafer",
    "edge_coverage": "fraction of the wafer edge (by angle) that is failing",
    "edge_max_sector": "worst edge sector fail density",
    "salient_area": "largest defect cluster (share of wafer)",
    "salient_r": "largest cluster's distance from centre (0 centre, 1 edge)",
    "salient_elong": "largest cluster elongation (length/width)",
    "salient_solidity": "largest cluster solidity (1 = convex blob)",
    "radon_peak": "straight-line strength (Radon projection peak)",
    "moran_i": "spatial clustering of fails (Moran's I)",
    "clustered_frac": "share of failed dies inside large clusters",
    "n_clusters": "number of large clusters",
}


def _moran_i(fail: np.ndarray, on: np.ndarray) -> float:
    """Moran's I with 4-neighbour weights: ~0 for random speckle, -> 1 when
    fails cluster together."""
    x = fail[on]
    if x.size < 10 or x.std() < 1e-6:
        return 0.0
    z = np.zeros_like(fail, dtype=np.float64)
    z[on] = x - x.mean()
    num, w = 0.0, 0
    for dy, dx in ((0, 1), (1, 0)):
        a = z[: z.shape[0] - dy, : z.shape[1] - dx]
        b = z[dy:, dx:]
        m = on[: on.shape[0] - dy, : on.shape[1] - dx] & on[dy:, dx:]
        num += 2 * (a[m] * b[m]).sum()
        w += 2 * m.sum()
    return float((x.size / w) * num / (z[on] ** 2).sum())


def _radon_peak(fail: np.ndarray) -> float:
    """Max over 12 angles of (max column sum / mean column sum) - a straight
    scratch concentrates into one sharp column at its angle."""
    s = fail.shape[0]
    best = 0.0
    for ang in range(0, 180, 15):
        M = cv2.getRotationMatrix2D(((s - 1) / 2, (s - 1) / 2), ang, 1.0)
        col = cv2.warpAffine(fail, M, (s, s), flags=cv2.INTER_LINEAR).sum(0)
        best = max(best, float(col.max() / (col.mean() + 1e-6)))
    return best


def features(x: np.ndarray) -> dict:
    """x: (2, S, S) model input -> dict of scalar features."""
    wafer, fail = x[0].astype(np.float32), x[1].astype(np.float32)
    on = wafer > 0.5
    s = x.shape[-1]
    c = (s - 1) / 2
    yy, xx = np.mgrid[:s, :s]
    r = np.hypot(xx - c, yy - c) / (s / 2)
    area = max(on.sum(), 1)
    f = {"fail_frac": float(fail[on].sum() / area)}

    def dens(m):
        m = m & on
        return float(fail[m].mean()) if m.any() else 0.0

    radial = [dens((r >= i / N_RADIAL) & (r < (i + 1) / N_RADIAL)) for i in range(N_RADIAL)]
    f.update({f"radial_{i}": v for i, v in enumerate(radial)})
    inner, mid, outer = dens(r < 0.3), dens((r >= 0.3) & (r < 0.75)), dens(r >= 0.8)
    f["center_contrast"] = inner - dens(r >= 0.3)
    f["donut_contrast"] = mid - max(inner, outer)
    f["edge_contrast"] = outer - dens(r < 0.8)

    # 13 regions: centre + 4 middle-ring sectors + 8 outer-ring sectors (sorted -> rotation invariant)
    th = np.arctan2(yy - c, xx - c)
    mids = sorted(dens((r >= 1 / 3) & (r < 2 / 3) & (np.floor((th + np.pi) / (np.pi / 2)) % 4 == k)) for k in range(4))
    outs = sorted(dens((r >= 2 / 3) & (np.floor((th + np.pi) / (np.pi / 4)) % 8 == k)) for k in range(8))
    f["region_center"] = dens(r < 1 / 3)
    f.update({f"region_mid_{k}": v for k, v in enumerate(mids)})
    f.update({f"region_out_{k}": v for k, v in enumerate(outs)})

    polar = to_polar(fail, wafer, n_r=16, n_theta=64)
    edge = polar[12:].mean(0)                                  # r >= 0.75, per angle
    thr = max(0.3, 2 * f["fail_frac"])
    f["edge_coverage"] = float((edge > thr).mean())
    f["edge_max_sector"] = float(np.convolve(np.r_[edge, edge[:7]], np.ones(8) / 8, "valid").max())

    # connected clusters of failed dies (8-connected)
    bin_ = ((fail > 0.5) & on).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(8, 0.004 * area)]
    fails = max(bin_.sum(), 1)
    f["clustered_frac"] = float(sum(stats[i, cv2.CC_STAT_AREA] for i in big) / fails)
    f["n_clusters"] = float(len(big))
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pts = np.column_stack(np.nonzero(lab == i)).astype(np.float32)
        f["salient_area"] = float(len(pts) / area)
        f["salient_r"] = float(np.hypot(cent[i][0] - c, cent[i][1] - c) / (s / 2))
        if len(pts) >= 3:
            ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))
            f["salient_elong"] = float(np.sqrt((ev[1] + 1e-3) / (ev[0] + 1e-3)))
            hull = cv2.convexHull(pts[:, ::-1].reshape(-1, 1, 2))
            f["salient_solidity"] = float(len(pts) / max(cv2.contourArea(hull), 1.0))
        else:
            f["salient_elong"], f["salient_solidity"] = 1.0, 1.0
    else:
        f.update(salient_area=0.0, salient_r=0.0, salient_elong=1.0, salient_solidity=1.0)
    f["radon_peak"] = _radon_peak(fail * on)
    f["moran_i"] = _moran_i(fail, on)
    return f


FEATURE_NAMES = list(features(np.zeros((2, 64, 64), np.float32) + np.eye(64, dtype=np.float32)[None]).keys())


def feature_matrix(X: np.ndarray) -> np.ndarray:
    return np.array([[fd[k] for k in FEATURE_NAMES] for fd in map(features, X)], dtype=np.float32)
