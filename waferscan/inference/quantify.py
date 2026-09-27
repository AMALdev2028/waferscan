"""
Affected-area quantification on the die grid.

Units: a die map knows dies, not millimetres. We convert with the wafer
diameter (default 300 mm) minus edge exclusion, spread over the grid span:
die pitch = usable diameter / dies across. Pass real die dimensions when you
know them (die_w_mm / die_h_mm) - then areas are exact.

Uncertainty, honestly labelled:
- wilson95: 95% Wilson score interval for a proportion of dies. It treats
  dies as independent trials - fine for "what is this wafer's fail rate as
  an estimate of the process rate", optimistic for spatially clustered fails.
- segmentation_range: the pattern's area is a *count*; its uncertainty comes
  from where you draw the pattern boundary. We re-segment with looser/stricter
  settings and report the min-max.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from waferscan.data.images import FAIL, OFF

WHOLE_WAFER = {"Random", "Near-full"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _neighbors(fail: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(fail.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)


def segment_pattern(dm: np.ndarray, cls: str, cam: np.ndarray | None = None,
                    min_neighbors: int = 2, cam_frac: float = 0.5, min_size: int = 3) -> np.ndarray:
    """Boolean mask of failed dies that belong to the classified pattern.
    Isolated random fails (fewer than `min_neighbors` failing neighbours) are
    background noise, not the pattern. If a CAM heatmap is given, keep only the
    clusters the network actually looked at."""
    fail = dm == FAIL
    if cls == "None":
        return np.zeros_like(fail)
    if cls in WHOLE_WAFER:
        return fail.copy()
    core = fail & (_neighbors(fail) >= min_neighbors)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_size]
    if cam is not None and comps and cam.max() > 0:
        hot = cam >= cam_frac * cam.max()
        focused = [i for i in comps if (hot & (lab == i)).any()]
        comps = focused or comps
    if not comps:
        return np.zeros_like(fail)
    return np.isin(lab, comps)


def _moran(fail: np.ndarray, on: np.ndarray) -> float:
    x = fail[on].astype(np.float64)
    if x.size < 10 or x.std() == 0:
        return 0.0
    z = np.zeros(fail.shape)
    z[on] = x - x.mean()
    num = w = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        a, b = z[: z.shape[0] - dy, : z.shape[1] - dx], z[dy:, dx:]
        m = on[: on.shape[0] - dy, : on.shape[1] - dx] & on[dy:, dx:]
        num += 2 * (a[m] * b[m]).sum()
        w += 2 * m.sum()
    return float(x.size / w * num / (z[on] ** 2).sum())


def quantify(dm: np.ndarray, cls: str, cam: np.ndarray | None = None, wafer_diameter_mm: float = 300.0,
             edge_exclusion_mm: float = 3.0, die_w_mm: float | None = None, die_h_mm: float | None = None,
             die_affected: np.ndarray | None = None, max_die_list: int = 3000) -> dict:
    """dm: native die map (H, W) 0/1/2 (cropped or not). cam: same shape, >= 0."""
    dm = np.asarray(dm, np.uint8)
    on, fail = dm != OFF, dm == FAIL
    H, W = dm.shape
    ys, xs = np.nonzero(on)
    span_r, span_c = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    usable = wafer_diameter_mm - 2 * edge_exclusion_mm
    dw = die_w_mm or usable / span_c
    dh = die_h_mm or usable / span_r
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    radius_dies = max(span_r, span_c) / 2

    def to_mm(r, c):
        return [round(float((c - cx) * dw), 3), round(float((cy - r) * dh), 3)]      # x right, y up

    total, nfail = int(on.sum()), int(fail.sum())
    pattern = segment_pattern(dm, cls, cam)
    npat = int(pattern.sum())
    variants = [int(segment_pattern(dm, cls, cam, mn, cf).sum()) for mn in (1, 2, 3) for cf in (0.3, 0.5, 0.7)]
    die_area = dw * dh

    out = {
        "units": {"die_w_mm": round(dw, 4), "die_h_mm": round(dh, 4),
                  "die_size_source": "given" if die_w_mm else "estimated from grid span",
                  "wafer_diameter_mm": wafer_diameter_mm, "edge_exclusion_mm": edge_exclusion_mm},
        "wafer": {"dies_total": total, "dies_failed": nfail, "dies_passed": total - nfail,
                  "fail_pct": round(100 * nfail / max(total, 1), 3),
                  "clean_pct": round(100 * (total - nfail) / max(total, 1), 3),
                  "fail_pct_wilson95": [round(100 * v, 3) for v in wilson(nfail, total)]},
        "pattern": {"dies": npat,
                    "affected_pct": round(100 * npat / max(total, 1), 3),
                    "clean_pct": round(100 - 100 * npat / max(total, 1), 3),
                    "affected_pct_wilson95": [round(100 * v, 3) for v in wilson(npat, total)],
                    "affected_pct_segmentation_range": [round(100 * min(variants) / max(total, 1), 3),
                                                        round(100 * max(variants) / max(total, 1), 3)],
                    "area_mm2": round(npat * die_area, 2),
                    "area_mm2_segmentation_range": [round(min(variants) * die_area, 2),
                                                    round(max(variants) * die_area, 2)],
                    "background_fail_dies": int((fail & ~pattern).sum())},
    }
    if npat:
        pr, pc = np.nonzero(pattern)
        r_mean, c_mean = pr.mean(), pc.mean()
        rad = math.hypot(c_mean - cx, r_mean - cy) / radius_dies
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pattern.astype(np.uint8), connectivity=8)
        regions = []
        for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:20]:
            comp = (lab == i).astype(np.uint8)
            # contours on a 2x upsampled mask trace boundary-pixel centres (k/2 - 0.25 in die
            # units); snapping to the nearest half-integer puts vertices exactly on die corners
            big = cv2.resize(comp, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
            poly = np.floor(pts / 2.0 - 0.25) + 0.5
            rr, cc = np.nonzero(comp)
            regions.append({"dies": int(comp.sum()), "area_mm2": round(float(comp.sum() * die_area), 2),
                            "centroid_die": [round(float(rr.mean()), 2), round(float(cc.mean()), 2)],
                            "centroid_mm": to_mm(rr.mean(), cc.mean()),
                            "polygon_die": [[round(float(y), 2), round(float(x), 2)] for x, y in poly],
                            "polygon_mm": [to_mm(y, x) for x, y in poly]})
        out["pattern"].update({
            "centroid_die": [round(float(r_mean), 2), round(float(c_mean), 2)],
            "centroid_mm": to_mm(r_mean, c_mean),
            "centroid_radius_norm": round(float(rad), 4),
            "centroid_angle_deg": round(float(math.degrees(math.atan2(cy - r_mean, c_mean - cx))) % 360, 1),
            "regions": regions,
        })

    # radial / angular distribution of fails (all failed dies)
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(xx - cx, yy - cy) / radius_dies
    th = (np.degrees(np.arctan2(cy - yy, xx - cx)) + 360) % 360
    radial = []
    for i in range(10):
        m = on & (r >= i / 10) & (r < (i + 1) / 10 if i < 9 else r <= 10)
        radial.append({"r": [i / 10, (i + 1) / 10], "dies": int(m.sum()),
                       "fail_density": round(float(fail[m].mean()) if m.any() else 0.0, 4)})
    angular = []
    for i in range(12):
        m = on & (th >= 30 * i) & (th < 30 * (i + 1))
        angular.append({"deg": [30 * i, 30 * (i + 1)], "fail_density": round(float(fail[m].mean()) if m.any() else 0.0, 4)})
    n, _, stats, _ = cv2.connectedComponentsWithStats(fail.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    out["distribution"] = {"radial": radial, "angular": angular}
    out["clustering"] = {
        "morans_i": round(_moran(fail, on), 4),
        "clusters_ge3": int((sizes >= 3).sum()),
        "largest_cluster_dies": int(sizes.max()),
        "clustered_fail_frac": round(float(sizes[sizes >= 3].sum() / max(nfail, 1)), 4),
        "mean_fail_neighbors": round(float(_neighbors(fail)[fail].mean()) if nfail else 0.0, 3),
    }
    heat = cv2.GaussianBlur(fail.astype(np.float32), (0, 0), 1.5) * on
    out["heatmap"] = np.round(heat / (heat.max() + 1e-9), 3).tolist()      # spatial fail-density gradient

    fr, fc = np.nonzero(fail)
    dies = [{"row": int(a), "col": int(b), "xy_mm": to_mm(a, b), "in_pattern": bool(pattern[a, b])}
            for a, b in zip(fr[:max_die_list], fc[:max_die_list])]
    if die_affected is not None:
        for d_ in dies:
            d_["affected_pct"] = round(100 * float(die_affected[d_["row"], d_["col"]]), 3)
    out["failed_dies"] = dies
    out["failed_dies_truncated"] = bool(len(fr) > max_die_list)
    return out


def quantify_batch(maps: list[np.ndarray], classes: list[str], results: list[dict], grid: int = 64) -> dict:
    """Pool per-wafer quantifications into batch metrics + a batch heatmap."""
    from waferscan.data.preprocess import to_input
    tot = sum(r["wafer"]["dies_total"] for r in results)
    fl = sum(r["wafer"]["dies_failed"] for r in results)
    pat = sum(r["pattern"]["dies"] for r in results)
    per = np.array([r["wafer"]["fail_pct"] for r in results])
    heat = np.mean([to_input(m, grid)[1] for m in maps], 0)
    half = 1.96 * per.std(ddof=1) / math.sqrt(len(per)) if len(per) > 1 else 0.0
    return {
        "wafers": len(results),
        "dies_total": tot, "dies_failed": fl,
        "fail_pct_pooled": round(100 * fl / max(tot, 1), 3),
        "fail_pct_pooled_wilson95": [round(100 * v, 3) for v in wilson(fl, tot)],
        "fail_pct_wafer_mean": round(float(per.mean()), 3),
        "fail_pct_wafer_mean_ci95": [round(float(per.mean() - half), 3), round(float(per.mean() + half), 3)],
        "pattern_affected_pct_pooled": round(100 * pat / max(tot, 1), 3),
        "class_counts": {c: classes.count(c) for c in sorted(set(classes))},
        "heatmap": np.round(heat, 3).tolist(),
        "radial_profile": [round(float(np.mean([r["distribution"]["radial"][i]["fail_density"] for r in results])), 4)
                           for i in range(10)],
    }
