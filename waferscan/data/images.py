"""
Turning pictures into die maps.

A WM-811K wafer map is a small 2-D grid: 0 = no die (off-wafer), 1 = die passed
test, 2 = die failed. Everything downstream (the CNN, the morphology checks,
the mm^2 area maths) works on that grid. This module recovers the grid from
three kinds of pictures:

1. rendered_to_diemap: a colour-rendered wafer map (the Roboflow JPGs in the
   wafer_defect repo: purple / teal / yellow viridis colours, resized to
   640x640). We find the die pitch from where colours change, then read the
   colour at every die centre.
2. optical_to_diemap: a photo/scan of a real wafer. Classic die-to-die
   inspection: find the die grid from the scribe lines, cut the wafer into
   die tiles, build a "golden die" (pixel-wise median of all tiles) and mark
   a die as failed when enough of its pixels differ from the golden die.
   Also returns a per-die "% of pixels affected".
3. sem_segment: an SEM review image of one defect site. No die grid here;
   we segment the defect blob and measure it in um^2.
"""
from __future__ import annotations

import cv2
import numpy as np

# viridis(0.0), viridis(0.5), viridis(1.0) -> WM-811K codes 0, 1, 2
PALETTE = np.array([[68, 1, 84], [33, 145, 140], [253, 231, 37]], dtype=np.float32)
OFF, PASS, FAIL = 0, 1, 2

# matplotlib's viridis colormap, 256 RGB entries (embedded so the API needs no matplotlib)
_VIRIDIS_HEX = (
    "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164471365481467481668481769"
    "48186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a"
    "472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85"
    "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b"
    "3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d"
    "33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e"
    "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e"
    "26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d"
    "21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88"
    "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad81"
    "28ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc74"
    "3fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863"
    "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d"
    "84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32"
    "addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a"
    "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"
)
_VIRIDIS = np.frombuffer(bytes.fromhex(_VIRIDIS_HEX), np.uint8).reshape(256, 3).astype(np.float32)
_LUT = None


def _viridis_lut() -> np.ndarray:
    """32x32x32 lookup: quantised RGB -> nearest viridis position (0..255)."""
    global _LUT
    if _LUT is None:
        q = (np.arange(32) * 8 + 4).astype(np.float32)
        grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
        _LUT = np.concatenate([((grid[i:i + 4096, None] - _VIRIDIS[None]) ** 2).sum(-1).argmin(1)
                               for i in range(0, len(grid), 4096)]).astype(np.uint8).reshape(32, 32, 32)
    return _LUT


def rgb_to_labels(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Pixel colours -> die states 0/1/2.
    Most renders use viridis(0 / 0.5 / 1) for off / pass / fail. Some use a
    different normalisation (e.g. levels at 0.33 / 0.67), so we find the
    colour levels actually present: if they match the standard palette we
    map absolutely, otherwise lowest level = off, next = pass, higher = fail."""
    lut = _viridis_lut()
    q = (rgb // 8).astype(np.intp)
    v = lut[q[..., 0], q[..., 1], q[..., 2]].astype(np.int16)
    hist = np.bincount(v.ravel(), minlength=256)
    # Decide on where the *mass* sits: resizing blends neighbouring colours into
    # small in-between peaks, so counting peaks is fragile; mass is not.
    near_std = sum(hist[max(0, s - 24):s + 25].sum() for s in (0, 128, 255)) / v.size
    standard = near_std >= 0.9
    if standard:
        centers, codes = np.array([0, 128, 255]), np.array([OFF, PASS, FAIL])
        levels = [0, 128, 255]
    else:
        levels = []
        for i in np.argsort(hist)[::-1]:                # peaks holding >= 0.4% of pixels
            if hist[i] < 0.004 * v.size:
                break
            if all(abs(int(i) - l) > 24 for l in levels):
                levels.append(int(i))
        levels.sort()
        centers = np.array(levels)
        if len(levels) == 2:                            # off + one on-wafer colour: dark half = pass
            codes = np.array([OFF, PASS if levels[1] < 128 else FAIL])
        else:
            codes = np.array([OFF, PASS] + [FAIL] * max(0, len(levels) - 2))[: len(levels)]
    nearest = np.abs(v[..., None] - centers[None, None, :]).argmin(-1)
    return codes[nearest].astype(np.uint8), ("standard" if standard else f"levels{levels}")


def _grid_count(labels: np.ndarray, axis: int, lo: int = 12, hi: int = 160) -> int:
    """How many dies fit along `axis`. Die boundaries sit on multiples of the
    pitch p, so exp(2*pi*i*x/p) lines up (|mean| ~ 1) only for the right p.
    Multiples 2n, 3n also line up, so we take the smallest n near the max."""
    size = labels.shape[axis]
    step = max(1, labels.shape[1 - axis] // 160)          # ~160 scan lines is plenty
    lines = labels[::step, :] if axis == 1 else labels[:, ::step]
    pos = np.nonzero(np.diff(lines.astype(np.int16), axis=axis))[axis] + 1.0
    if pos.size < 20:
        raise ValueError("too few colour boundaries to find the die grid")
    ns = np.arange(lo, hi + 1, dtype=np.float64)
    scores = np.abs(np.exp(2j * np.pi * np.outer(ns, pos) / size).mean(1))
    return int(ns[np.argmax(scores >= 0.8 * scores.max())])


def labels_to_diemap(labels: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Majority label in a small window around each die centre."""
    h, w = labels.shape
    k = max(1, int(0.4 * min(h / n_rows, w / n_cols)))
    votes = np.stack([cv2.blur((labels == c).astype(np.float32), (k, k)) for c in (OFF, PASS, FAIL)])
    ys = ((np.arange(n_rows) + 0.5) * h / n_rows).astype(int)
    xs = ((np.arange(n_cols) + 0.5) * w / n_cols).astype(int)
    return votes[:, ys][:, :, xs].argmax(0).astype(np.uint8)


def rendered_to_diemap(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Colour-rendered wafer map image -> (die map, info).
    info['agreement'] = fraction of pixels the recovered grid re-renders
    correctly; well below ~0.9 means the grid estimate is wrong."""
    labels, palette = rgb_to_labels(rgb)
    n_rows, n_cols = _grid_count(labels, 0), _grid_count(labels, 1)
    dm = labels_to_diemap(labels, n_rows, n_cols)
    h, w = labels.shape
    back = dm[(np.arange(h) * n_rows // h)[:, None], (np.arange(w) * n_cols // w)[None, :]]
    return dm, {"n_rows": n_rows, "n_cols": n_cols, "palette": palette,
                "agreement": float((back == labels).mean())}


def looks_rendered(rgb: np.ndarray) -> bool:
    """True if >90% of pixels are (JPEG-close to) a viridis colour - i.e. the
    picture is a colour-rendered die map rather than a photo."""
    lut = _viridis_lut()
    px = rgb.reshape(-1, 3)[:: max(1, rgb.size // 3 // 20000)]
    idx = lut[px[:, 0] // 8, px[:, 1] // 8, px[:, 2] // 8]
    d = np.sqrt(((px.astype(np.float32) - _VIRIDIS[idx]) ** 2).sum(-1))
    return bool((d < 40).mean() > 0.9)


# --------------------------------------------------------------------------
# Optical scans (full-wafer photos / scanner images)
# --------------------------------------------------------------------------
def normalize_wafer_image(gray: np.ndarray, out: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE contrast normalisation + wafer localisation + distortion fix.
    A wafer photographed off-axis appears as an ellipse. We fit that ellipse
    and apply a pure stretch along its two axes (no rotation, so die rows stay
    horizontal) that turns it back into a circle filling an out x out image.
    Returns (image, circular wafer mask)."""
    # localise on a <=1024 px copy (fast at 4K); the ellipse is scaled back up
    s = min(1.0, 1024.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # bridge dark scribe lines
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or len(max(cnts, key=cv2.contourArea)) < 5:
        raise ValueError("no wafer outline found in image")
    (ex, ey), (ea, eb), eang = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
    rect = ((ex / s, ey / s), (ea / s, eb / s), eang)
    (cx, cy) = rect[0]
    if out is None:        # keep native resolution (small defects survive), within 1024..4096 px
        out = int(np.clip(max(rect[1]), 1024, 4096))
    box = cv2.boxPoints(rect)
    c = np.array([cx, cy])
    u, v = (box[0] + box[1]) / 2 - c, (box[1] + box[2]) / 2 - c   # the two semi-axes
    o = out / 2.0
    A = np.column_stack([u / np.linalg.norm(u), v / np.linalg.norm(v)])
    L = A @ np.diag([o / np.linalg.norm(u), o / np.linalg.norm(v)]) @ np.linalg.inv(A)
    M = np.hstack([L, (np.array([o, o]) - L @ c)[:, None]])
    # CLAHE is only used to *find* the wafer; the returned pixels are the raw
    # intensities, because CLAHE's local remapping differs near the wafer edge
    # and would make edge dies look different from the golden die
    warped = cv2.warpAffine(gray, M, (out, out), flags=cv2.INTER_AREA)
    yy, xx = np.mgrid[:out, :out]
    circle = ((xx + 0.5 - o) ** 2 + (yy + 0.5 - o) ** 2) <= o * o
    # on-wafer = bright die area, not just "inside the fitted circle": the dies form
    # a staircase, so circle corners near the edge hold no die. (Scribe lines are dark
    # too; callers test die *interiors*, so they don't need bridging.)
    t, _ = cv2.threshold(cv2.resize(warped, (512, 512), interpolation=cv2.INTER_AREA), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return warped, circle & (cv2.GaussianBlur(warped, (0, 0), 1) > t)


def _pitch_phase(img: np.ndarray, mask: np.ndarray, axis: int, lo: int = 6) -> tuple[float, float]:
    """Die pitch + grid offset along one axis, from the scribe lines.
    Every scribe line produces strong edges, so the edge-strength profile is
    a pulse train: its autocorrelation peaks at the pitch, and the phase of
    its Fourier coefficient at 1/pitch gives where the scribe lines sit."""
    grad = np.abs(cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1 - axis, axis, ksize=3)) * mask
    prof = grad.sum(axis=axis)
    prof = prof - prof.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(prof, 2 * len(prof))) ** 2)[: len(prof) // 3]
    ac /= ac[0] + 1e-9
    peaks = [i for i in range(lo, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
    best = max((ac[i] for i in peaks), default=0.0)
    if best <= 0:   # no peaks, or only negative correlation (e.g. a plain gradient): nothing periodic
        raise ValueError("no periodic die grid found")
    p0 = float(next(i for i in peaks if ac[i] >= 0.5 * best))
    # refine to sub-pixel: a 0.5 px pitch error drifts 10 px over 20 dies
    x = np.arange(len(prof))
    cands = np.arange(p0 - 1.5, p0 + 1.5, 0.01)
    pitch = float(cands[np.argmax(np.abs(np.exp(-2j * np.pi * np.outer(1 / cands, x)) @ prof))])
    # scribe lines at x0 + k*pitch  ->  sum prof*exp(-2*pi*i*x/pitch) has angle -2*pi*x0/pitch
    phase = np.angle((prof * np.exp(-2j * np.pi * x / pitch)).sum())
    offset = (-phase / (2 * np.pi) * pitch) % pitch
    return pitch, offset


def optical_to_diemap(gray: np.ndarray, k_mad: float = 6.0, min_affected: float = 0.05,
                      pitch_px: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Die-to-die inspection on a full-wafer optical scan.
    min_affected: share of a die's interior pixels that must deviate (robust
    z > k_mad vs. its neighbours) to call the die failed. Calibration knob:
    lower it to catch smaller defects once your scanner's noise floor is known.
    Returns (die map 0/1/2, per-die affected fraction in [0,1], info)."""
    img, mask = normalize_wafer_image(gray)
    (py, oy), (px, ox) = _pitch_phase(img, mask, 0), _pitch_phase(img, mask, 1)
    if pitch_px:
        py = px = float(pitch_px)
    # die-grid registration: resample so every die starts exactly at k*P (P integer).
    # Cutting tiles at rounded positions of a fractional pitch would shift some
    # tiles by a pixel and make the scribe line itself look like a defect.
    P = int(round(max(py, px)))
    ny, nx = int((img.shape[0] - oy) // py), int((img.shape[1] - ox) // px)
    M = np.float32([[P / px, 0, -ox * P / px], [0, P / py, -oy * P / py]])
    reg = cv2.warpAffine(img, M, (nx * P, ny * P), flags=cv2.INTER_LINEAR).astype(np.float32)
    regm = cv2.warpAffine(mask.astype(np.uint8), M, (nx * P, ny * P), flags=cv2.INTER_NEAREST)
    m = max(1, P // 8)                                              # die interiors, not the streets
    tiles = reg.reshape(ny, P, nx, P).transpose(0, 2, 1, 3)[:, :, m:P - m, m:P - m]
    inside = regm.reshape(ny, P, nx, P).transpose(0, 2, 1, 3)[:, :, m:P - m, m:P - m].mean((2, 3)) > 0.98
    if inside.sum() < 4:
        raise ValueError("die grid too coarse for this wafer")
    # per-die brightness offset removal: cancels uneven illumination / vignetting.
    # (Offset only - dividing by each die's own spread amplifies noise on clean,
    # flat dies and turns it into false defects; one global scale is used below.)
    tiles = tiles - np.median(tiles, axis=(2, 3), keepdims=True)
    # reference ("golden") die per position = median of its 8 neighbours (how
    # inspection tools arbitrate): local references absorb slow drifts across the
    # wafer (focus, residual pitch error) that one global golden die would report
    # as defects at the wafer edge. Missing neighbours (wafer edge) are filled with
    # the global golden die. Vectorised one die-row at a time (4K = thousands of dies).
    ny, nx, th, tw = tiles.shape
    global_golden = np.median(tiles[inside], axis=0)
    pad = np.broadcast_to(global_golden, (ny + 2, nx + 2, th, tw)).copy()
    pad[1:-1, 1:-1][inside] = tiles[inside]
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    diff = np.zeros_like(tiles)
    for r in range(ny):
        if inside[r].any():
            nb = np.stack([pad[r + 1 + dr, 1 + dc:1 + dc + nx] for dr, dc in offsets])   # (8, nx, th, tw)
            diff[r] = np.abs(tiles[r] - np.median(nb, axis=0))
    mad = float(np.median(diff[inside])) + 1e-6
    defect_px = diff > k_mad * 1.4826 * mad                         # robust z-score above k
    affected = np.where(inside, defect_px.mean((2, 3)), 0.0)
    dm = np.where(inside, np.where(affected >= min_affected, FAIL, PASS), OFF).astype(np.uint8)
    return dm, affected.astype(np.float32), {"pitch_px": [py, px], "grid": list(dm.shape)}


# --------------------------------------------------------------------------
# SEM review images (single defect site)
# --------------------------------------------------------------------------
def sem_segment(gray: np.ndarray, nm_per_px: float = 5.0) -> dict:
    """Segment the defect in an SEM review image and measure it.
    Background = heavy median blur; the defect is what deviates from it.
    Shape descriptors give a *hint* (particle / scratch / residue) - this is a
    geometric heuristic, not a trained SEM classifier (no SEM training data)."""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bg = cv2.medianBlur(g, 2 * (min(g.shape) // 8) + 1)
    dev = cv2.absdiff(g, bg)
    _, th = cv2.threshold(dev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted((c for c in cnts if cv2.contourArea(c) >= 20), key=cv2.contourArea, reverse=True)[:10]
    um2 = (nm_per_px / 1000.0) ** 2
    out = []
    for c in cnts:
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per + 1e-9)
        w_, h_ = cv2.minAreaRect(c)[1]
        elong = max(w_, h_) / (min(w_, h_) + 1e-9)
        m = cv2.moments(c)
        out.append({
            "area_um2": round(area * um2, 5),
            "centroid_px": [round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)],
            "polygon_px": cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist(),
            "circularity": round(float(circ), 3),
            "elongation": round(float(elong), 2),
            "shape_hint": "scratch" if elong > 4 else ("particle" if circ > 0.7 else "residue/irregular"),
        })
    return {"defects": out, "defect_area_fraction": round(float((th > 0).mean()), 5), "nm_per_px": nm_per_px}
