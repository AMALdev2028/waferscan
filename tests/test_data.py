import io
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from waferscan.data.build import near_duplicate_pairs, source_id
from waferscan.data.images import PALETTE, optical_to_diemap, rendered_to_diemap, sem_segment
from waferscan.data.preprocess import input_to_native, to_input
from waferscan.data.synthetic import make_wafer, synthetic_optical_scan


def render(dm, colors=PALETTE, size=640, jpeg=85):
    """Die map -> Roboflow-style JPEG: colour per die, stretched to size x size."""
    rgb = colors[dm].astype(np.uint8)
    img = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=jpeg)
    return np.asarray(Image.open(buf).convert("RGB"))


def test_rendered_roundtrip_recovers_every_die():
    rng = np.random.default_rng(0)
    for cls in ("Edge-Ring", "Scratch", "None", "Near-full"):
        dm = make_wafer(cls, rng, n=52)
        got, info = rendered_to_diemap(render(dm))
        assert got.shape == dm.shape, (cls, info)
        assert (got == dm).mean() > 0.99, cls


def test_nonstandard_palette_is_decoded():
    # levels at viridis 0 / 0.33 / 0.67 instead of 0 / 0.5 / 1 (seen in the Random class)
    odd = np.array([[68, 1, 84], [49, 104, 142], [53, 183, 121]], np.float32)
    dm = make_wafer("Random", np.random.default_rng(1), n=40)
    got, info = rendered_to_diemap(render(dm, colors=odd))
    assert info["palette"] != "standard"
    assert (got == dm).mean() > 0.99


def test_source_id_strips_roboflow_suffix():
    assert source_id("image_ER_26289_png_jpg.rf.0e198527160046f21209a5afd7513fc5.jpg") == "image_ER_26289"
    assert source_id("811KNF_73_png_jpg.rf.0ee923ddbcc23c539c28ccd55757003a.jpg") == "811KNF_73"
    assert source_id("plain.png") == "plain"


def test_near_duplicates_found_but_not_for_lookalikes():
    rng = np.random.default_rng(2)
    a = to_input(make_wafer("Edge-Loc", rng))
    b = np.rot90(a, 1, axes=(-2, -1))                      # renamed rotated copy
    c = to_input(make_wafer("Edge-Loc", rng))              # different wafer, same class
    d1, d2 = to_input(make_wafer("Near-full", rng)), to_input(make_wafer("Near-full", rng))
    X = np.stack([a, b, c, d1, d2])
    pairs = {(i, j) for i, j, _ in near_duplicate_pairs(X[:, 1] > 0.5, X[:, 0] > 0.5)}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (3, 4) not in pairs     # two ~90%-fail wafers are not copies


def test_input_to_native_inverts_crop_and_resize():
    dm = make_wafer("Center", np.random.default_rng(3), n=37)
    x = to_input(dm)
    back = input_to_native(x[1], dm)
    assert back.shape == dm.shape
    assert np.corrcoef(back[dm > 0], (dm == 2)[dm > 0])[0, 1] > 0.8


def test_optical_scan_die_to_die_inspection():
    n = 25
    yy, xx = np.mgrid[:n, :n]
    dm = np.where(np.hypot(xx - 12, yy - 12) <= 11.5, 1, 0).astype(np.uint8)
    fails = [(5, 12), (12, 5), (12, 12), (13, 12), (18, 16)]      # interior dies only
    for r, c in fails:
        dm[r, c] = 2
    got, affected, info = optical_to_diemap(synthetic_optical_scan(dm))
    assert (got == 2).sum() == len(fails), info
    assert (got == 1).sum() > 0.7 * ((dm == 1).sum())
    assert affected[got == 2].min() > affected[got == 1].max()


def test_sem_segmentation_measures_and_hints():
    img = np.full((400, 400), 90, np.uint8)
    img = np.clip(img + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    cv2.circle(img, (150, 200), 30, 230, -1)
    cv2.line(img, (250, 60), (380, 330), 30, 3)
    res = sem_segment(img, nm_per_px=10.0)
    blob = min(res["defects"], key=lambda d: abs(d["centroid_px"][0] - 150) + abs(d["centroid_px"][1] - 200))
    assert abs(blob["area_um2"] - np.pi * 30 ** 2 * 1e-4) / (np.pi * 30 ** 2 * 1e-4) < 0.15
    assert blob["shape_hint"] == "particle"
    assert any(d["shape_hint"] == "scratch" for d in res["defects"])


def test_lswmd_parsing(tmp_path):
    from waferscan.data.wm811k import load_lswmd
    rng = np.random.default_rng(4)
    rows = []
    for i, lab in enumerate(["Edge-Ring", "none", "Loc", None, "Near-full", "none"]):
        rows.append({"waferMap": make_wafer("None", rng, n=30), "dieSize": 700.0, "lotName": f"lot{i // 2}",
                     "waferIndex": float(i + 1), "trianTestLabel": np.array([["Training"]]),
                     "failureType": np.array([[lab]]) if lab else np.zeros((0, 0))})
    p = tmp_path / "LSWMD.pkl"
    with open(p, "wb") as fh:
        pickle.dump(pd.DataFrame(rows), fh)
    X, y, groups = load_lswmd(str(p), none_cap=1)
    assert X.shape == (4, 2, 64, 64)                        # unlabeled wafer dropped, one 'none' capped
    assert len(set(groups)) <= 3
