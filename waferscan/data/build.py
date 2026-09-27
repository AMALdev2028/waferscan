"""
Build the training set from a folder of rendered wafer-map images
(<root>/<Class>/*.jpg, e.g. wafer_defect/backend/dataset) and audit it for
leakage.

    python -m waferscan.data.build --images <root> --out data/processed/rendered_wm811k.npz

Leakage audit (why the groups matter)
-------------------------------------
Roboflow exports contain several augmented copies of one source wafer
("image_ER_26289_png_jpg.rf.<hash1>.jpg", "...rf.<hash2>.jpg"). If one copy
lands in training and the other in validation, validation measures memory,
not skill. So every image gets a *group* = its source wafer, and all splits
keep a group on one side. We also look for copies that were renamed
(different source id, same wafer): the random fail speckle on a wafer is a
fingerprint, so two maps that agree on >90% of dies after the best of the 8
rotations/flips are the same wafer, and their groups are merged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from waferscan.classes import CLASSES, canonical
from waferscan.data.images import rendered_to_diemap
from waferscan.data.preprocess import to_input

_RF = re.compile(r"(_(png|jpe?g|bmp))*\.rf\.[0-9a-f]+\.\w+$", re.I)


def native_map(d, i: int) -> np.ndarray:
    """Die map i of a processed npz, cropped to its wafer - works for both
    layouts (padded `maps` from images, flat `maps_flat` from LSWMD.pkl)."""
    if "maps_flat" in d:
        sizes = np.prod(d["shapes"], 1)
        off = int(sizes[:i].sum())
        m = d["maps_flat"][off:off + sizes[i]].reshape(d["shapes"][i])
    else:
        m = d["maps"][i]
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def source_id(fname: str) -> str:
    stem = _RF.sub("", fname)
    return stem if stem != fname else os.path.splitext(fname)[0]


def _convert(path: str):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rendered_to_diemap(rgb)


def _dihedral(x: np.ndarray):
    for k in range(4):
        r = np.rot90(x, k, axes=(-2, -1))
        yield r
        yield r[..., ::-1]


def near_duplicate_pairs(fail: np.ndarray, wafer: np.ndarray, thresh: float = 0.8) -> list[tuple[int, int, float]]:
    """Pairs (i, j, similarity) of maps that match after one of the 8
    rotations/flips. similarity = min(IoU of failed dies, IoU of passing dies):
    plain "% of dies that agree" is useless here, because two unrelated
    near-full wafers (or two nearly clean ones) agree on ~95% of dies by
    chance. Requiring *both* the fail set and the pass set to overlap only
    fires for genuine copies. On the Roboflow set the score is bimodal
    (unrelated < 0.6, copies > 0.8), hence thresh=0.8."""
    n = len(fail)
    pas = wafer & ~fail
    f = fail.reshape(n, -1).astype(np.float32)
    p = pas.reshape(n, -1).astype(np.float32)
    nf, npas = f.sum(1), p.sum(1)
    best = np.zeros((n, n), np.float32)
    for tf, tp in zip(_dihedral(fail), _dihedral(pas)):
        g = tf.reshape(n, -1).astype(np.float32)
        q = tp.reshape(n, -1).astype(np.float32)
        ff, pp = f @ g.T, p @ q.T
        iou_f = ff / np.maximum(nf[:, None] + g.sum(1)[None] - ff, 1.0)
        iou_p = pp / np.maximum(npas[:, None] + q.sum(1)[None] - pp, 1.0)
        np.maximum(best, np.minimum(iou_f, iou_p), out=best)
    iu = np.triu_indices(n, 1)
    keep = best[iu] >= thresh
    return [(int(i), int(j), float(a)) for i, j, a in zip(iu[0][keep], iu[1][keep], best[iu][keep])]


def _union_find(n: int, pairs) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, *_ in pairs:
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def build(images_root: str, out_path: str, size: int = 64, workers: int = 2) -> dict:
    items = []
    for folder in sorted(os.listdir(images_root)):
        d = os.path.join(images_root, folder)
        if not os.path.isdir(d):
            continue
        cls = canonical(folder)
        items += [(os.path.join(d, f), cls, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_convert, [p for p, _, _ in items], chunksize=32))

    maps = [r[0] for r in results]
    infos = [r[1] for r in results]
    hmax = max(m.shape[0] for m in maps)
    wmax = max(m.shape[1] for m in maps)
    padded = np.zeros((len(maps), hmax, wmax), np.uint8)       # pad with 0 = off-wafer
    for i, m in enumerate(maps):
        y0, x0 = (hmax - m.shape[0]) // 2, (wmax - m.shape[1]) // 2
        padded[i, y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    X = np.stack([to_input(m, size) for m in maps])           # (N, 2, S, S) model inputs
    y = np.array([CLASSES.index(c) for _, c, _ in items], np.int64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, X=X.astype(np.float16), y=y, groups=np.zeros(len(y), np.int64), maps=padded,
                        shapes=np.array([m.shape for m in maps]),
                        sources=np.array([source_id(f) for _, _, f in items]),
                        files=np.array([f for _, _, f in items]),
                        agreement=np.array([inf["agreement"] for inf in infos]),
                        palette=np.array([inf["palette"] for inf in infos]))
    return regroup(out_path)


def regroup(npz_path: str) -> dict:
    """(Re)compute leakage groups + the audit report for a built dataset."""
    d = dict(np.load(npz_path))
    X, y, sources = d["X"].astype(np.float32), d["y"], d["sources"]
    src_ids = np.unique(sources, return_inverse=True)[1]
    conflicts = sorted(s for s in np.unique(sources) if len(np.unique(y[sources == s])) > 1)
    # near duplicates across *different* source ids -> merge groups
    # (32x32 is enough to fingerprint the fail speckle and keeps the N^2 check fast)
    s = X.shape[-1] // 32
    small = X.reshape(len(X), 2, 32, s, 32, s).mean((3, 5))
    pairs = [p for p in near_duplicate_pairs(small[:, 1] > 0.5, small[:, 0] > 0.5)
             if src_ids[p[0]] != src_ids[p[1]]]
    first, links = {}, []
    for i, sid in enumerate(src_ids):
        links.append((i, first.setdefault(sid, i)))
    groups = _union_find(len(y), links + pairs)
    d["groups"] = groups
    np.savez_compressed(npz_path, **d)

    palettes, agreement = d["palette"], d["agreement"]
    shapes, counts = np.unique(d["shapes"], axis=0, return_counts=True)
    audit = {
        "images": int(len(y)),
        "unique_source_ids": int(src_ids.max() + 1),
        "groups_after_near_duplicate_merge": int(groups.max() + 1),
        "near_duplicate_pairs_across_ids": len(pairs),
        "near_duplicate_pairs_with_different_labels": int(sum(1 for i, j, _ in pairs if y[i] != y[j])),
        "label_conflicts_same_source": conflicts[:20],
        "per_class": {c: {"images": int((y == k).sum()),
                          "independent_wafers": int(len(np.unique(groups[y == k]))),
                          "images_with_renamed_copy": int(len({i for p in pairs for i in p[:2] if y[i] == k})),
                          "nonstandard_palette": int(((y == k) & (palettes != "standard")).sum())}
                      for k, c in enumerate(CLASSES)},
        "render_agreement": {"min": float(agreement.min()), "p01": float(np.percentile(agreement, 1)),
                             "median": float(np.median(agreement))},
        "grid_shapes": {f"{a}x{b}": int(n) for (a, b), n in zip(shapes, counts)},
    }
    with open(os.path.splitext(npz_path)[0] + "_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2)
    return audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder with <Class>/*.jpg")
    ap.add_argument("--out", default="data/processed/rendered_wm811k.npz")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--regroup", action="store_true", help="only recompute groups/audit for --out")
    a = ap.parse_args()
    res = regroup(a.out) if a.regroup else build(a.images, a.out, a.size, a.workers)
    print(json.dumps({k: v for k, v in res.items() if k != "grid_shapes"}, indent=2))
