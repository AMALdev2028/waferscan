"""
Loader for the original WM-811K pickle (LSWMD.pkl, 811,457 wafers, 46,393
lots; ~173k carry a failure-type label). On Kaggle it is attached at e.g.
/kaggle/input/wm811k-wafer-map/LSWMD.pkl.

Quirks handled:
- The pickle was written by pandas 0.x; newer pandas cannot find the old
  module path `pandas.indexes`, so we alias it to `pandas.core.indexes`.
- failureType / trianTestLabel (sic) are nested arrays like [['Edge-Ring']];
  unlabeled wafers hold an empty array and are skipped.
- Leakage unit = lotName: wafers of one lot share process history and look
  alike, so a lot must never straddle train/validation.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

from waferscan.classes import INDEX, canonical
from waferscan.data.preprocess import to_input


def read_lswmd(path: str) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, ImportError, AttributeError, UnicodeDecodeError):
        pass
    # Written by pandas 0.x under Python 2: old module paths AND py2 byte strings.
    import pandas.core.indexes as idx
    from pandas.compat import pickle_compat
    sys.modules.setdefault("pandas.indexes", idx)
    for sub in ("base", "range", "numeric", "multi", "category", "datetimes", "frozen"):
        try:
            sys.modules.setdefault(f"pandas.indexes.{sub}", importlib.import_module(f"pandas.core.indexes.{sub}"))
        except ImportError:
            pass
    with open(path, "rb") as fh:
        return pickle_compat.Unpickler(fh, encoding="latin1").load()


def _label(a) -> str | None:
    a = np.asarray(a)
    return str(a.ravel()[0]) if a.size else None


def load_lswmd(path: str, size: int = 64, none_cap: int | None = 20000, seed: int = 0,
               return_frame: bool = False):
    """-> X (N,2,S,S) float32, y (N,), groups (N,) [, frame with lot/wafer/dieSize]."""
    df = read_lswmd(path)
    df = df.assign(label=df["failureType"].map(_label)).dropna(subset=["label"])
    df["cls"] = df["label"].map(canonical)
    if none_cap is not None:
        none = df.index[df["cls"] == "None"]
        if len(none) > none_cap:
            drop = np.random.default_rng(seed).choice(none, len(none) - none_cap, replace=False)
            df = df.drop(index=drop)
    X = np.stack([to_input(m, size) for m in df["waferMap"]]).astype(np.float32)
    y = df["cls"].map(INDEX).to_numpy()
    groups = pd.factorize(df["lotName"])[0]
    if return_frame:
        return X, y, groups, df.reset_index(drop=True)
    return X, y, groups


def to_npz(pkl: str, out: str, size: int = 64, none_cap: int | None = 20000, seed: int = 0) -> dict:
    """LSWMD.pkl -> the same npz layout the rest of the pipeline uses. Native die
    maps are stored flat (maps_flat + shapes) because they vary up to ~300x200."""
    import json
    import os
    X, y, groups, df = load_lswmd(pkl, size, none_cap, seed, return_frame=True)
    maps = [np.asarray(m, np.uint8) for m in df["waferMap"]]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, X=X.astype(np.float16), y=y, groups=groups,
                        maps_flat=np.concatenate([m.ravel() for m in maps]),
                        shapes=np.array([m.shape for m in maps]),
                        # plain unicode arrays, not object arrays: np.load(allow_pickle=False) must read them
                        sources=df["lotName"].astype(str).to_numpy(dtype=str),
                        files=(df["lotName"].astype(str) + "_w"
                               + df["waferIndex"].astype(int).astype(str)).to_numpy(dtype=str))
    audit = {"source": os.path.basename(pkl), "wafers": int(len(y)), "lots": int(groups.max() + 1),
             "none_cap": none_cap, "per_class": {c: int((y == i).sum()) for c, i in INDEX.items()}}
    with open(os.path.splitext(out)[0] + "_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2)
    return audit


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="convert LSWMD.pkl to a processed npz")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="data/processed/wm811k.npz")
    ap.add_argument("--none-cap", type=int, default=20000)
    a = ap.parse_args()
    print(json.dumps(to_npz(a.pkl, a.out, none_cap=a.none_cap), indent=2))
