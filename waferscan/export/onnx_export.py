"""
Export trained fold models for serving.

    python -m waferscan.export.onnx_export --run runs/cpu_wafernet --bundle model_bundle

Writes to the bundle:
  ensemble.onnx   all K fold models in one graph (deep ensemble, full path)
  fast.onnx       fold-0 model alone (fast path + FP32 edge fallback)
  edge_int8.onnx  fast.onnx statically quantised to INT8 (offline fab stations)
  manifest.json   input/output contract, parity + INT8 accuracy checks
Outputs of every graph: logits (B[,K],C), evid (B[,K],C), g (B[,K],D), fmap (B[,K],D,h,w)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import yaml

from waferscan.classes import CLASSES
from waferscan.models.wafernet import Ensemble, build_model

OUTPUTS = ["logits", "evid", "g", "fmap"]


def load_members(run: str) -> tuple[list[torch.nn.Module], dict]:
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    members = []
    for k in range(cfg["train"]["folds"]):
        p = os.path.join(run, f"fold{k}.pt")
        if os.path.exists(p):
            m = build_model(cfg["model"]["backbone"], len(CLASSES), drop=cfg["model"].get("dropout", 0.3))
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            members.append(m.eval())
    return members, cfg


def export(model: torch.nn.Module, path: str, size: int = 64):
    x = torch.zeros(2, 2, size, size)
    dyn = {"input": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS}}
    torch.onnx.export(model, (x,), path, input_names=["input"], output_names=OUTPUTS, dynamic_axes=dyn,
                      opset_version=17, dynamo=False)


def parity(model: torch.nn.Module, path: str, X: np.ndarray) -> float:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(X))
    got = sess.run(None, {"input": X})
    return float(max(np.abs(r.numpy() - g).max() for r, g in zip(ref, got)))


class _Calib:
    def __init__(self, X):
        self.it = iter([{"input": X[i:i + 16]} for i in range(0, len(X), 16)])

    def get_next(self):
        return next(self.it, None)


def quantize_int8(fp32: str, out: str, calib: np.ndarray):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = out.replace(".onnx", "_pre.onnx")
    quant_pre_process(fp32, pre)
    quantize_static(pre, out, _Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    os.remove(pre)


def latency(path: str, x: np.ndarray, n: int = 200) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(10):
        sess.run(None, {"input": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {"input": x})
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
            "samples_ms": [round(t, 3) for t in ts[-64:]]}


def benchmark_end_to_end(bundle: str) -> dict:
    """Whole-request latency (inference + verification + CAM + quantification)
    and the 4K optical-scan -> die-map step, on this machine."""
    from waferscan.data.images import optical_to_diemap
    from waferscan.data.synthetic import synthetic_optical_scan
    from waferscan.inference.predictor import Predictor
    p = Predictor(bundle)
    maps = [s["map"] for s in p.samples]
    out = {}
    for mode in ("auto", "full"):
        ts = []
        for m in maps * 2:
            t = time.perf_counter()
            p.predict(m, mode=mode, include_die_map=False)
            ts.append((time.perf_counter() - t) * 1000)
        out[f"predict_{mode}"] = {"p50_ms": round(float(np.percentile(ts, 50)), 2),
                                  "p95_ms": round(float(np.percentile(ts, 95)), 2)}
    n = 64
    yy, xx = np.mgrid[:n, :n]
    dm = np.where(np.hypot(xx - 31.5, yy - 31.5) <= 31.5, 1, 0).astype(np.uint8)
    dm[20:23, 30:33] = 2
    img = synthetic_optical_scan(dm, die=62, tilt=0.95)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        optical_to_diemap(img)
        ts.append((time.perf_counter() - t) * 1000)
    out["optical_4k_to_diemap"] = {"image_wh": [int(img.shape[1]), int(img.shape[0])], "best_ms": round(min(ts), 1)}
    return out


def main(runs: list[str], data: str, bundle: str):
    """runs[0] also provides the fast/edge model; every run becomes one ensemble graph
    (ensemble.onnx for run 0, ensemble_run<r>.onnx for extra backbones)."""
    from waferscan.training.train import make_folds
    os.makedirs(bundle, exist_ok=True)
    d = np.load(data)
    X, y, groups = d["X"].astype(np.float32), d["y"], d["groups"]
    run_meta, checks = [], {}
    for r, run in enumerate(runs):
        mem, c = load_members(run)
        key = "ensemble" if r == 0 else f"ensemble_run{r}"
        path = os.path.join(bundle, f"{key}.onnx")
        export(Ensemble(mem).eval(), path)
        checks[f"parity_max_abs_diff_{key}"] = parity(Ensemble(mem).eval(), path, X[:32])
        run_meta.append({"key": key, "file": f"{key}.onnx", "backbone": c["model"]["backbone"],
                         "members": len(mem), "dropout": c["model"].get("dropout", 0.3)})
        if r == 0:
            members, cfg = mem, c
    t = cfg["train"]
    va0 = make_folds(y, groups, t["folds"], t.get("split", "grouped"), t.get("seed", 42))[0][1]
    Xv, yv = X[va0], y[va0]                          # fold-0 validation: never seen by member 0

    ens_path, fast_path, int8_path = (os.path.join(bundle, n) for n in ("ensemble.onnx", "fast.onnx", "edge_int8.onnx"))
    export(members[0], fast_path)
    checks["parity_max_abs_diff_fast"] = parity(members[0], fast_path, Xv[:64])

    rng = np.random.default_rng(0)
    calib = X[np.setdiff1d(np.arange(len(X)), va0)][rng.choice(len(X) - len(va0), 256, replace=False)]
    quantize_int8(fast_path, int8_path, calib)
    acc = {}
    for name, p in (("fast_fp32", fast_path), ("edge_int8", int8_path)):
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        pred = np.concatenate([sess.run(["logits"], {"input": Xv[i:i + 256]})[0].argmax(1)
                               for i in range(0, len(Xv), 256)])
        acc[name] = float((pred == yv).mean())
    checks["fold0_val_accuracy"] = acc
    x1 = Xv[:1]
    lat = {"fast_fp32": latency(fast_path, x1), "edge_int8": latency(int8_path, x1),
           "ensemble_fp32": latency(ens_path, x1)}
    manifest = {
        "classes": CLASSES, "input": {"name": "input", "shape": ["batch", 2, 64, 64], "dtype": "float32",
                                      "channels": ["on_wafer_mask", "failed_die_mask"]},
        "outputs": OUTPUTS, "members": len(members), "backbone": cfg["model"]["backbone"],
        "dropout": cfg["model"].get("dropout", 0.3), "runs": run_meta, "checks": checks,
        "cpu_latency_batch1": {k: {kk: vv for kk, vv in v.items() if kk != "samples_ms"} for k, v in lat.items()},
        "cpu": os.cpu_count(), "onnxruntime": ort.__version__,
        "files": {p: round(os.path.getsize(os.path.join(bundle, p)) / 1e6, 2)
                  for p in ("ensemble.onnx", "fast.onnx", "edge_int8.onnx")},
    }
    with open(os.path.join(bundle, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(bundle, "latency_samples.json"), "w") as fh:
        json.dump({k: v["samples_ms"] for k, v in lat.items()}, fh)
    if all(os.path.exists(os.path.join(bundle, f)) for f in ("policy.json", "stack.joblib", "samples.npz")):
        manifest["end_to_end_ms"] = benchmark_end_to_end(bundle)      # needs `stack` to have run first
        with open(os.path.join(bundle, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k in ("members", "checks", "cpu_latency_batch1", "files", "end_to_end_ms")}, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/cpu_wafernet"], help="same order as used for stacking")
    ap.add_argument("--data", default="data/processed/rendered_wm811k.npz")
    ap.add_argument("--bundle", default="model_bundle")
    a = ap.parse_args()
    main(a.runs, a.data, a.bundle)
