"""
The inference pipeline (what the API calls):

  die map -> model input + morphology features
          -> FAST path: one small model. Accepted only if confident AND the
             morphology check passes (dynamic selection by input difficulty).
          -> otherwise FULL path: K-member deep ensemble (+ optional 8-way
             rotation/flip TTA) with MC dropout, evidential uncertainty,
             morphology GBM and the stacking meta-learner.
          -> verification -> decision (auto_accept / human_review)
          -> CAM heatmap on the native die grid -> affected-area quantification

Backends: local ONNX Runtime (default) or NVIDIA Triton over the KServe v2
HTTP protocol (WAFERSCAN_BACKEND=triton, TRITON_URL=http://triton:8000), so
the same API can run CPU-only or in front of a GPU Triton deployment.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from waferscan.classes import CLASSES
from waferscan.data.preprocess import input_to_native, to_input
from waferscan.features.morphology import FEATURE_NAMES, features
from waferscan.inference.policy import decide, dirichlet, entropy, mc_dropout, softmax, verify
from waferscan.inference.quantify import quantify

OUTPUTS = ["logits", "evid", "g", "fmap"]


class OnnxBackend:
    def __init__(self, bundle: str, keys: list[str], threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        # small hosts (e.g. 0.1 vCPU / 512 MB free tiers): os.cpu_count() reports the HOST's cores,
        # and too many threads under a CPU quota is much slower, so let the deployment set it
        so.intra_op_num_threads = threads or int(os.environ.get("WAFERSCAN_THREADS", 0)) or max(1, os.cpu_count() or 1)
        if os.environ.get("WAFERSCAN_LOW_MEMORY") == "1":
            so.enable_cpu_mem_arena = False      # smaller peak RAM, slightly slower
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sessions = {k: ort.InferenceSession(os.path.join(bundle, f"{k}.onnx"), so, providers=prov)
                         for k in ["fast", *keys]}
        self.name = f"onnxruntime ({prov[0]})"

    def run(self, which: str, x: np.ndarray) -> dict:
        return dict(zip(OUTPUTS, self.sessions[which].run(OUTPUTS, {"input": x.astype(np.float32)})))

    def ready(self) -> bool:
        return True


class TritonBackend:
    """Triton Inference Server via the KServe v2 HTTP/JSON protocol. Model
    names: waferscan_<key><suffix>, suffix "" (ONNX Runtime) or "_trt" (TensorRT)."""

    def __init__(self, url: str, suffix: str | None = None, timeout: float = 10.0, client=None):
        import httpx
        self.client = client or httpx.Client(base_url=url, timeout=timeout)
        self.suffix = os.environ.get("TRITON_MODEL_SUFFIX", "") if suffix is None else suffix
        self.name = f"triton ({url}, models waferscan_*{self.suffix})"

    def run(self, which: str, x: np.ndarray) -> dict:
        body = {"inputs": [{"name": "input", "shape": list(x.shape), "datatype": "FP32",
                            "data": x.astype(np.float32).ravel().tolist()}],
                "outputs": [{"name": n} for n in OUTPUTS]}
        r = self.client.post(f"/v2/models/waferscan_{which}{self.suffix}/infer", json=body)
        r.raise_for_status()
        return {o["name"]: np.asarray(o["data"], np.float32).reshape(o["shape"]) for o in r.json()["outputs"]}

    def ready(self) -> bool:
        try:
            return self.client.get("/v2/health/ready", timeout=2.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False


def _dihedral_batch(x: np.ndarray) -> np.ndarray:
    return np.stack([t for k in range(4) for t in (np.rot90(x, k, (-2, -1)), np.rot90(x, k, (-2, -1))[..., ::-1])])


class Predictor:
    def __init__(self, bundle: str = "model_bundle", backend=None):
        self.bundle = bundle
        load = lambda n: json.load(open(os.path.join(bundle, n)))  # noqa: E731
        self.manifest, self.policy, self.card = load("manifest.json"), load("policy.json"), load("model_card.json")
        self.rules = self.policy["verifier_rules"]
        h = np.load(os.path.join(bundle, "heads.npz"))
        # one entry per trained backbone (run); run 0 also supplies the fast model
        self.runs = [dict(r) for r in (self.manifest.get("runs")    # copies: the manifest is served as JSON
                                       or [{"key": "ensemble", "members": self.manifest["members"]}])]
        for r, run in enumerate(self.runs):
            run["heads"] = [{n: h[f"run{r}_m{i}_{n}"] for n in ("fc.weight", "fc.bias", "evid.weight", "evid.bias")}
                            for i in range(run["members"])]
            run["dropout"] = float(h[f"run{r}_dropout"])
        self.heads, self.dropout = self.runs[0]["heads"], self.runs[0]["dropout"]
        self.stack = joblib.load(os.path.join(bundle, "stack.joblib"))
        s = np.load(os.path.join(bundle, "samples.npz"))
        self.samples = [{"id": str(i), "label": CLASSES[int(y)], "map": self._crop(m, sh)}
                        for i, (m, sh, y) in enumerate(zip(s["maps"], s["shapes"], s["y"]))]
        if backend is None:
            if os.environ.get("WAFERSCAN_BACKEND", "onnx") == "triton":
                backend = TritonBackend(os.environ.get("TRITON_URL", "http://localhost:8000"))
            else:
                backend = OnnxBackend(bundle, [r["key"] for r in self.runs])
        self.backend = backend

    @staticmethod
    def _crop(m: np.ndarray, shape) -> np.ndarray:
        ys, xs = np.nonzero(m)
        return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    # ------------------------------------------------------------------
    def _cam(self, fmap: np.ndarray, members: list[int], cls: int) -> np.ndarray:
        """fmap: (K, D, h, w) -> class activation map in input space (S, S)."""
        cams = [np.tensordot(self.heads[m]["fc.weight"][cls], fmap[i], axes=(0, 0)) for i, m in enumerate(members)]
        cam = np.maximum(np.mean(cams, 0), 0)
        cam = cv2.resize(cam.astype(np.float32), (64, 64), interpolation=cv2.INTER_CUBIC)
        return np.maximum(cam, 0) / (cam.max() + 1e-9)

    def predict(self, dm: np.ndarray, mode: str = "auto", tta: bool = False, mc_samples: int = 30,
                geometry: dict | None = None, die_affected: np.ndarray | None = None,
                include_die_map: bool = True) -> dict:
        t0 = time.perf_counter()
        dm = np.asarray(dm, np.uint8)
        if dm.ndim != 2 or not (dm > 0).any() or dm.max() > 2:
            raise ValueError("wafer map must be a 2-D grid of 0 (off-wafer), 1 (pass), 2 (fail) with dies")
        x = to_input(dm)[None]
        feat = features(x[0])
        fast = self.backend.run("fast", x)
        # temperature scaling: the raw head is underconfident (label smoothing + focal loss)
        p_fast = softmax(fast["logits"] / self.policy.get("fast_temperature", 1.0))[0]
        v_fast = verify(feat, CLASSES[int(p_fast.argmax())], self.rules)
        # fast-path cut-off: calibrated on out-of-fold data for >= 99.8% accuracy of this single model
        fast_ok = bool(p_fast.max() >= self.policy["fast_path_threshold"] and v_fast["status"] == "PASS")
        use_fast = mode == "fast" or (mode == "auto" and fast_ok)
        h0 = self.heads[0]
        if use_fast:
            mc = mc_dropout(fast["g"], h0["fc.weight"], h0["fc.bias"], self.dropout, mc_samples)
            ev = dirichlet(fast["evid"])
            probs, members, fmap = p_fast, [0], fast["fmap"][:1]    # the calibrated probabilities
            unc = {"mc_dropout_mutual_info": float(mc["mutual_info"][0]), "evidential_u": float(ev["u"][0]),
                   "ensemble_std": None, "ensemble_mutual_info": None, "members_used": 1}
        else:
            xb = _dihedral_batch(x[0]) if tta else x
            p_runs, mis, us, p_members, fmap = [], [], [], [], None
            for run in self.runs:                                      # one deep ensemble per backbone
                out = self.backend.run(run["key"], xb)                 # (B, K, ...)
                p_mc = []
                for m, hm in enumerate(run["heads"]):
                    r = mc_dropout(out["g"][:, m], hm["fc.weight"], hm["fc.bias"], run["dropout"], mc_samples, seed=m)
                    p_mc.append(r["probs"].mean(0))
                    mis.append(r["mutual_info"].mean())
                p_runs.append(np.mean(p_mc, 0))
                p_members.append(softmax(out["logits"]).mean(0))       # (K, C), averaged over TTA
                us.append(dirichlet(out["evid"])["u"].mean())
                if fmap is None:
                    fmap = out["fmap"][0]                              # CAM from run 0, identity transform
            p_members = np.concatenate(p_members)
            u = float(np.mean(us))
            # one OpenMP thread: sklearn's tree predictor opens a parallel region per
            # tree (2,700 of them); on a busy machine / concurrent API requests the
            # threads oversubscribe the cores and one wafer took 4-20 s instead of ~3 ms
            with threadpool_limits(limits=1, user_api="openmp"):
                p_gbm = self.stack["gbm"].predict_proba(np.array([[feat[n] for n in FEATURE_NAMES]]))[0]
            logs = [np.log(np.clip(p, 1e-6, 1)) for p in (*p_runs, p_gbm)]
            probs = self.stack["meta"].predict_proba(np.concatenate([*logs, [u]])[None])[0]
            pm = p_members.mean(0)
            members = list(range(self.runs[0]["members"]))
            unc = {"mc_dropout_mutual_info": float(np.mean(mis)), "evidential_u": u,
                   "ensemble_std": float(p_members.std(0).max()),
                   "ensemble_mutual_info": float(entropy(pm) - entropy(p_members).mean()),
                   "members_used": len(p_members), "tta_transforms": len(xb),
                   "morphology_gbm_top": CLASSES[int(p_gbm.argmax())]}
        unc["predictive_entropy"] = float(entropy(probs))
        t_inf = time.perf_counter() - t0

        cls_i = int(probs.argmax())
        ver = verify(feat, CLASSES[cls_i], self.rules)
        dec = decide(probs, ver, feat, self.rules, self.policy)
        if use_fast:          # the stacked threshold doesn't apply to single-model probabilities
            dec["action"] = "auto_accept" if fast_ok and not dec["note"] else "human_review"
        final_i = CLASSES.index(dec["final_class"])
        cam = self._cam(fmap, members, final_i)
        cam_native = input_to_native(cam, dm)
        q = quantify(dm, dec["final_class"], cam_native, die_affected=die_affected, **(geometry or {}))
        order = np.argsort(probs)[::-1]
        # secondary alarms: OTHER patterns above their Youden threshold, on this path's own probability scale
        yt = self.policy.get("youden_thresholds_fast" if use_fast else "youden_thresholds", {})
        res = {
            "prediction": {"class": dec["final_class"], "model_top_class": CLASSES[cls_i],
                           "confidence": dec["confidence"], "path": "fast" if use_fast else "full",
                           "probabilities": {CLASSES[i]: round(float(probs[i]), 6) for i in order},
                           "top2": [CLASSES[i] for i in order[:2]],
                           "alarms": [c for i, c in enumerate(CLASSES)
                                      if c not in ("None", dec["final_class"]) and probs[i] >= yt.get(c, 1.1)]},
            "uncertainty": unc,
            "verification": ver,
            "decision": {"action": dec["action"], "note": dec["note"],
                         "auto_accept_threshold": self.policy["auto_accept_threshold"]},
            "quantification": q,
            "cam": np.round(cam_native, 3).tolist(),
            "wafer": {"rows": int(dm.shape[0]), "cols": int(dm.shape[1])},
            "timing_ms": {"inference": round(t_inf * 1000, 2), "total": round((time.perf_counter() - t0) * 1000, 2)},
            "model": {"backend": self.backend.name, "backbones": [r.get("backbone", "wafernet") for r in self.runs],
                      "members": sum(r["members"] for r in self.runs), "trained_on": self.card["trained_on"]},
        }
        if include_die_map:
            res["wafer"]["die_map"] = dm.tolist()
        return res
