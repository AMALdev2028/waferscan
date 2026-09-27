"""
Cross-validated training with MLflow tracking.

    python -m waferscan.training.train --config configs/cpu_wafernet.yaml
    python -m waferscan.training.train --config configs/gpu_full.yaml --backbone swin_t

What happens:
1. Load wafers (processed npz / original LSWMD.pkl / synthetic), each with a
   *group* id (source wafer or lot).
2. StratifiedGroupKFold: every fold keeps class proportions AND never splits a
   group across train/validation (no leakage from duplicate copies or lots).
3. Per fold: Lookahead(AdamW) + cosine annealing with warm restarts, focal
   loss (scheduled gamma, label smoothing, class-balanced weights) + an
   evidential head, rotation/flip augmentation, optional minority
   oversampling. Checkpoint every epoch (Colab can disconnect - resume).
4. Out-of-fold (OOF) outputs for every wafer: predictions from the fold
   model that never saw it. These feed stacking, thresholds, the verifier and
   the honest metrics. The K fold models together form the deep ensemble.

We train a fixed number of epochs (ending on a cosine cycle boundary) and
keep the last weights - no "best epoch on validation" picking, which would
leak validation information into the reported score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from waferscan.classes import CLASSES
from waferscan.data.preprocess import rotate
from waferscan.models.wafernet import build_model
from waferscan.training.losses import class_balanced_weights, evidential_loss, focal_loss, gamma_at


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(dcfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (N,2,S,S) float32, y (N,), groups (N,)."""
    src = dcfg.get("source", "npz")
    if src == "npz":
        d = np.load(dcfg["path"], allow_pickle=False)
        return d["X"].astype(np.float32), d["y"], d["groups"]
    if src == "wm811k":
        from waferscan.data.wm811k import load_lswmd
        return load_lswmd(dcfg["path"], size=dcfg.get("size", 64), none_cap=dcfg.get("none_cap"))
    if src == "synthetic":
        from waferscan.data.synthetic import make_dataset
        from waferscan.data.preprocess import to_input
        maps, y, groups = make_dataset(dcfg.get("per_class", 200), seed=dcfg.get("seed", 0))
        return np.stack([to_input(m, dcfg.get("size", 64)) for m in maps]), y, groups
    raise ValueError(f"unknown data source {src}")


def make_folds(y, groups, n_splits: int, split: str, seed: int):
    if split == "random":        # the leaky protocol, kept only to measure the leak
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rotate(s, rng.uniform(0, 360), flip=bool(rng.integers(2))) for s in x])


def oversample(idx: np.ndarray, y: np.ndarray, target: int, rng) -> np.ndarray:
    """Augmentation-based minority oversampling: repeat indices of classes
    with fewer than `target` samples; the random rotation/flip applied every
    time a sample is drawn makes each repeat a new, physically valid wafer.
    (SMOTE-style pixel interpolation between two wafers would invent blurry,
    non-physical maps - that's why it is not used on images.)"""
    if not target:
        return idx
    out = [idx]
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        if len(members) < target:
            out.append(rng.choice(members, target - len(members), replace=True))
    return np.concatenate(out)


@torch.no_grad()
def predict(model, X: np.ndarray, device, bs: int = 256) -> dict:
    model.eval()
    outs = {"logits": [], "evid": [], "g": []}
    for i in range(0, len(X), bs):
        lo, ev, g, _ = model(torch.from_numpy(X[i:i + bs]).to(device))
        outs["logits"].append(lo.float().cpu().numpy())
        outs["evid"].append(ev.float().cpu().numpy())
        outs["g"].append(g.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


# ----------------------------------------------------------------------------
# one fold
# ----------------------------------------------------------------------------
def train_fold(cfg: dict, X, y, tr, va, fold: int, out_dir: str, device, log=print) -> dict:
    import timm.optim
    t = cfg["train"]
    seed = t.get("seed", 42) + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg["model"].get("backbone", "wafernet"), len(CLASSES),
                        drop=cfg["model"].get("dropout", 0.3),
                        pretrained=cfg["model"].get("pretrained", False)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = timm.optim.create_optimizer_v2(model, opt="lookahead_adamw", lr=t["lr"], weight_decay=t["weight_decay"])
    cw = t["cosine_restarts"]
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cw["T_0"], T_mult=cw["T_mult"],
                                                                 eta_min=cw["eta_min"])
    # weights from the class counts the model actually sees per epoch (after the
    # oversampling floor), so the two balancing tricks don't double-correct
    seen = np.maximum(np.bincount(y[tr], minlength=len(CLASSES)), t.get("oversample_to", 0) or 0)
    weights = class_balanced_weights(seen, t["class_balance_beta"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fg = t["focal_gamma"]

    ckpt = os.path.join(out_dir, f"fold{fold}.ckpt")
    start = 0
    if os.path.exists(ckpt):                      # resume after a Colab/Kaggle disconnect
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start = st["epoch"] + 1
        log(f"fold {fold}: resumed at epoch {start}")

    epochs, bs = t["epochs"], t["batch_size"]
    history = []
    for epoch in range(start, epochs):
        model.train()
        order = rng.permutation(oversample(tr, y, t.get("oversample_to", 0), rng))
        steps = math.ceil(len(order) / bs)
        t0, tot = time.time(), 0.0
        for i in range(steps):
            b = order[i * bs:(i + 1) * bs]
            xb = torch.from_numpy(augment(X[b], rng)).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            progress = (epoch + i / steps) / epochs
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, evid, _, _ = model(xb)
            loss = focal_loss(logits, yb, gamma_at(progress, fg["start"], fg["end"], fg["ramp"]), weights,
                              t["label_smoothing"])
            loss = loss + t["evidential_weight"] * evidential_loss(evid, yb, progress, t["evidential_anneal"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step(epoch + (i + 1) / steps)
            tot += loss.item()
        out = predict(model, X[va], device)
        pred = out["logits"].argmax(1)
        rec = {"epoch": epoch, "loss": tot / steps, "val_acc": accuracy_score(y[va], pred),
               "val_macro_f1": f1_score(y[va], pred, average="macro"), "lr": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        history.append(rec)
        log(f"fold {fold} ep {epoch + 1}/{epochs} loss {rec['loss']:.4f} "
            f"val_acc {rec['val_acc']:.4f} macroF1 {rec['val_macro_f1']:.4f} ({rec['sec']:.0f}s)")
        _mlflow_log({f"f{fold}_{k}": v for k, v in rec.items() if k != "epoch"}, step=epoch)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch}, ckpt)
    torch.save(model.state_dict(), os.path.join(out_dir, f"fold{fold}.pt"))
    if os.path.exists(ckpt):          # the fold is final; its resume checkpoint is ~4x the size of the weights
        os.remove(ckpt)
    out = predict(model, X[va], device)
    return {"va": va, **out, "history": history}


def predict_saved(cfg: dict, X, va, fold: int, out_dir: str, device) -> dict:
    """OOF outputs of a fold finished in an earlier session (same seed -> same folds)."""
    model = build_model(cfg["model"].get("backbone", "wafernet"), len(CLASSES),
                        drop=cfg["model"].get("dropout", 0.3), pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, f"fold{fold}.pt"), map_location=device,
                                     weights_only=True))
    return {"va": va, **predict(model, X[va], device), "history": []}


def _mlflow_log(metrics: dict, step: int | None = None):
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
    except Exception:  # noqa: BLE001 - tracking must never kill a training run
        pass


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    t = cfg["train"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    if t.get("num_threads"):
        torch.set_num_threads(t["num_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, groups = load_data(cfg["data"])
    folds = make_folds(y, groups, t["folds"], t.get("split", "grouped"), t.get("seed", 42))
    run_folds = t.get("run_folds") or list(range(len(folds)))

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"].get("tracking_uri")
                            or "sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["mlflow"].get("experiment", "waferscan"))
    n, c = len(y), len(CLASSES)
    oof = {"logits": np.full((n, c), np.nan, np.float32), "evid": np.full((n, c), np.nan, np.float32),
           "g": None, "fold": np.full(n, -1)}
    with mlflow.start_run(run_name=cfg["output"].get("run_name", os.path.basename(out_dir))):
        mlflow.log_params({k: str(v)[:250] for k, v in _flat(cfg).items()})
        mlflow.log_params({"n_samples": n, "n_groups": int(len(np.unique(groups))), "device": device.type})
        # Fold-level resume: a fold whose final weights (fold{k}.pt) exist is never
        # retrained, only re-predicted, so folds split across sessions ("0 1 2" tonight,
        # "3 4" tomorrow) still add up to complete OOF predictions. Delete the run
        # directory to retrain from scratch.
        for k, (tr, va) in enumerate(folds):
            finished = os.path.exists(os.path.join(out_dir, f"fold{k}.pt"))
            if k in run_folds and not finished:
                assert not set(groups[tr]) & set(groups[va]) or t.get("split") == "random"
                res = train_fold(cfg, X, y, tr, va, k, out_dir, device)
            elif finished:
                res = predict_saved(cfg, X, va, k, out_dir, device)
            else:
                continue
            oof["logits"][va], oof["evid"][va], oof["fold"][va] = res["logits"], res["evid"], k
            if oof["g"] is None:
                oof["g"] = np.full((n, res["g"].shape[1]), np.nan, np.float32)
            oof["g"][va] = res["g"]
        done = oof["fold"] >= 0
        pred = oof["logits"][done].argmax(1)
        summary = {"oof_samples": int(done.sum()), "oof_acc": float(accuracy_score(y[done], pred)),
                   "oof_macro_f1": float(f1_score(y[done], pred, average="macro")),
                   "split": t.get("split", "grouped"),
                   "folds_run": sorted(int(f) for f in np.unique(oof["fold"][done]))}
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, float)})
        np.savez_compressed(os.path.join(out_dir, "oof.npz"), y=y, groups=groups, **oof)
        with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        mlflow.log_artifact(os.path.join(out_dir, "summary.json"))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", help="override model.backbone")
    ap.add_argument("--out", help="override output.dir")
    ap.add_argument("--split", choices=["grouped", "random"])
    ap.add_argument("--folds", type=int, nargs="*", help="run only these fold indices")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--data", help="override data.path")
    a = ap.parse_args()
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    if a.data:
        cfg["data"]["path"] = a.data
    if a.backbone:
        cfg["model"]["backbone"] = a.backbone
    if a.out:
        cfg["output"]["dir"] = a.out
    if a.split:
        cfg["train"]["split"] = a.split
    if a.folds is not None:
        cfg["train"]["run_folds"] = a.folds
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    run(cfg)


if __name__ == "__main__":
    main()
