"""
Every metric the spec asks for, from true labels + predicted probabilities.

- per-class precision / recall / F1 / support, macro / micro / weighted
- accuracy, balanced accuracy, top-2 accuracy
- Cohen's kappa (agreement beyond chance), Matthews correlation coefficient
- ECE: expected calibration error - when the model says 90%, is it right
  90% of the time? (15 equal-width confidence bins)
- per-class one-vs-rest ROC AUC and the Youden's J threshold
  (the probability cut-off maximising TPR - FPR) for each class
- group-bootstrap 95% confidence intervals: resample *wafer groups*, not
  images, so duplicate copies don't make the interval look tighter than it is.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
                             roc_curve, top_k_accuracy_score)


def ece(y: np.ndarray, proba: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, cf = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            total += m.mean() * abs(acc - cf)
            rel.append({"bin": [round(lo, 3), round(hi, 3)], "n": int(m.sum()), "accuracy": acc, "confidence": cf})
    return float(total), rel


def youden_thresholds(y: np.ndarray, proba: np.ndarray) -> list[float]:
    out = []
    for c in range(proba.shape[1]):
        t = (y == c).astype(int)
        if t.min() == t.max():
            out.append(0.5)
            continue
        fpr, tpr, thr = roc_curve(t, proba[:, c])
        out.append(float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1)))
    return out


def group_bootstrap_ci(y, pred, groups, fn, n_boot: int = 300, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        vals.append(fn(y[idx], pred[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def full_report(y: np.ndarray, proba: np.ndarray, classes: list[str], groups: np.ndarray | None = None) -> dict:
    labels = list(range(len(classes)))
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    onehot = np.eye(len(classes))[y]
    try:
        aucs = roc_auc_score(onehot, proba, average=None)
    except ValueError:
        aucs = [float("nan")] * len(classes)
    e, reliability = ece(y, proba)
    thr = youden_thresholds(y, proba)
    rep = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "top2_accuracy": float(top_k_accuracy_score(y, proba, k=2, labels=labels)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "ece": e,
        "macro": dict(zip(["precision", "recall", "f1"],
                          map(float, precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[:3]))),
        "micro": dict(zip(["precision", "recall", "f1"],
                          map(float, precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[:3]))),
        "weighted": dict(zip(["precision", "recall", "f1"],
                             map(float, precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[:3]))),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i]),
                          "auc_ovr": float(aucs[i]), "youden_threshold": thr[i]} for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "reliability": reliability,
    }
    if groups is not None:
        rep["ci95"] = {
            "accuracy": group_bootstrap_ci(y, pred, groups, accuracy_score),
            "macro_f1": group_bootstrap_ci(y, pred, groups, lambda a, b: f1_score(a, b, average="macro")),
        }
    return rep
