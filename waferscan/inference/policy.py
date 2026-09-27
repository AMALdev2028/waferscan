"""
Uncertainty, verification and routing - the parts that turn "a softmax
score" into a decision you can defend on a fab floor.

Uncertainty (three independent views):
  mc_dropout   - T dropout masks on the pooled features g -> T predictions.
                 Their disagreement (mutual information) = *epistemic*
                 uncertainty: the model hasn't seen enough wafers like this.
                 ponytail: dropout is sampled on the head only (one backbone
                 pass). Full-network MC dropout needs T backbone passes;
                 switch if calibration studies show head-only underestimates.
  dirichlet    - evidential head: alpha = softplus(evidence)+1, u = C/sum(alpha).
  ensemble     - K fold models; their spread (computed in the predictor).

Verification (the "secondary morphological validation"): for each class we
learned which handcrafted features separate it from the rest and their
normal range. A prediction whose own wafer falls outside that range gets a
WARN/FAIL with a plain-English reason ("predicted Edge-Ring but only 12% of
the edge is failing; Edge-Ring wafers here show 35-100%").

Routing: auto-accept only when confidence and verification pass thresholds
chosen on out-of-fold data so that auto-accepted wafers hit a target
accuracy; everything else goes to human review. This is the honest version
of "zero misclassification": you can't make a model perfect, but you can
decide which wafers it is allowed to decide alone.
"""
from __future__ import annotations

import numpy as np

from waferscan.classes import CLASSES
from waferscan.features.morphology import FEATURE_NAMES, NAMES


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis)


def mc_dropout(g: np.ndarray, W: np.ndarray, b: np.ndarray, p: float, T: int = 30, seed: int = 0) -> dict:
    """g: (N, K) pooled features; W: (C, K); b: (C,). Returns mean probs,
    predictive entropy and mutual information (epistemic part)."""
    rng = np.random.default_rng(seed)
    masks = (rng.random((T, 1, g.shape[1])) >= p) / (1.0 - p)          # (T, 1, K)
    probs = softmax((g[None] * masks) @ W.T + b)                        # (T, N, C)
    mean = probs.mean(0)
    h = entropy(mean)
    return {"probs": mean, "entropy": h, "mutual_info": h - entropy(probs).mean(0), "std": probs.std(0)}


def dirichlet(evid_logits: np.ndarray) -> dict:
    alpha = np.logaddexp(0, evid_logits) + 1.0                           # softplus + 1
    s = alpha.sum(-1, keepdims=True)
    return {"probs": alpha / s, "u": (alpha.shape[-1] / s[..., 0]), "alpha": alpha}


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def fit_verifier(F: np.ndarray, y: np.ndarray, k: int = 4, lo: float = 1.0, hi: float = 99.0) -> dict:
    """Per class: the k features with the largest standardised gap to the other
    classes, and the class's [lo, hi] percentile range on each."""
    rules = {}
    sd = F.std(0) + 1e-9
    for c, name in enumerate(CLASSES):
        m = y == c
        if m.sum() < 10:
            continue
        gap = np.abs(F[m].mean(0) - F[~m].mean(0)) / sd
        top = np.argsort(gap)[::-1][:k]
        rules[name] = [{"feature": FEATURE_NAMES[j], "lo": float(np.percentile(F[m, j], lo)),
                        "hi": float(np.percentile(F[m, j], hi))} for j in top]
    return rules


def verify(feat: dict, cls: str, rules: dict) -> dict:
    checks, bad = [], []
    for r in rules.get(cls, []):
        v = feat[r["feature"]]
        margin = 0.05 * (r["hi"] - r["lo"] + 1e-9)
        ok = r["lo"] - margin <= v <= r["hi"] + margin
        checks.append({"feature": r["feature"], "value": round(float(v), 4),
                       "expected": [round(r["lo"], 4), round(r["hi"], 4)], "ok": bool(ok)})
        if not ok:
            label = NAMES.get(r["feature"], r["feature"])
            bad.append(f"{label} = {v:.3g}, but {cls} wafers here show {r['lo']:.3g}-{r['hi']:.3g}")
    status = "PASS" if not bad else ("WARN" if len(bad) == 1 else "FAIL")
    return {"status": status, "checks": checks, "reasons": bad}


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> tuple[float, float, float]:
    """Smallest confidence cut-off whose accepted set reaches `target`
    accuracy. Returns (threshold, coverage, accuracy_on_accepted)."""
    order = np.argsort(-conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(acc >= target)
    if len(ok) == 0:
        return 1.01, 0.0, float("nan")
    n = ok.max() + 1
    return float(conf[order][n - 1]), float(n / len(conf)), float(acc[n - 1])


def decide(prob: np.ndarray, verification: dict, feat: dict, rules: dict, policy: dict) -> dict:
    """Final decision for one wafer from its (stacked) probabilities."""
    order = np.argsort(prob)[::-1]
    top, second = CLASSES[order[0]], CLASSES[order[1]]
    final, action, note = top, "human_review", ""
    if verification["status"] == "FAIL" and policy.get("auto_correct", False):
        v2 = verify(feat, second, rules)
        if v2["status"] == "PASS" and prob[order[1]] >= policy.get("correct_min_prob", 0.2):
            final, note = second, f"top class {top} failed morphology checks; runner-up {second} passed"
    conf = float(prob[CLASSES.index(final)])
    if conf >= policy["auto_accept_threshold"] and verification["status"] != "FAIL" and not note:
        action = "auto_accept"
    return {"final_class": final, "action": action, "confidence": round(conf, 5), "note": note or None}
