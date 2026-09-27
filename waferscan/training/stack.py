"""
Post-training calibration: turn K fold models into a deployable, honest system.

    python -m waferscan.training.stack --runs runs/cpu_wafernet --data data/processed/rendered_wm811k.npz \
        --bundle model_bundle --leaky runs/leaky_random_split

Everything here is fitted on OUT-OF-FOLD predictions (each wafer scored by
the fold model that never trained on it):
1. uncertainty: MC dropout (per fold head) + evidential Dirichlet
2. base model #2: gradient boosting on handcrafted morphology features
3. stacking: multinomial logistic regression on [CNN log-probs, GBM
   log-probs, evidential uncertainty] - its coefficients are the learned
   meta-weights. Evaluated with its own grouped CV (nested), never in-sample.
4. verifier rules, Youden thresholds, auto-accept threshold, fast-path threshold
5. model card with every metric + the leaky-vs-grouped comparison
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import joblib
import numpy as np
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from waferscan.classes import CLASSES
from waferscan.evaluation.metrics import full_report, youden_thresholds
from waferscan.features.morphology import FEATURE_NAMES, feature_matrix, features
from waferscan.inference.policy import decide, dirichlet, fit_verifier, mc_dropout, pick_threshold, softmax, verify

AUTO_ACCEPT_TARGET = 0.995     # accuracy we demand on wafers the system decides alone
FAST_PATH_TARGET = 0.998       # stricter: the fast path skips the ensemble


def load_heads(run_dir: str) -> tuple[list[dict], float]:
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    heads = []
    for k in range(cfg["train"]["folds"]):
        p = os.path.join(run_dir, f"fold{k}.pt")
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=True)
        heads.append({"fold": k, **{n: sd[f"head.{n}"].numpy() for n in
                                    ("fc.weight", "fc.bias", "evid.weight", "evid.bias")}})
    return heads, float(cfg["model"].get("dropout", 0.3))


def run_uncertainty(run_dir: str) -> dict:
    """MC dropout + evidential stats for every OOF wafer of one run."""
    o = np.load(os.path.join(run_dir, "oof.npz"))
    heads, p = load_heads(run_dir)
    n, c = o["logits"].shape
    out = {"p_soft": softmax(o["logits"]), "p_mc": np.zeros((n, c)), "mi": np.zeros(n), "fold": o["fold"]}
    for h in heads:
        m = o["fold"] == h["fold"]
        r = mc_dropout(o["g"][m], h["fc.weight"], h["fc.bias"], p, T=30, seed=h["fold"])
        out["p_mc"][m], out["mi"][m] = r["probs"], r["mutual_info"]
    d = dirichlet(o["evid"])
    out["p_evid"], out["u_evid"] = d["probs"], d["u"]
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar T minimising NLL of
    softmax(logits / T). Label smoothing + focal loss leave the raw CNN
    underconfident (ECE ~0.28 while 99% accurate); T < 1 sharpens it back.
    Argmax never changes, so accuracy is untouched."""
    from scipy.optimize import minimize_scalar
    idx = np.arange(len(y))
    nll = lambda lt: -np.log(np.clip(softmax(logits / np.exp(lt))[idx, y], 1e-12, 1)).mean()  # noqa: E731
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def secondary_alarm_rate(p: np.ndarray, y: np.ndarray, fold: np.ndarray) -> float:
    """Mean number of *other* pattern classes per wafer whose probability clears
    its Youden threshold (the alarm load an operator would see), with thresholds
    cross-fitted like everything else. Mirrors Predictor: 'None' never alarms."""
    hits = 0
    for k in np.unique(fold):
        m = fold == k
        h = p[m] >= np.asarray(youden_thresholds(y[~m], p[~m]))
        h[np.arange(m.sum()), p[m].argmax(1)] = False
        h[:, CLASSES.index("None")] = False
        hits += int(h.sum())
    return hits / len(p)


def pr_curve(y: np.ndarray, p: np.ndarray, n: int = 60) -> dict:
    """Micro-averaged precision/recall curve (all classes one-vs-rest), thinned to n points."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    Y = np.eye(p.shape[1])[y]
    prec, rec, _ = precision_recall_curve(Y.ravel(), p.ravel())
    idx = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    o = np.argsort(rec[idx], kind="stable")
    return {"recall": np.round(rec[idx][o], 4).tolist(), "precision": np.round(prec[idx][o], 4).tolist(),
            "ap": float(average_precision_score(Y, p, average="micro"))}


def meta_features(p_cnns: list[np.ndarray], p_gbm: np.ndarray, u_evid: np.ndarray) -> np.ndarray:
    logs = [np.log(np.clip(p, 1e-6, 1)) for p in (*p_cnns, p_gbm)]
    return np.column_stack([*logs, u_evid])


def main(runs: list[str], data: str, bundle: str, leaky: str | None, seed: int = 42):
    d = np.load(data)
    X, y, groups = d["X"].astype(np.float32), d["y"], d["groups"]
    os.makedirs(bundle, exist_ok=True)

    unc = [run_uncertainty(r) for r in runs]
    fold = unc[0]["fold"]
    assert (fold >= 0).all(), "stacking needs OOF predictions for every wafer (run all folds)"

    feat_cache = os.path.join(runs[0], "features.npy")
    F = np.load(feat_cache) if os.path.exists(feat_cache) else feature_matrix(X)
    np.save(feat_cache, F)

    # base model #2: GBM on morphology features, same folds as the CNN
    gbm_params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
    p_gbm = np.zeros((len(y), len(CLASSES)))
    for k in np.unique(fold):
        m = fold == k
        p_gbm[m] = HistGradientBoostingClassifier(**gbm_params).fit(F[~m], y[~m]).predict_proba(F[m])

    u_evid = np.mean([u["u_evid"] for u in unc], 0)
    Z = meta_features([u["p_mc"] for u in unc], p_gbm, u_evid)
    meta = LogisticRegression(C=1.0, max_iter=5000)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed + 1)
    p_stack = cross_val_predict(meta, Z, y, groups=groups, cv=cv, method="predict_proba")

    names = [yaml.safe_load(open(os.path.join(r, "config.yaml")))["model"]["backbone"] for r in runs]
    reports = {}
    for name, u in zip(names, unc):                     # the backbone comparison, all out-of-fold
        reports[f"cnn_{name}"] = full_report(y, u["p_soft"], CLASSES, groups)
        reports[f"cnn_{name}_mc_dropout"] = full_report(y, u["p_mc"], CLASSES)
        reports[f"cnn_{name}_evidential"] = full_report(y, u["p_evid"], CLASSES)
    reports["morphology_gbm"] = full_report(y, p_gbm, CLASSES, groups)
    reports["stacked"] = full_report(y, p_stack, CLASSES, groups)

    # verification + routing. Every REPORTED number is cross-fitted: fold k is
    # judged by verifier rules / thresholds / temperature learned on the other
    # folds. The DEPLOYED rules / thresholds / temperature are then fitted on all
    # OOF wafers. (Scoring rules on the wafers they were fitted to would flatter
    # the "accuracy of auto-accepted wafers" we promise the fab.)
    feats = [dict(zip(FEATURE_NAMES, row)) for row in F]
    folds = np.unique(fold)
    rules_cf = {k: fit_verifier(F[fold != k], y[fold != k]) for k in folds}
    rules = fit_verifier(F, y)                                               # deployed
    status_of = lambda cls_idx, rs: np.array([verify(feats[i], CLASSES[j], rs(i))["status"]  # noqa: E731
                                              for i, j in enumerate(cls_idx)])
    pred = p_stack.argmax(1)
    correct = pred == y
    ver = [verify(feats[i], CLASSES[pred[i]], rules_cf[fold[i]]) for i in range(len(y))]
    status = np.array([v["status"] for v in ver])
    status_dep = status_of(pred, lambda i: rules)
    verification_stats = {s: {"n": int((status == s).sum()),
                              "accuracy": float(correct[status == s].mean()) if (status == s).any() else None}
                          for s in ("PASS", "WARN", "FAIL")}

    # does "switch to runner-up when top-1 fails verification" help? measure it.
    policy = {"auto_accept_threshold": 1.01, "auto_correct": True, "correct_min_prob": 0.2}
    corrected = np.array([CLASSES.index(decide(p_stack[i], ver[i], feats[i], rules_cf[fold[i]], policy)["final_class"])
                          for i in range(len(y))])
    policy["auto_correct"] = bool((corrected == y).mean() > correct.mean())

    def routed(conf_cf, conf_dep, ok, pop_cf, pop_dep, target: float):
        """Accept candidates (pop) whose confidence clears a threshold chosen for
        `target` accuracy. Returns (deployed threshold fitted on all OOF candidates,
        cross-fitted accept mask: fold k uses a threshold fitted on the other folds).
        Non-candidates (e.g. verification FAIL) are never accepted."""
        mask = np.zeros(len(ok), bool)
        for k in folds:
            m, sel = fold == k, pop_cf & (fold != k)
            t_k = pick_threshold(conf_cf[sel], ok[sel], target)[0] if sel.any() else 1.01
            mask[m] = pop_cf[m] & (conf_cf[m] >= t_k)
        t_dep = pick_threshold(conf_dep[pop_dep], ok[pop_dep], target)[0] if pop_dep.any() else 1.01
        return t_dep, mask

    # ---- stage 1, the fast path: one fold model, temperature-scaled
    logits0 = np.load(os.path.join(runs[0], "oof.npz"))["logits"]
    soft = np.zeros_like(unc[0]["p_soft"])
    for k in folds:
        m = fold == k
        soft[m] = softmax(logits0[m] / fit_temperature(logits0[~m], y[~m]))
    policy["fast_temperature"] = fit_temperature(logits0, y)
    soft_dep = softmax(logits0 / policy["fast_temperature"])
    reports[f"cnn_{names[0]}_temperature_scaled"] = full_report(y, soft, CLASSES)
    fpred = soft.argmax(1)                              # temperature never changes the argmax
    fcorrect = fpred == y
    fpass_cf = status_of(fpred, lambda i: rules_cf[fold[i]]) == "PASS"
    fpass_dep = status_of(fpred, lambda i: rules) == "PASS"
    fthr, fast = routed(soft.max(1), soft_dep.max(1), fcorrect, fpass_cf, fpass_dep, FAST_PATH_TARGET)
    policy["fast_path_threshold"] = fthr
    fast_dep = fpass_dep & (soft_dep.max(1) >= fthr)

    # ---- stage 2, the full path: in "auto" mode it only ever sees what stage 1
    # passed on - the harder wafers - so its threshold is fitted on exactly those
    conf = p_stack.max(1)

    def stage2(target: float):
        return routed(conf, conf, correct, ~fast & (status != "FAIL"), ~fast_dep & (status_dep != "FAIL"), target)

    thr, full_acc = stage2(AUTO_ACCEPT_TARGET)
    policy["auto_accept_threshold"] = thr
    # Youden thresholds are path-specific: each lives on its own path's probability scale
    policy["youden_thresholds"] = dict(zip(CLASSES, [reports["stacked"]["per_class"][c]["youden_threshold"]
                                                     for c in CLASSES]))
    policy["youden_thresholds_fast"] = dict(zip(CLASSES, [
        reports[f"cnn_{names[0]}_temperature_scaled"]["per_class"][c]["youden_threshold"] for c in CLASSES]))

    # the whole cascade, exactly as Predictor.predict(mode="auto") runs it
    sys_ok = np.where(fast, fcorrect, correct)           # which model's answer each wafer gets

    def cascade(full_mask: np.ndarray) -> dict:
        auto = fast | full_mask
        return {"coverage_auto_accepted": float(auto.mean()),
                "accuracy_auto_accepted": float(sys_ok[auto].mean()) if auto.any() else None,
                "sent_to_human_review": float(1 - auto.mean()),
                "errors_caught_by_review": int((~sys_ok & ~auto).sum()),
                "errors_passed_through": int((~sys_ok & auto).sum())}

    def tradeoff_row(tg: float) -> dict:
        t, m = stage2(tg)
        return {"full_path_target": tg, "full_path_threshold": t, **cascade(m)}

    reach = ~fast
    routing = {
        "note": "the 'auto' cascade as deployed (fast path first, then the stacked ensemble on the rest); "
                "cross-fitted: each fold judged by rules/thresholds/temperature learned on the other folds",
        **cascade(full_acc),
        "fast_path": {"threshold": fthr, "target_accuracy": FAST_PATH_TARGET, "coverage": float(fast.mean()),
                      "accuracy": float(fcorrect[fast].mean()) if fast.any() else None,
                      "temperature": policy["fast_temperature"],
                      "ece_raw_vs_calibrated": [reports[f"cnn_{names[0]}"]["ece"],
                                                reports[f"cnn_{names[0]}_temperature_scaled"]["ece"]]},
        "full_path": {"threshold": thr, "target_accuracy": AUTO_ACCEPT_TARGET, "wafers_reaching": int(reach.sum()),
                      "auto_accepted": int(full_acc.sum()),
                      "accuracy_on_accepted": float(correct[full_acc].mean()) if full_acc.any() else None,
                      "accuracy_of_ensemble_on_these_wafers": float(correct[reach].mean()) if reach.any() else None},
        "secondary_alarms_per_wafer": {"stacked": secondary_alarm_rate(p_stack, y, fold),
                                       "fast": secondary_alarm_rate(soft, y, fold)},
        "auto_correct_enabled": policy["auto_correct"],
        # the whole trade-off, so an engineer can pick the operating point
        "tradeoff": [tradeoff_row(tg) for tg in (0.95, 0.98, 0.99, 0.995, 0.999)],
    }
    np.savez_compressed(os.path.join(runs[0], "oof_stacked.npz"), p_stack=p_stack, p_gbm=p_gbm, status=status, y=y)

    leak = None
    if leaky and os.path.exists(os.path.join(leaky, "summary.json")):
        ls = json.load(open(os.path.join(leaky, "summary.json")))
        g0 = fold == 0
        leak = {"random_split_fold0_accuracy": ls["oof_acc"],
                "grouped_split_fold0_accuracy": float((unc[0]["p_soft"][g0].argmax(1) == y[g0]).mean()),
                "note": "same model, data and epochs; only the split differs. The random split lets copies of "
                        "one wafer sit in both train and validation."}

    # final fits on all OOF data
    gbm = HistGradientBoostingClassifier(**gbm_params).fit(F, y)
    meta.fit(Z, y)
    C = len(CLASSES)
    share = {n: float(np.abs(meta.coef_[:, i * C:(i + 1) * C]).mean()) for i, n in enumerate(names + ["morphology_gbm"])}
    share["evidential_u"] = float(np.abs(meta.coef_[:, -1]).mean())
    tot = sum(share.values())
    meta_weights = {k: round(v / tot, 4) for k, v in share.items()}
    joblib.dump({"gbm": gbm, "meta": meta, "feature_names": FEATURE_NAMES}, os.path.join(bundle, "stack.joblib"))

    heads = {}
    for r_i, r in enumerate(runs):
        hs, p = load_heads(r)
        for h in hs:
            for n in ("fc.weight", "fc.bias", "evid.weight", "evid.bias"):
                heads[f"run{r_i}_m{h['fold']}_{n}"] = h[n]
        heads[f"run{r_i}_dropout"] = np.array(p)
    np.savez(os.path.join(bundle, "heads.npz"), **heads)

    with open(os.path.join(bundle, "policy.json"), "w") as fh:
        json.dump({**policy, "verifier_rules": rules}, fh, indent=2)

    audit_path = os.path.splitext(data)[0] + "_audit.json"
    cfgs = [yaml.safe_load(open(os.path.join(r, "config.yaml"))) for r in runs]
    card = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "classes": CLASSES,
        "trained_on": os.path.basename(data),
        "dataset_audit": json.load(open(audit_path)) if os.path.exists(audit_path) else None,
        "validation_protocol": f"{cfgs[0]['train']['folds']}-fold StratifiedGroupKFold; groups = source wafer "
                               "(duplicate copies and renamed rotated/flipped copies merged); all numbers out-of-fold",
        "base_models": [c["model"]["backbone"] for c in cfgs] + ["morphology_gbm"],
        "meta_weights_note": "multinomial logistic regression over [log p_cnn (MC-dropout mean), log p_gbm, u_evidential]",
        "meta_weight_share": meta_weights,
        "metrics": reports,
        "curves": {"stacked": pr_curve(y, p_stack), "baseline_gbm": pr_curve(y, p_gbm)},
        "verification_oof": verification_stats,
        "routing_oof": routing,
        "leakage_experiment": leak,
        "limitations": [
            ("Trained on a curated, class-balanced subset of WM-811K rendered as images (one dominant 52x52 product). "
             "Accuracy on the full, imbalanced WM-811K or on another fab's products must be measured separately "
             "(notebooks/train_gpu.ipynb)." if "rendered" in os.path.basename(data) else
             "Trained on WM-811K (LSWMD.pkl, labelled wafers, 'none' subsampled). One public fab's data: "
             "re-validate before using on another fab's products."),
            "No model is error-free: WM-811K labels contain ambiguity (e.g. Loc vs Edge-Loc). The routing policy "
            "sends low-confidence or morphology-inconsistent wafers to human review instead of guessing.",
            "Root-cause rankings are statistical associations plus engineering priors, not proof of causation.",
        ],
    }
    with open(os.path.join(bundle, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2)

    # demo wafers for the landing page's NEW SCAN button (native die maps), taken from
    # fold 0's validation set: the fast model (fold-0 member, ~94% of decisions) never trained on them
    from waferscan.data.build import native_map
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(np.flatnonzero((y == c) & (fold == 0)), min(4, int(((y == c) & (fold == 0)).sum())),
                                      replace=False)
                           for c in range(len(CLASSES)) if ((y == c) & (fold == 0)).any()])
    nm = [native_map(d, int(i)) for i in pick]
    hmax, wmax = max(m.shape[0] for m in nm), max(m.shape[1] for m in nm)
    padded = np.zeros((len(nm), hmax, wmax), np.uint8)
    for k, m in enumerate(nm):
        padded[k, :m.shape[0], :m.shape[1]] = m
    np.savez_compressed(os.path.join(bundle, "samples.npz"), maps=padded, shapes=np.array([m.shape for m in nm]),
                        y=y[pick], ids=d["files"][pick])

    s = reports["stacked"]
    print(json.dumps({"cnn_acc": {n: reports[f"cnn_{n}"]["accuracy"] for n in names},
                      "gbm_acc": reports["morphology_gbm"]["accuracy"],
                      "stacked_acc": s["accuracy"], "stacked_macro_f1": s["macro"]["f1"], "ci95": s["ci95"],
                      "ece": s["ece"], "routing": routing, "verification": verification_stats, "leak": leak},
                     indent=2))
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bundle", default="model_bundle")
    ap.add_argument("--leaky")
    a = ap.parse_args()
    main(a.runs, a.data, a.bundle, a.leaky)
