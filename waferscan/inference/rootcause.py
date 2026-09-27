"""
Root-cause inference: which process problem most likely produced the defect
patterns in a batch?

Input: one row per wafer - its defect class (from the classifier), an order
column (seq / timestamp), equipment columns (chamber, tool...) and numeric
process parameters (etch_time_s, pr_thickness_nm, cmp_pad_cycles, ...).

Evidence (every test is against the clean 'None' wafers, BH-FDR corrected):
  parameter shift   - do Scratch wafers run at higher pad cycles than clean
                      ones? Mann-Whitney U test + standardised difference d.
  chamber matching  - is Edge-Ring concentrated in one chamber? chi-square +
                      per-chamber rate ratio.
  temporal drift    - EWMA control chart on each class's rate over the
                      sequence (overall and per chamber): when did it start?
  consumables       - class rate by quartile of a counter (pad cycles).
  transitions       - which class -> class changes between consecutive
                      wafers on the same equipment happen more than chance.
Scoring: P(cause | class, data) ~ prior(cause | class) * exp(evidence),
evidence = strongest significant effect among the cause's linked params /
factors. Batch contribution = class-frequency-weighted sum -> Pareto.

This ranks *associations*. Confirm the top cause with an engineer / DOE
before acting - correlation is where an investigation starts, not ends.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_kb(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def _max_llr(ind: np.ndarray, n_min: int) -> tuple[float, int]:
    """Best single split of a 0/1 series into two binomial rates (upward
    changes only). Returns (log-likelihood ratio, split index)."""
    n, k = len(ind), ind.sum()
    c = np.cumsum(ind)[n_min - 1:n - n_min]
    n1 = np.arange(n_min, n - n_min + 1)
    k1, n2, k2 = c, n - n1, k - c

    def ll(kk, nn):
        p = np.clip(kk / nn, 1e-12, 1 - 1e-12)
        return kk * np.log(p) + (nn - kk) * np.log(1 - p)

    llr = ll(k1, n1) + ll(k2, n2) - ll(np.array(k), np.array(n))
    llr = np.where(k2 / n2 > k1 / n1, llr, 0.0)
    i = int(np.argmax(llr))
    return float(llr[i]), int(n1[i])


def changepoint(ind: np.ndarray, n_min: int = 30, n_perm: int = 300, seed: int = 0) -> dict | None:
    """Did this class's rate step UP somewhere in the sequence? Binomial
    likelihood-ratio changepoint; p-value by shuffling the order (if the
    order doesn't matter, shuffling shouldn't change the best split)."""
    ind = np.asarray(ind, float)
    if len(ind) < 3 * n_min or ind.sum() < 5:
        return None
    stat, t = _max_llr(ind, n_min)
    rng = np.random.default_rng(seed)
    null = np.array([_max_llr(rng.permutation(ind), n_min)[0] for _ in range(n_perm)])
    return {"onset_index": t, "p": float((1 + (null >= stat).sum()) / (1 + n_perm)),
            "rate_before": round(float(ind[:t].mean()), 4), "rate_after": round(float(ind[t:].mean()), 4)}


def analyze(df: pd.DataFrame, kb: dict, class_col: str = "class", seq_col: str = "seq",
            factors: list[str] | None = None, params: list[str] | None = None,
            baseline: str = "None", alpha: float = 0.05, min_count: int = 5, beta: float = 1.0) -> dict:
    df = df.sort_values(seq_col).reset_index(drop=True)
    if factors is None:
        factors = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                   and c not in (class_col, "wafer_id", "lot_id") and df[c].nunique() <= 50]
    if params is None:
        params = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != seq_col
                  and df[c].nunique() > 5]
    counts = df[class_col].value_counts()
    classes = [c for c, n in counts.items() if c != baseline and n >= min_count]
    base = df[df[class_col] == baseline]

    # --- parameter shifts ---------------------------------------------------
    rows = []
    for c in classes:
        sub = df[df[class_col] == c]
        for j in params:
            a, b = sub[j].dropna().to_numpy(float), base[j].dropna().to_numpy(float)
            if len(a) < min_count or len(b) < min_count or np.std(b) == 0:
                continue
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            rows.append({"class": c, "param": j, "d": (a.mean() - b.mean()) / b.std(), "p": p,
                         "mean_class": a.mean(), "mean_baseline": b.mean()})
    q = _bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = qq
    param_effects = [{**{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()},
                      "significant": bool(r["q"] < alpha)} for r in rows]

    # --- equipment (chamber matching) ----------------------------------------
    frows = []
    for f in factors:
        for c in classes:
            tab = pd.crosstab(df[f], df[class_col] == c)
            if tab.shape != (tab.shape[0], 2) or tab.shape[0] < 2:
                continue
            p = stats.chi2_contingency(tab.to_numpy())[1]
            rate = tab[True] / tab.sum(axis=1)
            overall = (df[class_col] == c).mean()
            worst = rate.idxmax()
            frows.append({"class": c, "factor": f, "worst_level": str(worst), "p": p,
                          "rate_worst": float(rate.max()), "rate_overall": float(overall),
                          "rate_ratio": float(rate.max() / max(overall, 1e-9)),
                          "rates": {str(k): round(float(v), 4) for k, v in rate.items()}})
    for r, qq in zip(frows, _bh([r["p"] for r in frows])):
        r["q"] = qq
        r["significant"] = bool(qq < alpha)

    # --- temporal drift (rate step-up), overall and per equipment -------------
    drift = []
    for c in classes:
        scopes = [("all", df)] + [(f"{f}={lvl}", sub) for f in factors for lvl, sub in df.groupby(f)]
        for scope, sub in scopes:
            res = changepoint((sub[class_col] == c).to_numpy(float))
            if res:
                i = res.pop("onset_index")
                drift.append({"class": c, "scope": scope, "onset_seq": int(sub[seq_col].iloc[i]),
                              "onset_wafer": str(sub["wafer_id"].iloc[i]) if "wafer_id" in sub else None, **res})
    for r, qq in zip(drift, _bh([r["p"] for r in drift])):
        r["q"] = round(float(qq), 5)
    drift = [r for r in drift if r["q"] < alpha]

    # --- consumables: class rate by counter quartile -------------------------
    consumables = []
    for j in params:
        if not any(k in j for k in ("cycle", "count", "hours", "wafers_since")):
            continue
        qs = pd.qcut(df[j], 4, duplicates="drop")
        for c in classes:
            rates = (df[class_col] == c).groupby(qs, observed=True).mean()
            r, p = stats.pointbiserialr((df[class_col] == c).astype(int), df[j])
            if p < alpha and abs(r) > 0.05:
                consumables.append({"class": c, "counter": j, "r": round(float(r), 4), "p": float(p),
                                    "rate_by_quartile": {str(k): round(float(v), 4) for k, v in rates.items()}})

    # --- class transitions on the same equipment -----------------------------
    trans = []
    for f in factors[:1]:
        pairs = pd.concat([pd.DataFrame({"a": s[class_col].to_numpy()[:-1], "b": s[class_col].to_numpy()[1:]})
                           for _, s in df.groupby(f) if len(s) > 1], ignore_index=True)
        if len(pairs):
            tab = pd.crosstab(pairs["a"], pairs["b"])
            pb = pairs["b"].value_counts(normalize=True)
            for a_ in tab.index:
                for b_ in tab.columns:
                    n_ab = int(tab.loc[a_, b_])
                    exp = tab.loc[a_].sum() * pb[b_]
                    if b_ != baseline and n_ab >= 5 and n_ab / exp > 1.5:
                        trans.append({"from": a_, "to": b_, "count": n_ab, "ratio_vs_chance": round(float(n_ab / exp), 2)})
    trans.sort(key=lambda t: -t["ratio_vs_chance"])

    # --- score causes -----------------------------------------------------------
    causes, priors = kb["causes"], kb["priors"]
    n_def = sum(counts[c] for c in classes)
    per_class, contrib, drill = {}, {}, {}
    for c in classes:
        pri = priors.get(c, {})
        if not pri:
            continue
        scores = {}
        for k, p0 in pri.items():
            spec = causes.get(k, {})
            ev, why = 0.0, []
            for r in param_effects:
                if r["class"] == c and r["param"] in spec.get("params", []) and r["significant"]:
                    if abs(r["d"]) > ev:
                        ev = abs(r["d"])
                    why.append(f"{r['param']}: {r['d']:+.2f} sd vs clean wafers (q={r['q']:.1e})")
            for r in frows:
                if r["class"] == c and r["factor"] in spec.get("factors", []) and r["significant"]:
                    e = math.log(max(r["rate_ratio"], 1.0)) * 2
                    ev = max(ev, e)
                    why.append(f"{r['factor']}={r['worst_level']}: {c} rate {r['rate_worst']:.1%} "
                               f"vs {r['rate_overall']:.1%} overall (q={r['q']:.1e})")
            scores[k] = (p0 * math.exp(beta * min(ev, 6.0)), why)
        z = sum(s for s, _ in scores.values())
        per_class[c] = {k: round(s / z, 4) for k, (s, _) in sorted(scores.items(), key=lambda kv: -kv[1][0])}
        w = counts[c] / n_def
        for k, (s, why) in scores.items():
            contrib[k] = contrib.get(k, 0.0) + w * s / z
            drill.setdefault(k, {"classes": {}, "evidence": []})
            drill[k]["classes"][c] = round(s / z, 4)
            drill[k]["evidence"] += [f"[{c}] {x}" for x in why]

    pareto, cum = [], 0.0
    for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]):
        cum += v
        pareto.append({"cause": k, "label": causes.get(k, {}).get("label", k), "contribution": round(v, 4),
                       "cumulative": round(cum, 4), **drill[k]})
    return {
        "n_wafers": int(len(df)), "defective_wafers": int(n_def),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "pareto": pareto, "per_class": per_class,
        "parameter_effects": [r for r in param_effects if r["significant"]],
        "equipment_effects": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                              for r in frows if r["significant"]],
        "drift": drift, "consumables": consumables, "transitions": trans[:10],
        "method": "P(cause|class,data) ~ prior(cause|class) * exp(strongest significant evidence); "
                  "tests vs clean wafers with Benjamini-Hochberg FDR control",
        "caveat": "Associations, not proven causes. Confirm with process engineering / DOE.",
    }
