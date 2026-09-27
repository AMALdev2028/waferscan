"""Quantification, root cause, metrics, losses, model math, policy."""
import numpy as np
import torch

from waferscan.classes import CLASSES, canonical
from waferscan.data.synthetic import make_wafer, simulate_fab
from waferscan.evaluation.metrics import ece, full_report, youden_thresholds
from waferscan.inference.policy import dirichlet, mc_dropout, pick_threshold, verify
from waferscan.inference.quantify import quantify, quantify_batch, wilson
from waferscan.inference.rootcause import analyze, load_kb
from waferscan.models.wafernet import Ensemble, WaferNet
from waferscan.training.losses import class_balanced_weights, evidential_loss, focal_loss, gamma_at


# ---------------------------------------------------------------- quantification
def _block_wafer():
    n, c = 21, 10
    yy, xx = np.mgrid[:n, :n]
    dm = np.where(np.hypot(xx - c, yy - c) <= 10, 1, 0).astype(np.uint8)
    dm[4:7, 12:15] = 2          # a 3x3 defect cluster
    dm[15, 3] = dm[10, 18] = 2  # two isolated fails (background noise)
    return dm


def test_quantify_known_geometry():
    q = quantify(_block_wafer(), "Loc", die_w_mm=10, die_h_mm=10)
    p = q["pattern"]
    assert q["wafer"]["dies_total"] == 317 and q["wafer"]["dies_failed"] == 11
    assert p["dies"] == 9 and p["background_fail_dies"] == 2
    assert p["area_mm2"] == 900.0
    assert p["centroid_die"] == [5.0, 13.0] and p["centroid_mm"] == [30.0, 50.0]
    assert sorted(p["regions"][0]["polygon_mm"]) == [[15.0, 35.0], [15.0, 65.0], [45.0, 35.0], [45.0, 65.0]]
    assert abs(p["affected_pct"] + p["clean_pct"] - 100) < 1e-9
    lo, hi = p["affected_pct_wilson95"]
    assert lo < p["affected_pct"] < hi


def test_quantify_whole_wafer_classes_and_none():
    dm = make_wafer("Random", np.random.default_rng(0))
    q = quantify(dm, "Random")
    assert q["pattern"]["dies"] == q["wafer"]["dies_failed"]
    assert quantify(dm, "None")["pattern"]["dies"] == 0


def test_wilson_matches_reference_values():
    lo, hi = wilson(10, 100)                     # textbook: 0.0552 - 0.1744
    assert abs(lo - 0.0552) < 1e-3 and abs(hi - 0.1744) < 1e-3
    assert wilson(0, 0) == [0.0, 0.0]


def test_batch_pools_counts():
    rng = np.random.default_rng(1)
    maps = [make_wafer(c, rng) for c in ("Center", "None", "Edge-Ring")]
    res = [quantify(m, c) for m, c in zip(maps, ("Center", "None", "Edge-Ring"))]
    b = quantify_batch(maps, ["Center", "None", "Edge-Ring"], res)
    assert b["dies_failed"] == sum(r["wafer"]["dies_failed"] for r in res)
    assert np.asarray(b["heatmap"]).shape == (64, 64)


# ---------------------------------------------------------------- root cause
def test_rootcause_recovers_planted_causes():
    df = simulate_fab().rename(columns={"true_class": "class"})
    r = analyze(df, load_kb("configs/rootcause_kb.yaml"))
    top = {c: next(iter(v)) for c, v in r["per_class"].items()}
    assert top["Edge-Ring"] == "etch_edge_nonuniformity"
    assert top["Scratch"] == "cmp_pad_wear"
    assert top["Center"] == "deposition_temperature"
    eq = {(e["class"], e["worst_level"]) for e in r["equipment_effects"]}
    assert ("Edge-Ring", "B") in eq
    assert any(d["class"] == "Edge-Ring" and d["scope"] == "chamber=B" and d["onset_seq"] > 500 for d in r["drift"])
    assert abs(r["pareto"][-1]["cumulative"] - 1.0) < 1e-3


def test_rootcause_no_false_drift_on_stationary_data():
    rng = np.random.default_rng(0)
    import pandas as pd
    df = pd.DataFrame({"seq": np.arange(1500), "chamber": rng.choice(list("ABC"), 1500),
                       "class": rng.choice(["None", "Loc", "Scratch"], 1500, p=[0.9, 0.05, 0.05]),
                       "etch_time_s": rng.normal(62, 1, 1500)})
    r = analyze(df, load_kb("configs/rootcause_kb.yaml"))
    assert r["drift"] == [] and r["equipment_effects"] == []


# ---------------------------------------------------------------- metrics
def test_metrics_suite():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 900)
    p = np.full((900, 9), 0.01)
    p[np.arange(900), y] = 0.92                  # confident and right
    rep = full_report(y, p, CLASSES, groups=np.arange(900) // 3)
    assert rep["accuracy"] == 1.0 and rep["mcc"] == 1.0 and rep["cohen_kappa"] == 1.0
    assert rep["top2_accuracy"] == 1.0
    assert abs(rep["ece"] - 0.08) < 1e-6         # says 92%, is right 100%
    assert rep["ci95"]["accuracy"] == [1.0, 1.0]
    for c in CLASSES:
        assert 0.01 < rep["per_class"][c]["youden_threshold"] <= 0.92


def test_ece_zero_when_calibrated():
    y = np.array([0] * 80 + [1] * 20)
    p = np.tile([0.8, 0.2], (100, 1))            # always 80% on class 0, right 80% of the time
    assert ece(y, p)[0] < 1e-9


def test_youden_picks_separating_threshold():
    y = np.array([0] * 50 + [1] * 50)
    s = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    t = youden_thresholds(y, np.column_stack([1 - s, s]))[1]
    assert 0.4 < t <= 0.6


# ---------------------------------------------------------------- losses
def test_focal_gamma0_no_smoothing_equals_cross_entropy():
    logits, y = torch.randn(16, 9), torch.randint(0, 9, (16,))
    ref = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(focal_loss(logits, y, gamma=0.0, smoothing=0.0), ref, atol=1e-6)


def test_focal_downweights_easy_examples():
    logits = torch.tensor([[8.0] + [0.0] * 8])
    y = torch.tensor([0])
    assert focal_loss(logits, y, 2.0, smoothing=0.0) < 0.01 * focal_loss(logits, y, 0.0, smoothing=0.0)


def test_class_balanced_weights_favour_rare_classes():
    w = class_balanced_weights([147000, 149, 5000], beta=0.999)
    assert w[1] > w[2] > w[0] and abs(float(w.sum()) - 3) < 1e-5


def test_gamma_schedule_and_evidential_loss():
    assert gamma_at(0.0) == 0.0 and gamma_at(0.25) == 1.0 and gamma_at(0.9) == 2.0
    y = torch.tensor([2])
    weak = torch.zeros(1, 9)
    strong = torch.zeros(1, 9)
    strong[0, 2] = 6.0
    assert evidential_loss(strong, y, 1.0) < evidential_loss(weak, y, 1.0)


# ---------------------------------------------------------------- model math
def test_cam_equals_logits_through_gap():
    """CAM correctness: mean over space of (W . F) + b must equal the logits."""
    m = WaferNet().eval()
    x = torch.rand(2, 2, 64, 64)
    with torch.no_grad():
        logits, evid, g, f = m(x)
        cam_logits = torch.einsum("ck,bkhw->bchw", m.head.fc.weight, f).mean((2, 3)) + m.head.fc.bias
    assert torch.allclose(cam_logits, logits, atol=1e-5)
    assert torch.allclose(f.mean((2, 3)), g, atol=1e-6)


def test_ensemble_stacks_members():
    ms = [WaferNet().eval() for _ in range(3)]
    with torch.no_grad():
        out = Ensemble(ms)(torch.rand(4, 2, 64, 64))
    assert out[0].shape == (4, 3, 9) and out[3].shape[:3] == (4, 3, 160)


# ---------------------------------------------------------------- policy
def test_mc_dropout_and_dirichlet():
    rng = np.random.default_rng(0)
    g, W, b = rng.random((5, 16)), rng.normal(size=(9, 16)), np.zeros(9)
    r = mc_dropout(g, W, b, p=0.3, T=40)
    assert np.allclose(r["probs"].sum(1), 1) and (r["mutual_info"] >= -1e-9).all()
    d = dirichlet(np.array([[10.0] + [-10.0] * 8]))
    assert d["probs"][0, 0] > 0.5 and d["u"][0] < 0.9


def test_verify_flags_inconsistent_morphology():
    rules = {"Edge-Ring": [{"feature": "edge_coverage", "lo": 0.35, "hi": 1.0},
                           {"feature": "edge_contrast", "lo": 0.1, "hi": 0.9}]}
    ok = verify({"edge_coverage": 0.6, "edge_contrast": 0.3}, "Edge-Ring", rules)
    bad = verify({"edge_coverage": 0.05, "edge_contrast": -0.1}, "Edge-Ring", rules)
    assert ok["status"] == "PASS" and bad["status"] == "FAIL" and len(bad["reasons"]) == 2


def test_pick_threshold_reaches_target():
    conf = np.linspace(0, 1, 1000)
    correct = conf > 0.3                          # errors only at low confidence
    t, cov, acc = pick_threshold(conf, correct, 0.995)
    assert acc >= 0.995 and 0.25 < t < 0.35 and cov > 0.65


def test_class_aliases():
    assert canonical("Local") == "Loc" and canonical("none") == "None" and canonical("near full") == "Near-full"


# ---------------------------------------------------------------- calibration
def test_temperature_scaling_recovers_known_temperature_and_fixes_ece():
    from waferscan.inference.policy import softmax
    from waferscan.training.stack import fit_temperature
    rng = np.random.default_rng(0)
    true_t = 0.4                                   # the logits are "too flat" by 2.5x, like our label-smoothed CNN
    z = rng.normal(0, 1.5, (4000, 9))
    y = np.array([rng.choice(9, p=p) for p in softmax(z / true_t)])
    t = fit_temperature(z, y)
    assert abs(t - true_t) < 0.05, t
    assert ece(y, softmax(z / t))[0] < ece(y, softmax(z))[0] / 3
    assert (softmax(z / t).argmax(1) == z.argmax(1)).all()      # accuracy can't change
