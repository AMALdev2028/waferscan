"""End-to-end API tests against the real model bundle (skipped if not built yet)."""
import io
import os

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

BUNDLE = os.environ.get("WAFERSCAN_BUNDLE", "model_bundle")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE, "manifest.json")),
                                reason="model bundle not built (make train stack export)")


@pytest.fixture(scope="module")
def client():
    from waferscan.api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def sample(client):
    return client.get("/api/v1/samples/0").json()


def test_health_and_card(client):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and len(h["classes"]) == 9
    card = client.get("/api/v1/model/card").json()
    assert card["metrics"]["stacked"]["n"] > 0 and "routing_oof" in card and "curves" in card


def test_predict_full_response(client, sample):
    r = client.post("/api/v1/predict", json={"wafer_map": sample["wafer_map"], "mode": "full"})
    assert r.status_code == 200, r.text
    j = r.json()
    p = j["prediction"]
    assert p["class"] in p["probabilities"] and abs(sum(p["probabilities"].values()) - 1) < 1e-3
    assert j["decision"]["action"] in ("auto_accept", "human_review")
    assert j["verification"]["status"] in ("PASS", "WARN", "FAIL")
    assert j["uncertainty"]["members_used"] >= 1
    q = j["quantification"]
    assert q["wafer"]["dies_total"] > 0 and 0 <= q["pattern"]["affected_pct"] <= 100
    assert np.asarray(j["cam"]).shape == np.asarray(sample["wafer_map"]).shape


def test_modes_and_tta(client, sample):
    fast = client.post("/api/v1/predict", json={"wafer_map": sample["wafer_map"], "mode": "fast"}).json()
    tta = client.post("/api/v1/predict", json={"wafer_map": sample["wafer_map"], "mode": "full", "tta": True}).json()
    assert fast["prediction"]["path"] == "fast" and tta["uncertainty"]["tta_transforms"] == 8


def test_input_validation(client):
    assert client.post("/api/v1/predict", json={"wafer_map": [[0, 1], [5, 1]]}).status_code == 422
    assert client.post("/api/v1/predict", json={"wafer_map": [[0, 1], [1]]}).status_code == 422
    assert client.post("/api/v1/predict", json={"wafer_map": [[0, 0], [0, 0]]}).status_code == 422


def test_image_upload_rendered_map(client, sample):
    from waferscan.data.images import PALETTE
    dm = np.asarray(sample["wafer_map"], np.uint8)
    img = Image.fromarray(PALETTE[dm].astype(np.uint8)).resize((640, 640), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = client.post("/api/v1/predict/image", files={"file": ("w.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input_kind"] == "rendered_map"
    assert j["wafer"]["rows"] == dm.shape[0] and j["wafer"]["cols"] == dm.shape[1]
    bad = client.post("/api/v1/predict/image", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 422


def test_batch_with_process_logs_and_report(client):
    from waferscan.data.synthetic import simulate_fab
    df, maps = simulate_fab(n_lots=2, wafers_per_lot=20, with_maps=True)
    wafers = [{"wafer_id": w, "lot_id": l, "wafer_map": m.tolist()} for w, l, m in zip(df.wafer_id, df.lot_id, maps)]
    proc = df.drop(columns=["true_class"]).to_dict("records")
    r = client.post("/api/v1/batch", json={"wafers": wafers, "process": proc})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["wafers"] == 40 and j["rootcause"]["n_wafers"] == 40
    rep = client.get(j["report_url"])
    assert rep.status_code == 200 and "BATCH" in rep.text and "data:image/png;base64" in rep.text


def test_rootcause_demo(client):
    j = client.get("/api/v1/rootcause/demo").json()
    assert j["pareto"][0]["contribution"] > 0 and j["per_class"]["Edge-Ring"]


def test_landing_page_served(client):
    r = client.get("/site/")
    assert r.status_code == 200 and "WAFER<br>DEFECT" in r.text and 'src="app.js"' in r.text
    assert "<script>" not in r.text                      # no inline script (Windows Defender rule)


def test_triton_backend_speaks_kserve_v2(sample):
    """Mock Triton (KServe v2 JSON protocol) backed by the real ONNX models."""
    import onnxruntime as ort
    from waferscan.inference.predictor import OnnxBackend, Predictor, TritonBackend
    sessions = {k: ort.InferenceSession(os.path.join(BUNDLE, f"{k}.onnx")) for k in ("fast", "ensemble")}
    mock = FastAPI()

    @mock.get("/v2/health/ready")
    def ready():
        return {}

    @mock.post("/v2/models/{name}/infer")
    async def infer(name: str, req: Request):
        body = await req.json()
        inp = body["inputs"][0]
        x = np.asarray(inp["data"], np.float32).reshape(inp["shape"])
        outs = sessions[name.removeprefix("waferscan_")].run(None, {"input": x})
        names = ["logits", "evid", "g", "fmap"]
        return {"outputs": [{"name": n, "datatype": "FP32", "shape": list(o.shape), "data": o.ravel().tolist()}
                            for n, o in zip(names, outs)]}

    triton = Predictor(BUNDLE, backend=TritonBackend("http://mock", suffix="", client=TestClient(mock)))
    local = Predictor(BUNDLE, backend=OnnxBackend(BUNDLE, ["ensemble"]))
    dm = np.asarray(sample["wafer_map"], np.uint8)
    a, b = triton.predict(dm, mode="full"), local.predict(dm, mode="full")
    assert a["prediction"]["class"] == b["prediction"]["class"]
    assert abs(a["prediction"]["confidence"] - b["prediction"]["confidence"]) < 1e-5
    assert triton.backend.ready()


def test_public_deploy_limits(client):
    one = [[1] * 100 for _ in range(100)]
    too_many = client.post("/api/v1/batch", json={"wafers": [{"wafer_map": [[1]]}] * 257})
    assert too_many.status_code == 422                       # rejected in validation, before any inference
    too_big = client.post("/api/v1/batch", json={"wafers": [{"wafer_map": one}] * 201})   # 2.01 M dies
    assert too_big.status_code == 422 and "limit" in too_big.text
    huge = client.post("/api/v1/predict", content=b"x", headers={"content-length": str(64 * 2**20),
                                                                  "content-type": "application/json"})
    assert huge.status_code == 413
