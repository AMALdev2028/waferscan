"""
WaferScan API (FastAPI, async). OpenAPI docs at /docs, spec at /openapi.json.

    uvicorn waferscan.api.main:app --host 0.0.0.0 --port 8080

Endpoints are `async`; CPU-heavy work (inference, image decoding, stats) is
pushed to a worker thread with run_in_threadpool so one slow request never
blocks the event loop for everyone else.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import io
import json
import os
import time
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from waferscan.api.schemas import MAX_DIM, BatchRequest, PredictRequest, RootCauseRequest
from waferscan.classes import canonical

BUNDLE = os.environ.get("WAFERSCAN_BUNDLE", "model_bundle")
KB_PATH = os.environ.get("WAFERSCAN_KB", "configs/rootcause_kb.yaml")
WEB_DIR = os.environ.get("WAFERSCAN_WEB", "web")
MAX_UPLOAD = int(os.environ.get("WAFERSCAN_MAX_UPLOAD_MB", "20")) * 1024 * 1024

@contextlib.asynccontextmanager
async def lifespan(_app):
    # at most N inference jobs at once: on a 0.1 vCPU / 512 MB host, parallel jobs only add memory
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = int(os.environ.get("WAFERSCAN_MAX_JOBS", "8"))
    # load the model in the background: the port opens at once (platform health checks pass)
    # and the first visitor after a cold start doesn't pay the model-loading time
    warmup = asyncio.create_task(run_in_threadpool(predictor))
    yield
    warmup.cancel()


app = FastAPI(title="WaferScan API", version="1.0.0", lifespan=lifespan,
              description="WM-811K wafer-map defect classification with uncertainty, morphological verification, "
                          "affected-area quantification and root-cause inference.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])

_state: dict = {"predictor": None, "kb": None, "batches": collections.OrderedDict(),
                "latency": collections.deque(maxlen=64)}


def predictor():
    if _state["predictor"] is None:
        from waferscan.inference.predictor import Predictor
        _state["predictor"] = Predictor(BUNDLE)
    return _state["predictor"]


def kb():
    if _state["kb"] is None:
        from waferscan.inference.rootcause import load_kb
        _state["kb"] = load_kb(KB_PATH)
    return _state["kb"]


def _predict(dm, **kw):
    r = predictor().predict(dm, **kw)
    _state["latency"].append(r["timing_ms"]["total"])
    return r


MAX_BODY = int(os.environ.get("WAFERSCAN_MAX_BODY_MB", "32")) * 1024 * 1024


@app.middleware("http")
async def security_headers(request, call_next):
    # refuse oversized bodies before JSON parsing (validation limits only apply after parsing)
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": f"request body larger than {MAX_BODY // 2**20} MB"}, status_code=413)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


# ---------------------------------------------------------------------------
# health / model info
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["ops"])
async def health():
    try:
        p = await run_in_threadpool(predictor)
        ready = await run_in_threadpool(p.backend.ready)     # a slow Triton must not stall the event loop
        return {"status": "ok" if ready else "degraded", "backend": p.backend.name,
                "members": sum(r["members"] for r in p.runs), "classes": p.manifest["classes"]}
    except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
        raise HTTPException(503, f"model not loaded: {exc}") from exc


@app.get("/api/v1/model/card", tags=["model"])
async def model_card():
    """Every out-of-fold metric, the dataset/leakage audit, routing policy and export checks."""
    p = await run_in_threadpool(predictor)
    lat_path = os.path.join(BUNDLE, "latency_samples.json")
    lat = json.load(open(lat_path)).get("ensemble_fp32") if os.path.exists(lat_path) else None
    return {**p.card, "manifest": p.manifest, "latency_samples": lat}


@app.get("/api/v1/stats", tags=["ops"])
async def stats():
    lat = list(_state["latency"])
    return {"recent_latency_ms": lat, "p50_ms": float(np.percentile(lat, 50)) if lat else None,
            "p95_ms": float(np.percentile(lat, 95)) if lat else None}


@app.get("/api/v1/samples", tags=["demo"])
async def samples():
    p = await run_in_threadpool(predictor)
    return [{"id": s["id"], "label": s["label"], "rows": s["map"].shape[0], "cols": s["map"].shape[1]}
            for s in p.samples]


@app.get("/api/v1/samples/{sample_id}", tags=["demo"])
async def sample(sample_id: int):
    p = await run_in_threadpool(predictor)
    if not 0 <= sample_id < len(p.samples):
        raise HTTPException(404, "no such sample")
    s = p.samples[sample_id]
    return {"id": s["id"], "label": s["label"], "wafer_map": s["map"].tolist()}


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@app.post("/api/v1/predict", tags=["inference"])
async def predict(req: PredictRequest):
    dm = np.asarray(req.wafer_map, np.uint8)
    try:
        return await run_in_threadpool(_predict, dm, mode=req.mode, tta=req.tta, mc_samples=req.mc_samples,
                                       geometry=req.geometry.model_dump(exclude_none=True) if req.geometry else None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD // 2**20} MB")
    if not data:
        raise HTTPException(400, "empty file")
    return data


def _image_pipeline(data: bytes, kind: str, nm_per_px: float, pitch_px: float | None, geometry: dict, mode: str,
                    min_affected: float = 0.05):
    from waferscan.data.images import looks_rendered, optical_to_diemap, rendered_to_diemap, sem_segment
    try:
        img = Image.open(io.BytesIO(data))          # reads only the header
        if img.width * img.height > 64_000_000:     # check BEFORE decoding pixels (decompression bombs)
            raise ValueError("image larger than 64 megapixels")
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("not a decodable image") from exc
    rgb = np.asarray(img.convert("RGB"))
    if kind == "auto":
        kind = "rendered_map" if looks_rendered(rgb) else "optical_scan"
    if kind == "sem":
        gray = np.asarray(img.convert("L"))
        return {"input_kind": "sem", "sem": sem_segment(gray, nm_per_px)}
    t0 = time.perf_counter()
    if kind == "rendered_map":
        dm, info = rendered_to_diemap(rgb)
        affected = None
    else:
        dm, affected, info = optical_to_diemap(np.asarray(img.convert("L")), pitch_px=pitch_px,
                                              min_affected=min_affected)
    conv_ms = (time.perf_counter() - t0) * 1000
    if max(dm.shape) > MAX_DIM:                     # same limit as JSON maps (e.g. a tiny pitch_px)
        raise ValueError(f"die grid {dm.shape[0]}x{dm.shape[1]} exceeds {MAX_DIM}x{MAX_DIM}; check pitch_px")
    res = _predict(dm, mode=mode, geometry=geometry, die_affected=affected)
    res["input_kind"] = kind
    res["image_to_diemap"] = {**{k: v for k, v in info.items()}, "ms": round(conv_ms, 2)}
    return res


@app.post("/api/v1/predict/image", tags=["inference"])
async def predict_image(file: UploadFile = File(...), kind: str = Form("auto"), nm_per_px: float = Form(5.0),
                        pitch_px: float | None = Form(None, ge=2), wafer_diameter_mm: float = Form(300.0, gt=0, le=450),
                        mode: str = Form("auto"), min_affected: float = Form(0.05, gt=0, lt=1)):
    """Upload a rendered wafer map, a full-wafer optical scan, or an SEM review image.
    kind = auto | rendered_map | optical_scan | sem"""
    if kind not in ("auto", "rendered_map", "optical_scan", "sem") or mode not in ("auto", "fast", "full"):
        raise HTTPException(422, "bad kind/mode")
    data = await _read_upload(file)
    try:
        return await run_in_threadpool(_image_pipeline, data, kind, nm_per_px, pitch_px,
                                       {"wafer_diameter_mm": wafer_diameter_mm}, mode, min_affected)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _batch(req: BatchRequest) -> dict:
    from waferscan.inference.quantify import quantify_batch
    maps = [np.asarray(w.wafer_map, np.uint8) for w in req.wafers]
    geo = req.geometry.model_dump(exclude_none=True) if req.geometry else None
    results = [_predict(m, mode=req.mode, geometry=geo, include_die_map=False) for m in maps]
    classes = [r["prediction"]["class"] for r in results]
    summary = quantify_batch(maps, classes, [r["quantification"] for r in results])
    summary["human_review"] = sum(r["decision"]["action"] == "human_review" for r in results)
    rows = [{"wafer_id": w.wafer_id or f"#{i}", "lot_id": w.lot_id, "class": r["prediction"]["class"],
             "confidence": r["prediction"]["confidence"], "action": r["decision"]["action"],
             "verification": r["verification"]["status"],
             "fail_pct": r["quantification"]["wafer"]["fail_pct"],
             "affected_pct": r["quantification"]["pattern"]["affected_pct"],
             "area_mm2": r["quantification"]["pattern"]["area_mm2"]}
            for i, (w, r) in enumerate(zip(req.wafers, results))]
    rc = None
    if req.process:
        proc = pd.DataFrame(req.process)
        if "wafer_id" not in proc:
            raise ValueError("process rows need a wafer_id column matching the wafers")
        pred = pd.DataFrame({"wafer_id": [r["wafer_id"] for r in rows], "class": classes})
        rc = _rootcause(proc.drop(columns=["class"], errors="ignore").merge(pred, on="wafer_id"))
    from waferscan.api.report import wafer_png
    bid = uuid.uuid4().hex[:12]
    b = _state["batches"]
    # keep only what the report shows: small PNG thumbnails, not full results (a 512x512
    # wafer's result is ~17 MB of Python lists; a thumbnail is a few KB)
    b[bid] = {"summary": summary, "rows": rows, "rootcause": rc, "created": time.time(),
              "thumbs": [wafer_png(m, r["quantification"]) for m, r in zip(maps, results)]}
    while len(b) > 50:            # ponytail: in-memory LRU, one worker per container; Redis once >1 replica
        b.popitem(last=False)
    return {"batch_id": bid, "summary": summary, "wafers": rows, "rootcause": rc,
            "report_url": f"/api/v1/batch/{bid}/report"}


@app.post("/api/v1/batch", tags=["inference"])
async def batch(req: BatchRequest):
    try:
        return await run_in_threadpool(_batch, req)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v1/batch/{batch_id}/report", response_class=HTMLResponse, tags=["reports"])
async def batch_report(batch_id: str):
    from waferscan.api.report import render
    b = _state["batches"].get(batch_id)
    if b is None:
        raise HTTPException(404, "unknown or expired batch id")
    return await run_in_threadpool(render, batch_id, b)


# ---------------------------------------------------------------------------
# root cause
# ---------------------------------------------------------------------------
def _rootcause(df: pd.DataFrame) -> dict:
    from waferscan.inference.rootcause import analyze
    if "class" not in df or "seq" not in df:
        raise ValueError("need at least 'class' and 'seq' columns (plus process parameters / equipment columns)")
    df["class"] = df["class"].map(canonical)
    return analyze(df, kb())


@app.post("/api/v1/rootcause", tags=["root cause"])
async def rootcause(req: RootCauseRequest):
    try:
        return await run_in_threadpool(_rootcause, pd.DataFrame(req.records))
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/v1/rootcause/csv", tags=["root cause"])
async def rootcause_csv(file: UploadFile = File(...)):
    data = await _read_upload(file)
    try:
        df = await run_in_threadpool(pd.read_csv, io.BytesIO(data))
        return await run_in_threadpool(_rootcause, df)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v1/rootcause/demo", tags=["root cause"])
async def rootcause_demo():
    """Root-cause analysis on the synthetic fab with planted causes (see data/synthetic.py)."""
    from waferscan.data.synthetic import simulate_fab
    df = await run_in_threadpool(simulate_fab)
    return await run_in_threadpool(_rootcause, df.rename(columns={"true_class": "class"}))


# ---------------------------------------------------------------------------
# landing page
# ---------------------------------------------------------------------------
if os.path.isdir(WEB_DIR):
    app.mount("/site", StaticFiles(directory=WEB_DIR, html=True), name="site")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/site/") if os.path.isdir(WEB_DIR) else RedirectResponse("/docs")
