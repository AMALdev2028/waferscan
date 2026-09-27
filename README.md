# WaferScan: wafer-map defect detection you can defend

Classifies WM-811K wafer maps into the 9 standard failure patterns: **Center · Donut · Edge-Loc · Edge-Ring · Loc · Near-full · Random · Scratch · None**. For each wafer it:

- says how sure it is (three independent uncertainty methods)
- double-checks the prediction against the wafer's physical shape
- decides whether a human should look
- measures the damaged area in dies, % and mm²
- ranks likely process root causes from equipment and process logs

It ships with a FastAPI service, the WaferScan landing page wired to live inference (design unchanged), ONNX/INT8/TensorRT exports, Triton + Kubernetes manifests with GPU node affinity, MLflow tracking, and a Colab/Kaggle notebook for the big GPU backbones.

## Results (measured, out-of-fold, no leakage)
Trained on the 6,309-image WM-811K set from the `wafer_defect` repo, recovered to die level. It holds **4,162 independent wafers**; the rest are duplicate copies, which the grouping handles.

| | value |
|---|---|
| Stacked accuracy (5-fold grouped CV) | **99.21%** (95% CI 98.90%–99.49%) |
| Macro-F1 · MCC · Cohen's κ · top-2 | 0.9921 · 0.9911 · 0.9911 · 99.97% |
| Calibration error (ECE), stacked · fast model raw → temperature-scaled | 0.0030 · 0.276 → 0.0025 |
| Decided automatically / accuracy on those (the whole auto cascade, cross-fitted) | **93.8% / 99.73%**. The other 6.2% go to human review and hold 37 of the 53 errors |
| Same model with a *random* (leaky) split, fold 0 | 99.29% vs 98.81% grouped: the leak measured |
| CPU latency per request, 2 cores (auto · full ensemble) | p50 11 ms · 50 ms |
| INT8 edge model vs FP32 | 98.73% vs 98.81% accuracy, 0.67 MB |

Every metric (per class P/R/F1, AUC, Youden thresholds, reliability bins, confusion matrix) is in `model_bundle/model_card.json` and at `GET /api/v1/model/card`.

> **Why not "100%"?** Some WM-811K wafers are genuinely ambiguous, and people drew those labels. So no model is error-free on real data. What this system *does* guarantee is measurable: which wafers it decides alone, at what accuracy, and which it hands to an engineer. See [docs/ARCHITECTURE.md §10](docs/ARCHITECTURE.md).

## Quick start (CPU, ~2 minutes)
The trained model bundle is committed, so no training is needed.

```bash
pip install -e ".[api,dev]"
make serve                     # http://localhost:8080  -> landing page with live inference
make test                      # full test suite
```

Or with Docker: `docker compose up --build` (API on :8080, MLflow on :5000).

```bash
curl -s localhost:8080/api/v1/samples/0 | python -c "import sys,json; print(json.dumps({'wafer_map': json.load(sys.stdin)['wafer_map']}))" > w.json
curl -s -X POST localhost:8080/api/v1/predict -H 'content-type: application/json' -d @w.json | head -c 600
curl -s localhost:8080/api/v1/rootcause/demo | head -c 600
```
API docs: http://localhost:8080/docs · spec: `docs/openapi.json`

## Rebuild everything from the images (CPU, ~1 h)
```bash
pip install -e ".[api,train,dev]"
make data IMAGES=../wafer_defect/backend/dataset   # images -> die maps + leakage audit
make train                                          # 5-fold grouped CV + leaky comparison run
make stack export                                   # calibration, model card, ONNX/INT8, benchmarks
make mlflow-ui                                      # browse the runs
```

## Big backbones on a free GPU
Open `notebooks/train_gpu.ipynb` in **Kaggle** (GPU T4 x2, Internet on, add the *WM-811K wafer map* dataset) or **Colab** (T4). It trains EfficientNet-B7, SE-ResNeXt-101 and Swin-T on the full WM-811K, grouped by lot. It checkpoints every epoch, resumes after disconnects, stacks all backbones, and hands you a `model_bundle.zip` to drop into this repo.

- **Kaggle** (about 30 free GPU hours a week) is the easiest for the full dataset: WM-811K is already hosted there, and *Save Version → Save & Run All* keeps training after you close the tab.
- **Colab** free sessions last at most 12 h and drop when idle, so the notebook checkpoints to Google Drive.
- **Antigravity** is an AI code editor, not a GPU. It can connect a notebook to a Colab runtime through the Colab extension, but run the cells yourself: its agent can't yet use that GPU ([issue #433](https://github.com/googlecolab/colab-vscode/issues/433)).

## Deploy
| Target | How |
|---|---|
| Free public link (Koyeb) | Push to GitHub, then Koyeb → Web Service → your repo → Dockerfile → Free → port 8080 (see below) |
| Live demo from your laptop | `make serve`, then `cloudflared tunnel --url http://localhost:8080` for a temporary public https link |
| Single machine, CPU | `docker compose up --build` |
| Single machine + NVIDIA GPU | `make triton-repo && docker compose --profile gpu up --build` (Triton-backed API on :8081) |
| Kubernetes | `make images` (push them), then `kubectl apply -k deploy/k8s`: API (HPA 2–10), Triton on GPU nodes with TensorRT engines built on-node, MLflow, GPU training Job |
| Offline fab-floor PC | `model_bundle/edge_int8.onnx` + `onnxruntime`, no GPU or network needed |

### Free public link on Koyeb
The root `Dockerfile` is sized for free tiers (0.1 vCPU, 512 MB): the app uses about 300 MB at rest and under 400 MB under load.
1. Create an empty GitHub repo and push this folder to it.
2. On koyeb.com, sign in with GitHub, then **Create Web Service → GitHub →** this repo.
3. Builder: **Dockerfile** (found at the repo root). Instance: **Free**. Exposed port: **8080**, HTTP. Health check path: `/api/v1/health`.
4. Deploy. The first build takes several minutes. Then open `https://<your-app>.koyeb.app`.

The free instance sleeps after 1 hour without visitors and wakes in a few seconds on the next request.

## Repo map
```
waferscan/data        die-map recovery (rendered, optical, SEM), WM-811K loader, leakage audit, synthetic fab
waferscan/features    35 physical morphology features
waferscan/models      WaferNet (CBAM attention) + timm backbones, shared CAM/evidential head
waferscan/training    losses, grouped-CV training (MLflow), stacking + calibration
waferscan/inference   predictor, uncertainty/verification/routing, quantification, root cause
waferscan/export      ONNX, INT8, Triton repository
waferscan/api         FastAPI app, HTML batch report
web/                  landing page (original design, now live)
deploy/               Dockerfiles, Triton, TensorRT, Kubernetes
docs/                 ARCHITECTURE.md (deep dive) · LEARN.md (mentor guide + exercises)
```

New to this? Start with **[docs/LEARN.md](docs/LEARN.md)**.
