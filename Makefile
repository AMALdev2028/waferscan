# WaferScan - common tasks.  `make help`
PY ?= python3
IMAGES ?= ../wafer_defect/backend/dataset
DATA ?= data/processed/rendered_wm811k.npz
REG ?= ghcr.io/your-org
export MLFLOW_DISABLE_AGENT_HINT=1

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:        ## serving + tests (no PyTorch)
	$(PY) -m pip install -e ".[api,dev]"

install-train:  ## + PyTorch/timm/MLflow for training & export
	$(PY) -m pip install -e ".[api,train,dev]"

data:           ## rendered map images -> die maps + leakage audit (IMAGES=<folder with class subfolders>)
	$(PY) -m waferscan.data.build --images $(IMAGES) --out $(DATA)

train:          ## 5-fold grouped CV on CPU (~50 min), then the leaky random-split comparison (~10 min)
	$(PY) -m waferscan.training.train --config configs/cpu_wafernet.yaml
	$(PY) -m waferscan.training.train --config configs/cpu_wafernet.yaml --split random --folds 0 --out runs/leaky_random_split

stack:          ## GBM + stacking + verifier + routing thresholds + model card
	$(PY) -m waferscan.training.stack --runs runs/cpu_wafernet --data $(DATA) --bundle model_bundle --leaky runs/leaky_random_split

export:         ## ONNX (ensemble, fast, INT8 edge) + parity/latency checks
	$(PY) -m waferscan.export.onnx_export --runs runs/cpu_wafernet --data $(DATA) --bundle model_bundle

triton-repo:    ## assemble build/triton from model_bundle
	$(PY) -m waferscan.export.triton --bundle model_bundle --out build/triton

serve:          ## API + landing page on http://localhost:8080
	uvicorn waferscan.api.main:app --host 0.0.0.0 --port 8080

test:           ## full test suite
	$(PY) -m pytest -q

openapi:        ## write docs/openapi.json
	$(PY) -c "import json; from waferscan.api.main import app; json.dump(app.openapi(), open('docs/openapi.json', 'w'), indent=1)"

mlflow-ui:      ## browse local training runs on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

images: triton-repo  ## build all container images
	docker build -t $(REG)/waferscan-api:1.0.0 .
	docker build -f deploy/docker/Dockerfile.train -t $(REG)/waferscan-train:1.0.0 .
	docker build -f deploy/docker/Dockerfile.mlflow -t $(REG)/waferscan-mlflow:1.0.0 .
	docker build -f deploy/triton/Dockerfile -t $(REG)/waferscan-triton:1.0.0 .

k8s:            ## deploy to the current kubectl context
	kubectl apply -k deploy/k8s

.PHONY: help install install-train data train stack export triton-repo serve test openapi mlflow-ui images k8s
