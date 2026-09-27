# WaferScan API + landing page. CPU only (ONNX Runtime, no PyTorch), ~450 MB image, ~300-380 MB RAM.
# Any Docker host (Koyeb, Cloud Run, Railway, a VM, k8s) builds this file from the repo root:
#   docker build -t waferscan-api .   &&   docker run -p 8080:8080 waferscan-api   -> http://localhost:8080
FROM python:3.11-slim AS build
WORKDIR /src
COPY pyproject.toml ./
COPY waferscan ./waferscan
RUN pip install --no-cache-dir --prefix=/install ".[api]"

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /install /usr/local
WORKDIR /app
COPY waferscan ./waferscan
COPY configs ./configs
COPY web ./web
COPY model_bundle ./model_bundle
USER app
# Defaults sized for small hosts (free tiers: 0.1 vCPU / 512 MB). On a bigger machine raise
# WAFERSCAN_THREADS and WAFERSCAN_MAX_JOBS. One worker per container: batch reports live in
# that worker's memory (scale with more containers, not more workers).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WAFERSCAN_BUNDLE=/app/model_bundle \
    WAFERSCAN_KB=/app/configs/rootcause_kb.yaml \
    WAFERSCAN_WEB=/app/web \
    OMP_NUM_THREADS=1 \
    WAFERSCAN_THREADS=1 \
    WAFERSCAN_MAX_JOBS=2 \
    WAFERSCAN_LOW_MEMORY=1 \
    PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health',timeout=4).status==200 else 1)"
# $PORT: most platforms inject the port they route to; 8080 otherwise
CMD ["sh", "-c", "exec uvicorn waferscan.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
