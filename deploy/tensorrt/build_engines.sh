#!/usr/bin/env bash
# Build TensorRT FP16 engines for every *_trt model - ON THE TARGET GPU.
# An engine built on a T4 will not load on an A10G: build where you serve.
#
#   python -m waferscan.export.triton --bundle model_bundle --out build/triton
#   docker run --gpus all -v $PWD/build/triton:/r nvcr.io/nvidia/tensorrt:24.08-py3 \
#       bash -c "$(cat deploy/tensorrt/build_engines.sh)" _ /r/onnx /r/trt
#   tritonserver --model-repository=/models --model-repository=<trt repo>
# then point the API at them: TRITON_MODEL_SUFFIX=_trt
# (Use the TensorRT container of the SAME release as your Triton image.)
set -euo pipefail
ONNX_REPO=${1:-/models}    # contains <model>/1/model.onnx
TRT_REPO=${2:-/work}       # contains <model>_trt/config.pbtxt
for d in "$TRT_REPO"/*_trt; do
  m=$(basename "$d"); src=${m%_trt}
  mkdir -p "$d/1"
  trtexec --onnx="$ONNX_REPO/$src/1/model.onnx" --saveEngine="$d/1/model.plan" --fp16 \
          --minShapes=input:1x2x64x64 --optShapes=input:8x2x64x64 --maxShapes=input:64x2x64x64
done
ls -la "$TRT_REPO"/*_trt/1/
