# NVIDIA Triton Inference Server with the WaferScan models baked in.
#   python -m waferscan.export.triton --bundle model_bundle --out build/triton   (or: make triton-repo)
#   docker build -f deploy/triton/Dockerfile -t waferscan-triton .
#   docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 waferscan-triton
#
# TensorRT engines are specific to the GPU architecture they were built on, so
# they are NOT baked in. Their configs live in /trt; on the target GPU build the
# plans with deploy/tensorrt/build_engines.sh (the k8s Triton Deployment's
# initContainers do this) and add that folder as a second --model-repository.
# Until then Triton serves the ONNX Runtime models.
ARG TRITON_TAG=24.08-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}
COPY build/triton/onnx /models
COPY build/triton/trt /trt
CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]
