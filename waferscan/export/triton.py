"""
Assemble a Triton model repository from a model bundle.

    python -m waferscan.export.triton --bundle model_bundle --out build/triton

build/triton/onnx/<model>/{config.pbtxt, 1/model.onnx}   served by ONNX Runtime backend
build/triton/trt/<model>_trt/config.pbtxt                  TensorRT configs; the plan files
                                                           are built on the target GPU
Model names: waferscan_fast, waferscan_ensemble, waferscan_ensemble_run<r> (extra backbones).
Shapes are read from the ONNX graphs themselves, so configs always match the models.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import onnx


def _dims(path: str) -> dict:
    m = onnx.load(path, load_external_data=False)
    # dim_value 0 = symbolic (dynamic) dimension; Triton's config.pbtxt spells that -1
    return {o.name: [d.dim_value or -1 for d in o.type.tensor_type.shape.dim][1:] for o in m.graph.output}


def config(name: str, platform: str, dims: dict) -> str:
    fmt = lambda d: "[ " + ", ".join(map(str, d)) + " ]"  # noqa: E731
    outs = ",\n".join(f'  {{ name: "{n}" data_type: TYPE_FP32 dims: {fmt(d)} }}' for n, d in dims.items())
    return f"""name: "{name}"
platform: "{platform}"
max_batch_size: 64
input [
  {{ name: "input" data_type: TYPE_FP32 dims: [ 2, 64, 64 ] }}
]
output [
{outs}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 32 ]
  max_queue_delay_microseconds: 2000
}}
instance_group [ {{ count: 1 kind: KIND_GPU }} ]
"""


def assemble(bundle: str, out: str) -> list[str]:
    man = json.load(open(os.path.join(bundle, "manifest.json")))
    models = {"waferscan_fast": "fast.onnx"}
    models.update({f"waferscan_{r['key']}": r["file"] for r in man["runs"]})
    shutil.rmtree(out, ignore_errors=True)
    for name, fname in models.items():
        src = os.path.join(bundle, fname)
        dims = _dims(src)
        d = os.path.join(out, "onnx", name, "1")
        os.makedirs(d)
        shutil.copy(src, os.path.join(d, "model.onnx"))
        open(os.path.join(out, "onnx", name, "config.pbtxt"), "w").write(config(name, "onnxruntime_onnx", dims))
        t = os.path.join(out, "trt", f"{name}_trt")
        os.makedirs(os.path.join(t, "1"))
        open(os.path.join(t, "config.pbtxt"), "w").write(config(f"{name}_trt", "tensorrt_plan", dims))
    return list(models)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="model_bundle")
    ap.add_argument("--out", default="build/triton")
    a = ap.parse_args()
    print("triton models:", assemble(a.bundle, a.out))
