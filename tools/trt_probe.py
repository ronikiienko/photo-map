"""Where does the TensorRT copy of the vision encoder first differ from the CUDA one?

Adds every float16 intermediate tensor of vision_encoder_fp16_trt.onnx as an output, runs both providers on the same input and prints
the first tensors whose mean absolute value differs by more than 1 %, then the rows of the final output that disagree.

    LD_LIBRARY_PATH=runtime/cuda:runtime/cudnn:runtime/tensorrt .venv/bin/python src/trt_probe.py
"""

import ctypes

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper

import config

for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
    ctypes.CDLL(str(config.ROOT / "runtime" / "tensorrt" / lib), mode=ctypes.RTLD_GLOBAL)
D = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"
m = onnx.load(str(D / "vision_encoder_fp16_trt.onnx"), load_external_data=False)
kinds = {v.name: v.type.tensor_type.elem_type for v in onnx.shape_inference.infer_shapes(m).graph.value_info}
picks = [n.output[0] for n in m.graph.node if n.op_type in ("Add", "RMSNormalization") and kinds.get(n.output[0]) == 10]  # not every tensor: the card has 4 GB
for p in picks:
    m.graph.output.append(helper.make_tensor_value_info(p, T.FLOAT16, None))
probe = D / "probe_trt.onnx"
onnx.save(m, str(probe))

rng = np.random.default_rng(1)
grid = np.array([(c, r) for r in range(30) for c in range(42)], dtype=np.int64)
px = np.zeros((1, 1264, 768), np.float32)
px[:, :1260] = rng.random((1, 1260, 768), dtype=np.float32)
pos = np.full((1, 1264, 2), -1, np.int64)
pos[:, :1260] = grid
feed = {"pixel_values": px, "pixel_position_ids": pos}
so = ort.SessionOptions()
so.log_severity_level = 3
shapes = ",".join(f"{k}:1x1264x{d}" for k, d in (("pixel_values", 768), ("pixel_position_ids", 2)))
a = ort.InferenceSession(str(probe), so, providers=[("CUDAExecutionProvider", {})]).run(None, feed)
b = ort.InferenceSession(str(probe), so, providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_max_workspace_size": 1 << 30, "trt_profile_min_shapes": shapes, "trt_profile_opt_shapes": shapes, "trt_profile_max_shapes": shapes}), ("CUDAExecutionProvider", {})]).run(None, feed)
probe.unlink()

shown = 0
for name, x, y in zip(["final"] + picks, a, b):
    x, y = x.astype(np.float32), y.astype(np.float32)
    mx, my = np.abs(x).mean(), np.abs(y).mean()
    if name != "final" and abs(mx - my) > 0.01 * max(mx, 1e-9) and shown < 6:
        print(f"first differences: {name:14s} {str(x.shape):20s} cuda mean {mx:8.4f} max {np.abs(x).max():8.1f} | trt mean {my:8.4f} max {np.abs(y).max():8.1f}")
        shown += 1
fx, fy = a[0].astype(np.float32), b[0].astype(np.float32)
cos = (fx * fy).sum(-1) / (np.linalg.norm(fx, axis=-1) * np.linalg.norm(fy, axis=-1) + 1e-9)
print(f"final output {fx.shape}: rows with cosine < 0.99: {int((cos < 0.99).sum())} of {len(cos)}; zero rows in trt: {int((np.abs(fy).sum(-1) == 0).sum())}")
