"""Does TensorRT make the vision encoder faster, and are its outputs the same?

Runs the fused encoder on the CUDA provider (what the viewer uses) and the plain-operator copy (onnx_trt_ready.py) on TensorRT, with the
same inputs (random patches laid out as a 30 x 42 grid, padded to 1264), and prints speed and how closely the outputs agree.

    LD_LIBRARY_PATH=runtime/cuda:runtime/cudnn:runtime/tensorrt .venv/bin/python src/bench_trt.py [batches] [batch_size]
    # default 25 batches of 4; the first TensorRT run builds the engine (minutes)

TensorRT's libraries are in runtime/tensorrt/, its engines are cached in .cache/trt_plain/.
"""

import ctypes
import sys
import time

import numpy as np
import onnxruntime as ort

import config

BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ONNX = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"
PADDED, PATCH_VALUES = 1264, 768

for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
    ctypes.CDLL(str(config.ROOT / "runtime" / "tensorrt" / lib), mode=ctypes.RTLD_GLOBAL)

rng = np.random.default_rng(1)
grid = np.array([(c, r) for r in range(30) for c in range(42)], dtype=np.int64)  # 1260 patches


def inputs():
    pixels = np.zeros((BATCH, PADDED, PATCH_VALUES), np.float32)
    pixels[:, :1260] = rng.random((BATCH, 1260, PATCH_VALUES), dtype=np.float32)
    positions = np.full((BATCH, PADDED, 2), -1, np.int64)
    positions[:, :1260] = grid
    return {"pixel_values": pixels, "pixel_position_ids": positions}


batches = [inputs() for _ in range(BATCHES)]
so = ort.SessionOptions()
so.log_severity_level = 3


def run(name, file, providers):
    t0 = time.time()
    session = ort.InferenceSession(str(ONNX / file), so, providers=providers)
    session.run(None, batches[0])  # the first run: engine build / warm-up
    print(f"{name}: ready in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    out = [session.run(None, b)[0] for b in batches]
    seconds = time.time() - t0
    print(f"RESULT {name}: {BATCHES * BATCH / seconds:.1f} photos/s (encoder only)", flush=True)
    return out


cuda = run("cuda fused", "vision_encoder_fp16_fused.onnx", [("CUDAExecutionProvider", {"device_id": 0})])
cache = config.ROOT / ".cache" / "trt_plain"
cache.mkdir(parents=True, exist_ok=True)
shapes = lambda n: ",".join(f"{k}:{n}x{PADDED}x{d}" for k, d in (("pixel_values", PATCH_VALUES), ("pixel_position_ids", 2)))
trt = run("tensorrt plain", "vision_encoder_fp16_trt.onnx", [
    ("TensorrtExecutionProvider", {
        "device_id": 0, "trt_fp16_enable": True, "trt_max_workspace_size": 1 << 30, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache),
        "trt_profile_min_shapes": shapes(1), "trt_profile_opt_shapes": shapes(BATCH), "trt_profile_max_shapes": shapes(BATCH)}),
    ("CUDAExecutionProvider", {"device_id": 0})])

cos = []
for a, b in zip(cuda, trt):
    a, b = a.astype(np.float32), b.astype(np.float32)
    cos.append((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9))
cos = np.concatenate([c.ravel() for c in cos])
print(f"RESULT agreement (cosine per output row): mean {cos.mean():.5f}, worst {cos.min():.5f}")
