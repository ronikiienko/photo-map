"""How fast is SigLIP 2's photo encoder on ONNX Runtime (CUDA, then TensorRT), and do the outputs agree with PyTorch?

    LD_LIBRARY_PATH=runtime/cuda:runtime/cudnn:runtime/tensorrt .venv/bin/python src/bench_siglip.py [model] [batch ...] [cuda|tensorrt]
    # needs src/siglip_export.py first (the model takes float32 pixels); the first TensorRT run builds an engine (minutes), cached in .cache/trt_siglip/<model>
"""

import ctypes
import sys
import time

import numpy as np
import onnxruntime as ort

import config

args = sys.argv[1:]
name = args[0] if args and not args[0].isdigit() and args[0] not in ("cuda", "tensorrt") else "siglip2-base-patch16-512"
batches = [int(a) for a in args if a.isdigit()] or [1, 4, 8, 16, 32]
only = next((a for a in args if a in ("cuda", "tensorrt")), None)  # one provider alone: the other one's memory would starve TensorRT's engine build
model_dir = config.ROOT / ".cache" / "onnx" / name
size = int(name.rsplit("-", 1)[1])  # the checkpoint name ends with the image size, e.g. siglip2-large-patch16-512
for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
    ctypes.CDLL(str(config.ROOT / "runtime" / "tensorrt" / lib), mode=ctypes.RTLD_GLOBAL)

so = ort.SessionOptions()
so.log_severity_level = 3
cache = config.ROOT / ".cache" / "trt_siglip" / name
cache.mkdir(parents=True, exist_ok=True)
top = max(batches)
shape = lambda n: f"pixel_values:{n}x3x{size}x{size}"
providers = {
    "cuda": [("CUDAExecutionProvider", {"device_id": 0})],
    "tensorrt": [("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": True, "trt_max_workspace_size": 1 << 30, "trt_engine_cache_enable": True,
                                                "trt_engine_cache_path": str(cache), "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache),
                                                "trt_profile_min_shapes": shape(1), "trt_profile_opt_shapes": shape(top), "trt_profile_max_shapes": shape(top)}),
                 ("CUDAExecutionProvider", {"device_id": 0})],
}
rng = np.random.default_rng(0)
for label, ep in providers.items():
    if only and label != only:
        continue
    t0 = time.time()
    session = ort.InferenceSession(str(model_dir / "vision_fp16.onnx"), so, providers=ep)
    session.run(None, {"pixel_values": rng.standard_normal((1, 3, size, size)).astype(np.float32)})  # builds the engine the first time
    print(f"{label}: ready in {time.time() - t0:.0f} s", flush=True)
    for b in batches:
        x = {"pixel_values": rng.standard_normal((b, 3, size, size)).astype(np.float32)}
        session.run(None, x)
        runs = max(6, 192 // b)
        t0 = time.time()
        for _ in range(runs):
            session.run(None, x)
        print(f"RESULT {name} {label:9s} batch {b:2d}: {runs * b / (time.time() - t0):6.1f} photos/s", flush=True)
