"""Image embedding speed: PyTorch bf16 (sentence-transformers, what we use) vs ONNX Runtime fp16 on the GPU.
Both get exactly the same preprocessed inputs, so this compares the runtimes only. Also checks the vectors agree.
    python src/bench_onnx.py [n_images] [onnx_batch] [cuda|fused|trt]   (default 200 4 cuda; ONNX files in .cache/onnx/embeddinggemma-2/)
fused: the CUDA provider with the vision encoder rewritten for fused attention (onnx_fuse_attention.py).
trt: TensorRT fp16, engines cached in .cache/trt/ (the first run compiles them, which takes a while).
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BATCH = 16
ONNX = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"
images_dir = config.ROOT / "maps" / "wikivoyage.photomap" / "originals"


def load(p):
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


files = sorted(images_dir.iterdir())[:N]
images = [load(p) for p in files]
batches = [images[i : i + BATCH] for i in range(0, N, BATCH)]
print(f"{N} images loaded, batches of {BATCH}", flush=True)

# --- PyTorch (current) ---
import torch

model = config.load_model(images=True)
model[0].processor.image_processor.max_soft_tokens = 140
model.encode([{"image": images[0]}], batch_size=1)  # warm-up

t0 = time.time()
features = [model.preprocess([{"image": im} for im in b]) for b in batches]
prep = time.time() - t0
print(f"preprocessing: {N / prep:.1f} img/s (same for both runtimes)", flush=True)
print("inputs:", {k: tuple(v.shape) for k, v in features[0].items() if hasattr(v, "shape")}, flush=True)

torch.cuda.synchronize()
t0 = time.time()
pt = []
for k, b in enumerate(batches):
    pt.append(model.encode([{"image": im} for im in b], batch_size=BATCH, normalize_embeddings=True))
    print(f"  pytorch batch {k + 1}/{len(batches)}", flush=True)
torch.cuda.synchronize()
pt_total = time.time() - t0
pt = np.concatenate(pt)
print(f"RESULT pytorch bf16: {N / pt_total:.1f} img/s with preprocessing, ~{N / max(pt_total - prep, 1e-9):.1f} img/s model only", flush=True)
del model
torch.cuda.empty_cache()

# --- ONNX Runtime ---
import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 3
EP = sys.argv[3] if len(sys.argv) > 3 else "cuda"
providers = [("CUDAExecutionProvider", {"device_id": 0})]
if EP == "trt":
    import ctypes
    import tensorrt_libs  # NVIDIA's TensorRT 10 libraries: load them so ONNX Runtime finds libnvinfer.so.10

    for lib in sorted(Path(tensorrt_libs.__path__[0]).glob("lib*.so.10")):
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    cache = config.ROOT / ".cache" / "trt"
    cache.mkdir(parents=True, exist_ok=True)
    providers.insert(0, ("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                        "trt_engine_cache_path": str(cache), "trt_timing_cache_enable": True,
                                                        "trt_timing_cache_path": str(cache)}))
t0 = time.time()
# trt: copies with every size fixed for batch 4 and all shapes written in (TensorRT needs them; this export lacks some)
suffix = "_b4" if EP == "trt" else ""
vision = ort.InferenceSession(str(ONNX / f"vision_encoder_fp16{'_fused' if EP == 'fused' else suffix}.onnx"), so, providers=providers)
text = ort.InferenceSession(str(ONNX / f"model_fp16{suffix}.onnx"), so, providers=providers)
print(f"onnx sessions ready in {time.time() - t0:.0f}s, providers: {vision.get_providers()}", flush=True)
empty = np.zeros((0, 512), np.float32)


ONNX_BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # its attention needs the whole grid at once: batch 16 runs out of VRAM


def run_onnx(f):
    outs = []
    for s in range(0, len(f["input_ids"]), ONNX_BATCH):
        part = {k: v[s : s + ONNX_BATCH] for k, v in f.items() if hasattr(v, "shape")}
        pixels, positions = part["pixel_values"].float().numpy(), part["image_position_ids"].numpy()
        if EP == "fused":  # the fused attention kernel needs a sequence length that is a multiple of 8: pad with masked patches
            extra = -pixels.shape[1] % 8
            pixels = np.pad(pixels, ((0, 0), (0, extra), (0, 0)))
            positions = np.pad(positions, ((0, 0), (0, extra), (0, 0)), constant_values=-1)
        img = vision.run(None, {"pixel_values": pixels, "pixel_position_ids": positions})[0]
        outs.append(text.run(["sentence_embedding"], {"input_ids": part["input_ids"].numpy(), "attention_mask": part["attention_mask"].numpy(),
                                                       "image_features": img.astype(np.float32), "video_features": empty, "audio_features": empty})[0])
    out = np.concatenate(outs)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


t0 = time.time()
run_onnx(features[0])  # warm-up (TensorRT: builds or loads its engines here)
print(f"onnx warm-up: {time.time() - t0:.0f}s", flush=True)
t0 = time.time()
ox = []
for k, f in enumerate(features):
    ox.append(run_onnx(f))
    print(f"  onnx batch {k + 1}/{len(batches)}", flush=True)
ox_total = time.time() - t0
ox = np.concatenate(ox)
print(f"RESULT onnx {EP} fp16 (batch {ONNX_BATCH}): ~{N / ox_total:.1f} img/s model only, ~{N / (ox_total + prep):.1f} img/s with preprocessing", flush=True)

nan = int(np.isnan(ox).any(axis=1).sum())
cos = np.sum(pt * ox, axis=1)
print(f"RESULT vectors: {nan} NaN; cosine onnx vs pytorch min {np.nanmin(cos):.4f}, mean {np.nanmean(cos):.4f}", flush=True)
print("BENCH_DONE", flush=True)
