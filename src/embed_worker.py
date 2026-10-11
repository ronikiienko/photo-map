"""Embeds images and words for `photo-map make` (photo-map/src/make/embed.rs starts it and talks to it).
Prints "ready" once the model is loaded. Then reads one JSON line per job and writes the vectors to its "out" file
(float32, one row of 768 each) before printing "done":
    {"paths": [...], "out": ...}                  images (zeros if one can't be read)
    {"texts": [...], "prompt": ..., "out": ...}   texts, each with the prompt in front (e.g. a search query's)
Exits at the end of its input. Everything else it prints goes to stderr.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps

import config

BATCH, SOFT_TOKENS = 16, 140  # 16 measured fastest on the 4 GB GPU (bench_onnx.py)

answers, sys.stdout = sys.stdout, sys.stderr  # libraries print to stdout too: keep it for the answers only


def load(path):
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None


model = config.load_model(images=True)
model[0].processor.image_processor.max_soft_tokens = SOFT_TOKENS
print("ready", file=answers, flush=True)

batch = BATCH
with ThreadPoolExecutor(4) as pool:
    for line in sys.stdin:
        job = json.loads(line)
        if "texts" in job:
            out = model.encode(job["texts"], prompt=job.get("prompt", ""), batch_size=64, normalize_embeddings=True)
            out.astype(np.float32).tofile(job["out"] + ".tmp")
            os.replace(job["out"] + ".tmp", job["out"])
            print("done", file=answers, flush=True)
            continue
        images = list(pool.map(load, job["paths"]))
        out = np.zeros((len(images), 768), np.float32)
        ok = [k for k, im in enumerate(images) if im is not None]
        while ok:  # out of GPU memory: retry with half the batch (and keep it)
            try:
                out[ok] = model.encode([{"image": images[k]} for k in ok], batch_size=batch, normalize_embeddings=True)
                break
            except torch.OutOfMemoryError:
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"out of GPU memory: batch {batch}", file=sys.stderr, flush=True)
        out.tofile(job["out"] + ".tmp")
        os.replace(job["out"] + ".tmp", job["out"])
        print("done", file=answers, flush=True)
