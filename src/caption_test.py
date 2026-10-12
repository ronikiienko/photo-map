"""Experiment: do captions help? One region of the Wikivoyage map, made into two maps from the same photos:
    python src/caption_test.py [region]     (a coarsest-level region name of maps/wikivoyage.photomap; default "map")
    maps/wv-<region>-image.photomap       the stored image-only vectors
    maps/wv-<region>-captioned.photomap   each image embedded together with its text: "<article>: <caption> <|image|>"
                                          (just the article when there is no caption)
Same layout and naming code for both, so they can be compared side by side in the viewer.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import config
import photomap
from map_wikivoyage import gather

BATCH, SOFT_TOKENS, SHARD = 16, 140, 500  # same image budget as embed_images.py

region = sys.argv[1] if len(sys.argv) > 1 else "map"
bundle = config.ROOT / "maps/wikivoyage.photomap"
xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(-1, 2)
label = next(l for l in json.loads((bundle / "regions.json").read_text())["levels"][0] if l["name"] == region)
coarse = photomap.cluster(xy)[0]
ids = np.flatnonzero(coarse == coarse[np.argmin(((xy - [label["x"], label["y"]]) ** 2).sum(1))])
photos, files, vectors, chunks = gather()
photos, files, vectors, chunks = [photos[i] for i in ids], [files[i] for i in ids], vectors[ids], [chunks[i] for i in ids]
photomap.step(f'region "{region}": {len(ids):,} photos, {sum(bool(c["text"]) for c in chunks):,} with a caption')

model = config.load_model(images=True)
model[0].processor.image_processor.max_soft_tokens = SOFT_TOKENS
texts = [f"{c['title']}: {c['text']}" if c["text"] else c["title"] for c in chunks]


def load(path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


captioned = np.zeros_like(vectors)
start = time.time()
with ThreadPoolExecutor(4) as pool:
    for s in range(0, len(ids), SHARD):
        part = range(s, min(s + SHARD, len(ids)))
        images = list(pool.map(load, [files[i] for i in part]))
        ok = [k for k, im in enumerate(images) if im is not None]
        batch = BATCH
        while True:  # out of GPU memory: retry with half the batch
            try:
                out = model.encode([{"text": texts[s + k] + " <|image|>", "image": images[k]} for k in ok], batch_size=batch, normalize_embeddings=True)
                break
            except torch.OutOfMemoryError:
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
        captioned[[s + k for k in ok]] = out
        rate = part.stop / (time.time() - start)
        photomap.step(f"embedded {part.stop:,} / {len(ids):,}: {rate:.1f} img/s, ~{(len(ids) - part.stop) / rate / 60:.1f} min left")
del model
torch.cuda.empty_cache()

for kind, vecs in [("image", vectors), ("captioned", captioned)]:
    photomap.make(config.ROOT / f"maps/wv-{region.replace(' ', '-')}-{kind}.photomap", photos, files, vecs)
photomap.step("CAPTION_TEST_DONE")
