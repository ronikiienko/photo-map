"""Embed the images downloaded by fetch_images.py, one vector per image (no text, no prompt).
    index/<corpus>/image_shards/NNNN.npy   2,000 images each; finished shards are skipped on rerun
    index/<corpus>/image_embeddings.npy    all shards joined
    index/<corpus>/image_chunks.json       which article each image belongs to
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import config
import numpy as np
import torch
from PIL import Image

SHARD = 2_000
BATCH = 32
SOFT_TOKENS = 140  # tokens per image: 70 / 140 / 280; 140 was as good as 280 in our test, at twice the speed

items = json.loads((config.ROOT / "data" / config.CORPUS / "images_all.json").read_text())  # from collect_zim_images.py
shard_dir = config.INDEX_DIR / "image_shards"
shard_dir.mkdir(parents=True, exist_ok=True)
model = config.load_model(images=True)
model[0].processor.image_processor.max_soft_tokens = SOFT_TOKENS


def load(item):
    """Decode on the CPU (in a thread) while the GPU works on the previous batch. Broken files -> None."""
    try:
        with Image.open(config.ROOT / item["path"]) as im:
            return im.convert("RGB")
    except Exception:
        return None


n_shards = (len(items) + SHARD - 1) // SHARD
start, done = time.time(), 0
print(f"embedding {len(items)} images in {n_shards} shards, {SOFT_TOKENS} tokens each", flush=True)
with ThreadPoolExecutor(4) as pool:
    for s in range(n_shards):
        path = shard_dir / f"{s:04d}.npy"
        if path.exists():
            continue
        part = items[s * SHARD : (s + 1) * SHARD]
        images = list(pool.map(load, part))
        vectors = np.zeros((len(part), config.DIM), dtype=np.float32)  # broken images stay all-zero (never match)
        ok = [i for i, im in enumerate(images) if im is not None]
        batch = BATCH
        while True:  # out of GPU memory: retry with half the batch
            try:
                vectors[ok] = model.encode([{"image": images[i]} for i in ok], batch_size=batch, normalize_embeddings=True)
                break
            except torch.OutOfMemoryError:
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"  out of GPU memory, retrying shard {s + 1} with batch {batch}", flush=True)
        np.save(shard_dir / "tmp.npy", vectors)
        (shard_dir / "tmp.npy").rename(path)
        done += len(part)
        rate = done / (time.time() - start)
        left = len(items) - (s + 1) * SHARD
        print(f"image shard {s + 1}/{n_shards}: {len(part) - len(ok)} broken, {rate:.1f} img/s, ~{max(left, 0) / rate / 60:.0f} min left", flush=True)

vectors = np.concatenate([np.load(shard_dir / f"{s:04d}.npy") for s in range(n_shards)])
np.save(config.INDEX_DIR / "image_embeddings.npy", vectors)
url = {"wikivoyage": "https://en.wikivoyage.org/wiki/"}.get(config.CORPUS, "")
chunks = [{"title": i["title"], "text": i["caption"], "url": url + i["title"].replace(" ", "_"), "image": i["path"],
           "articles": i.get("articles", [i["title"]])} for i in items]
(config.INDEX_DIR / "image_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
print(f"IMAGES_DONE: {vectors.shape} vectors", flush=True)
