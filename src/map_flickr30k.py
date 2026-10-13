"""The Flickr30k photo map: maps/flickr30k.photomap (31,014 photos with five human-written captions each, from
data/flickr30k: huggingface.co/datasets/nlphuji/flickr30k, the images zip extracted to data/flickr30k/images/).
    python src/map_flickr30k.py
The first caption is each photo's caption (shown above it, searched); all five go in the metadata, and the caption vector
(captions.bin) is the average of the five captions' vectors. The image vectors are kept in index/flickr30k/.
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import config
import name_regions
import photomap

DATA = config.ROOT / "data/flickr30k"
INDEX = config.ROOT / "index/flickr30k"
BUNDLE = config.ROOT / "maps/flickr30k.photomap"
BATCH = 32
SOFT_TOKENS = 140  # as in embed_images.py
SEP = " | "


def records():
    """Per photo: its file and its five captions, in the annotation file's order."""
    images = next(p for p in (DATA / "images").rglob("*.jpg")).parent
    with open(DATA / "flickr_annotations_30k.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"file": images / r["filename"], "captions": [c.strip() for c in json.loads(r["raw"])]} for r in rows]


def load(path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def embed(files):
    cache = INDEX / "image_embeddings.npy"
    if cache.exists():
        return np.load(cache)
    model = config.load_model(images=True)
    model[0].processor.image_processor.max_soft_tokens = SOFT_TOKENS
    vectors = np.zeros((len(files), config.DIM), np.float32)  # unreadable images stay all-zero
    start, step = time.time(), 256
    with ThreadPoolExecutor(4) as pool:
        for s in range(0, len(files), step):
            images = list(pool.map(load, files[s : s + step]))
            ok = [i for i, im in enumerate(images) if im is not None]
            batch = BATCH
            while True:
                try:
                    vectors[[s + i for i in ok]] = model.encode([{"image": images[i]} for i in ok], batch_size=batch, normalize_embeddings=True)
                    break
                except torch.OutOfMemoryError:
                    batch = max(1, batch // 2)
                    torch.cuda.empty_cache()
                    print(f"  out of GPU memory, retrying with batch {batch}", flush=True)
            done = min(s + step, len(files))
            print(f"  {done:,} / {len(files):,} images ({done / (time.time() - start):.0f}/s)", flush=True)
    INDEX.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    return vectors


def add_captions(bundle):
    """captions.bin: per photo, the average of its captions' vectors (embedded like the other maps' captions)."""
    photos = json.loads((bundle / "photos.json").read_text())
    texts = [c for p in photos for c in p["meta"]["captions"].split(SEP)]
    per = [len(p["meta"]["captions"].split(SEP)) for p in photos]
    model = config.load_model()
    vectors = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    ends = np.cumsum(per)
    mean = np.stack([vectors[e - n : e].mean(0) for e, n in zip(ends, per)])
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    mean.astype(np.float16).tofile(bundle / "captions.bin")
    photomap.step(f"{len(photos):,} caption vectors (from {len(texts):,} captions)")


if __name__ == "__main__":
    recs = records()
    photomap.step(f"{len(recs):,} photos")
    vectors = embed([r["file"] for r in recs])
    photos = [
        {"title": r["captions"][0], "source": f"flickr30k/{r['file'].name}", "meta": {"caption": r["captions"][0], "captions": SEP.join(r["captions"])}}
        for r in recs
    ]
    photomap.make(BUNDLE, photos, [r["file"] for r in recs], vectors)
    add_captions(BUNDLE)
    name_regions.name(BUNDLE)
