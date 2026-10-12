"""The Rare Species photo map: maps/rare-species.photomap (11,983 photos of 400 IUCN-listed species, from
data/rare-species, i.e. huggingface.co/datasets/imageomics/rare-species).
    python src/map_rare_species.py
Images only: no captions, so the species names (kept as each photo's title and metadata) are never embedded and the
map's regions are named from the vocabulary. The image vectors are kept in index/rare-species/ so other models'
vectors or a rebuilt map can be compared. The photos are cut out of the parquet files into data/rare-species/images/.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

import config
import photomap

DATA = config.ROOT / "data/rare-species"
INDEX = config.ROOT / "index/rare-species"
BATCH = 32
SOFT_TOKENS = 140  # as in embed_images.py
RANK = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def extract():
    """Write every photo of the parquet files to data/rare-species/images/ (once); the photos' records, in file order."""
    (DATA / "images").mkdir(exist_ok=True)
    records = []
    for part in sorted((DATA / "data").glob("*.parquet")):
        f = pq.ParquetFile(part)
        for g in range(f.num_row_groups):
            for row in f.read_row_group(g).to_pylist():
                path = DATA / "images" / f"{len(records):05d}.jpg"
                if not path.exists():
                    path.write_bytes(row["file_name"]["bytes"])
                records.append({"file": path, **{k: row[k] for k in (*RANK, "sciName", "common")}})
    return records


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


if __name__ == "__main__":
    records = extract()
    photomap.step(f"{len(records):,} photos extracted")
    vectors = embed([r["file"] for r in records])
    photos = []
    for r in records:
        meta = {k: r[k] for k in (*RANK, "sciName", "common") if r[k]}
        photos.append({"title": r["common"] or r["sciName"], "source": f"images/{r['file'].name}", "meta": meta})
    photomap.make(config.ROOT / "maps/rare-species.photomap", photos, [r["file"] for r in records], vectors)
