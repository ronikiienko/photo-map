"""The Rare Species photo map made with BioCLIP 2 (imageomics/bioclip-2, ViT-L/14, 768-dim) instead of Gemma:
maps/rare-species-bioclip2.photomap. Same photos as map_rare_species.py (run that first to extract them).
    python src/map_rare_species_bioclip2.py
Image vectors (fp16 model, 224 px centre crop, 257 tokens) are kept in index/rare-species/bioclip2_image_embeddings.npy.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch

import config
import photomap
from map_rare_species import INDEX, RANK, extract, load

BATCH = 64
MODEL = "hf-hub:imageomics/bioclip-2"


def embed(files):
    cache = INDEX / "bioclip2_image_embeddings.npy"
    if cache.exists():
        return np.load(cache)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.half().cuda().eval()
    vectors = np.zeros((len(files), 768), np.float32)  # unreadable images stay all-zero
    start = time.time()
    with ThreadPoolExecutor(4) as pool, torch.no_grad():
        for s in range(0, len(files), BATCH):
            images = list(pool.map(load, files[s : s + BATCH]))
            ok = [i for i, im in enumerate(images) if im is not None]
            x = torch.stack([preprocess(images[i]) for i in ok]).half().cuda()
            vectors[[s + i for i in ok]] = model.encode_image(x, normalize=True).float().cpu().numpy()
            if (s // BATCH) % 10 == 0:
                done = min(s + BATCH, len(files))
                print(f"  {done:,} / {len(files):,} images ({done / (time.time() - start):.0f}/s)", flush=True)
    INDEX.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    return vectors


if __name__ == "__main__":
    records = extract()
    vectors = embed([r["file"] for r in records])
    photos = []
    for r in records:
        meta = {k: r[k] for k in (*RANK, "sciName", "common") if r[k]}
        photos.append({"title": r["common"] or r["sciName"], "source": f"images/{r['file'].name}", "meta": meta})
    photomap.make(config.ROOT / "maps/rare-species-bioclip2.photomap", photos, [r["file"] for r in records], vectors, "imageomics/bioclip-2")
