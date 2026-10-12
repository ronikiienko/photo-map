"""The label vocabulary (vocab/curated2_round*.txt, see vocab/README.md) with its vectors: the phrases embedded as search queries by the same model as the maps' photos.
    python src/vocab_embed.py        (re)build data/vocab/curated2.json + curated2.npy; name_regions.py does this itself when they are missing or older than the text
"""

import json
import time

import numpy as np

import config

SRC = config.ROOT / "vocab"
OUT = config.ROOT / "data" / "vocab"
ROUNDS = (1, 2, 3)
STEM = "curated2"


def phrases():
    """Every phrase once (its earliest round), with: general (the file's '# general' block), round, common = 1000 / 100 / 10 for the round."""
    items, seen = [], set()
    for rnd in ROUNDS:
        general = True
        for line in (SRC / f"{STEM}_round{rnd}.txt").read_text().splitlines():
            line = line.strip()
            if line.startswith("#"):
                general = line.lstrip("# ").strip() == "general"
            elif line and line.lower() not in seen:
                seen.add(line.lower())
                items.append({"name": line, "general": general, "round": rnd, "common": 10 ** (4 - rnd)})
    return items


def build():
    items = phrases()
    print(f"{len(items)} phrases, embedding", flush=True)
    model = config.load_model()
    start = time.time()
    vectors = model.encode([i["name"] for i in items], prompt=config.QUERY_PROMPT, normalize_embeddings=True, batch_size=128)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{STEM}.npy", vectors.astype(np.float16))
    (OUT / f"{STEM}.json").write_text(json.dumps(items))
    print(f"saved ({time.time() - start:.0f} s)", flush=True)


def load():
    """(items, vectors n x dim float32); built first if missing or older than the text files."""
    npy = OUT / f"{STEM}.npy"
    if not npy.exists() or npy.stat().st_mtime < max(f.stat().st_mtime for f in SRC.glob(f"{STEM}_round*.txt")):
        build()
    return json.loads((OUT / f"{STEM}.json").read_text()), np.load(npy).astype(np.float32)


if __name__ == "__main__":
    build()
