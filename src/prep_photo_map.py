"""Files for the desktop photo map viewer (photo-map/), from map_photos.py's layout and the original photos.
    python src/prep_photo_map.py            ->  data/wikivoyage/photo_map/
    python src/prep_photo_map.py --native   ->  only native.bin (fast: reads image headers)
meta.json    captions, image paths, articles, region labels
points.bin   per photo: x, y (f32, 0..10000), region (u32)
nn.bin       per photo: K look-alike indices (u32);  sim.bin: their similarity * 100 (u8)
native.bin   per photo: width, height (u16) of the original file, so the viewer knows how sharp it can stream it
fill.bin     per photo: width, height (u8) of the photo inside its tile: longer side 64, the shape kept
atlas.bin    every photo fitted into a 64 px tile (top-left corner, rest empty), raw RGBA, in LAYERS textures of 8192 x 8192 (128 x 128 tiles),
             each layer followed by its smaller mip levels (4096, 2048, ... 128), so the app uploads it as is
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index" / "wikivoyage"
OUT = ROOT / "data" / "wikivoyage" / "photo_map"
TILE, SIDE = 64, 8192
PER_ROW = SIDE // TILE
PER_LAYER = PER_ROW * PER_ROW
MIPS = 7  # 8192 ... 128: the last level holds each tile as a single pixel
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


BG = (21, 23, 28, 255)  # the viewer's background: transparent parts of flags and logos are filled with it


def tile(image):  # original photo -> 64 px tile with the photo in its top-left corner, and the photo's size
    out = np.zeros((TILE, TILE, 4), np.uint8)
    try:
        with Image.open(ROOT / image) as im:
            im.draft("RGB", (TILE * 2, TILE * 2))  # JPEGs decode at reduced size: much faster
            im = im.convert("RGBA")
            s = TILE / max(im.size)  # longer side becomes 64 (small images are enlarged)
            w, h = max(1, round(im.width * s)), max(1, round(im.height * s))
            im = Image.alpha_composite(Image.new("RGBA", im.size, BG), im).resize((w, h), Image.Resampling.LANCZOS)
        out[:h, :w] = np.asarray(im)
        return out, w, h
    except Exception:
        return out, TILE, TILE  # unreadable: an empty square


def native_size(image):
    try:
        with Image.open(ROOT / image) as im:
            return im.size
    except Exception:
        return 0, 0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = np.load(INDEX / "photo_map.npz")
    photos = json.loads((INDEX / "image_chunks.json").read_text())
    photos = [photos[i] for i in m["keep"]]
    with Pool(10) as pool:
        sizes = np.array(pool.map(native_size, [p["image"] for p in photos], chunksize=256), np.uint16)
    sizes.tofile(OUT / "native.bin")
    step(f"native sizes written: longest side median {int(np.median(sizes.max(axis=1)))} px, max {sizes.max()} px")
    if "--native" in __import__("sys").argv:
        raise SystemExit
    n, k = len(photos), m["nn"].shape[1]
    layers = -(-n // PER_LAYER)
    meta = {"n": n, "k": k, "tile": TILE, "side": SIDE, "mips": MIPS, "layers": layers,
            "labels": json.loads(str(m["labels"])), "cap": [p["text"] for p in photos], "img": [p["image"] for p in photos],
            "art": [p.get("articles") or [p["title"]] for p in photos]}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    points = np.zeros((n, 3), np.float32)
    points[:, :2] = m["xy"]
    points = points.view(np.uint32)
    points[:, 2] = m["region"]
    points.tofile(OUT / "points.bin")
    m["nn"].astype("<u4").tofile(OUT / "nn.bin")
    m["sim"].tofile(OUT / "sim.bin")
    step(f"{n:,} photos, {layers} atlas layers: meta, points, look-alikes written")

    fill = np.zeros((n, 2), np.uint8)
    with open(OUT / "atlas.bin", "wb") as f, Pool(10) as pool:
        for layer in range(layers):
            ids = [p["image"] for p in photos[layer * PER_LAYER : (layer + 1) * PER_LAYER]]
            atlas = np.zeros((SIDE, SIDE, 4), np.uint8)
            for j, (t, w, h) in enumerate(pool.imap(tile, ids, chunksize=64)):
                fill[layer * PER_LAYER + j] = w, h
                r, c = divmod(j, PER_ROW)
                atlas[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = t
            level = atlas
            for _ in range(MIPS):  # each mip: average of 2 x 2 pixels; tiles stay aligned since 64 is a power of 2
                f.write(level.tobytes())
                quads = level.reshape(level.shape[0] // 2, 2, level.shape[1] // 2, 2, 4).astype(np.uint16)
                level = ((quads.sum(axis=(1, 3)) + 2) // 4).astype(np.uint8)
            step(f"atlas layer {layer + 1}/{layers} done")
    fill.tofile(OUT / "fill.bin")
    step(f"PREP_DONE: {OUT} ({sum(p.stat().st_size for p in OUT.iterdir()) / 1e9:.2f} GB)")
