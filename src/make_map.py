"""Make a photo map bundle from folders of images, for the photo-map viewer.
    python src/make_map.py <bundle> <folder> [<folder> ...]
Embedding is saved shard by shard in <bundle>/work/, so a stopped run continues where it left off; work/ is
deleted at the end.

The bundle is self-contained:
    manifest.json     format version, counts, model, page layout
    photos.json       per photo: file (originals/NNNNNN.ext), source (where it came from), width, height, title,
                      metadata from the image itself
    originals/        copies of the images, flat: photo i is originals/<i, 6 digits>.<ext>
    embeddings.bin    per photo: EmbeddingGemma vector, float16
    layout.bin        per photo: x, y (float32, 0 ... 10000); similar photos close together
    pages_bc7.bin     the always-loaded 1 ... 64 px levels, baked by photo-map's bake_pages (run at the end)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SHARD, BATCH, SOFT_TOKENS = 2_000, 32, 140  # same image budget as embed_images.py
WORLD = 10_000
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


# what cameras write into the description field when the photographer wrote nothing
BOILERPLATE = re.compile(r"^(SONY DSC|OLYMPUS DIGITAL CAMERA|KONICA MINOLTA DIGITAL CAMERA|DIGITAL CAMERA|DCIM|"
                         r"Exif_JPEG_PICTURE|default|Default|SAMSUNG|LEICA|Picture|image|Untitled|[\W_]*)$", re.IGNORECASE)


def clean(v, limit=500):
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-16-le" if len(v) % 2 == 0 and b"\x00" in v else "utf-8", errors="ignore")
    v = str(v).replace("\x00", "").strip()
    return v[:limit] if v and not BOILERPLATE.match(v) else None


def gps(ifd):  # EXIF GPS -> decimal degrees
    def deg(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d
    try:
        return f"{deg(ifd[2], ifd[1]):.6f}, {deg(ifd[4], ifd[3]):.6f}"
    except Exception:
        return None


def read_meta(path):
    """Size and whatever the file says about itself: EXIF, PNG text. Unreadable -> None."""
    try:
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else im.size
            exif, meta = im.getexif(), {}
            sub = exif.get_ifd(0x8769)
            for name, value in [
                ("title", exif.get(0x9C9B)),  # Windows XPTitle
                ("description", exif.get(0x010E)),
                ("date taken", sub.get(0x9003) or exif.get(0x0132)),
                ("camera", " ".join(filter(None, [clean(exif.get(0x010F)), clean(exif.get(0x0110))]))),
                ("artist", exif.get(0x013B)),
                ("copyright", exif.get(0x8298)),
                ("location", gps(exif.get_ifd(0x8825))),
            ] + [(k.lower(), v) for k, v in im.info.items() if k.lower() in ("title", "description", "author", "comment")]:
                if (v := clean(value)) and name not in meta:
                    meta[name] = v
            title = meta.pop("title", None) or meta.get("description") or Path(path).stem
            return {"width": w, "height": h, "title": title, "meta": meta}
    except Exception:
        return None


def scan(folders, bundle):
    """Every image under the folders, with its metadata, copied flat into originals/:
    photo i is originals/<i, 6 digits>.<ext>. Where it came from is kept only as a label ("source")."""
    sources = []
    for folder in folders:
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMAGE_TYPES and p.is_file() and not any(part.startswith(".") for part in p.parts):
                sources.append((p, f"{folder.name}/{p.relative_to(folder).as_posix()}"))
    step(f"{len(sources):,} images found")
    with Pool(10) as pool:
        metas = pool.map(read_meta, [src for src, _ in sources], chunksize=256)
    photos, origin = [], []  # origin: the source file of each photo, for copying
    for (src, label), m in zip(sources, metas):
        if m:  # unreadable files are left out
            photos.append({"file": f"originals/{len(photos):06d}{src.suffix.lower()}", "source": label, **m})
            origin.append(str(src.resolve()))
    step(f"{len(photos):,} readable, {len(sources) - len(photos)} skipped; {sum(bool(p['meta']) for p in photos):,} carry metadata")

    (bundle / "originals").mkdir(exist_ok=True)

    def copy(i):
        shutil.copy2(origin[i], bundle / photos[i]["file"])

    with ThreadPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(copy, range(len(photos))), 1):
            if k % 20_000 == 0:
                step(f"copied {k:,} / {len(photos):,}")
    step("originals copied")
    return photos


def embed(bundle, photos):
    """One vector per photo, computed in shards saved to work/ (a restarted run skips the finished ones)."""
    vectors = np.zeros((len(photos), 768), np.float32)
    sys.path.insert(0, str(ROOT / "src"))
    import config
    import torch
    model = config.load_model(images=True)
    model[0].processor.image_processor.max_soft_tokens = SOFT_TOKENS
    work = bundle / "work"
    work.mkdir(exist_ok=True)
    shards = [list(range(s, min(s + SHARD, len(photos)))) for s in range(0, len(photos), SHARD)]
    step(f"embedding {len(photos):,} images in {len(shards)} shards")

    def load(i):
        try:
            with Image.open(bundle / photos[i]["file"]) as im:
                return ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            return None

    t0, done = time.time(), 0
    with ThreadPoolExecutor(4) as pool:
        for s, part in enumerate(shards):
            path = work / f"emb_{s:04d}.npy"
            if not path.exists():
                images = list(pool.map(load, part))
                out = np.zeros((len(part), 768), np.float32)
                ok = [k for k, im in enumerate(images) if im is not None]
                batch = BATCH
                while True:  # out of GPU memory: retry with half the batch
                    try:
                        out[ok] = model.encode([{"image": images[k]} for k in ok], batch_size=batch, normalize_embeddings=True)
                        break
                    except torch.OutOfMemoryError:
                        batch = max(1, batch // 2)
                        torch.cuda.empty_cache()
                np.save(work / "tmp.npy", out)
                (work / "tmp.npy").rename(path)
                done += len(part)
                rate = done / (time.time() - t0)
                step(f"shard {s + 1}/{len(shards)}: {rate:.1f} img/s, ~{sum(len(p) for p in shards[s + 1 :]) / rate / 60:.0f} min left")
            vectors[part] = np.load(path)
    vectors.astype(np.float16).tofile(bundle / "embeddings.bin")
    step("embeddings.bin written")
    return vectors


def lay_out(bundle, vectors):
    """Centre, PCA to 50, UMAP to 2D."""
    import umap
    from sklearn.decomposition import TruncatedSVD

    n = len(vectors)
    v = vectors - vectors.mean(axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    pcs = TruncatedSVD(min(50, n - 1, v.shape[1] - 1), random_state=0).fit_transform(v).astype(np.float32)
    pcs /= np.maximum(np.linalg.norm(pcs, axis=1, keepdims=True), 1e-9)
    reducer = umap.UMAP(n_neighbors=min(11, n - 1), min_dist=0.05, metric="cosine", low_memory=True, random_state=0)
    xy = reducer.fit_transform(pcs)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    ((xy - lo) / np.maximum(hi - lo, 1e-9) * WORLD).astype(np.float32).tofile(bundle / "layout.bin")
    step("layout written")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    bundle, folders = Path(sys.argv[1]).resolve(), [Path(f).resolve() for f in sys.argv[2:]]
    bundle.mkdir(parents=True, exist_ok=True)

    photos = scan(folders, bundle)
    (bundle / "photos.json").write_text(json.dumps(photos, ensure_ascii=False))
    vectors = embed(bundle, photos)
    lay_out(bundle, vectors)
    manifest = {"format": 1, "name": bundle.stem, "created": time.strftime("%Y-%m-%d %H:%M"), "photos": len(photos),
                "world": WORLD, "embedding": {"model": "google/embeddinggemma-2", "dim": 768,
                "dtype": "float16", "image_tokens": SOFT_TOKENS}, "page": 4096, "levels": 11, "baked_levels": 7}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))

    step("baking the always-loaded levels (photo-map bake_pages)")
    env = {**os.environ, "CARGO_HOME": str(ROOT / ".cargo-home")}
    subprocess.run(["cargo", "run", "--offline", "-q", "--bin", "bake_pages", "--", str(bundle)], cwd=ROOT / "photo-map", env=env, check=True)
    shutil.rmtree(bundle / "work", ignore_errors=True)
    step(f"MAP_DONE: {bundle}")
