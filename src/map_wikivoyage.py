"""The Wikivoyage photo map: maps/wikivoyage.photomap, from what embed_images.py already made (no new embedding).
    python src/map_wikivoyage.py
Per photo: its caption as the title (else its article's title), plus the caption, article(s) and article link as
metadata, so all of them are searchable in the viewer. Small uncaptioned images (inline icons) are left out.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
import photomap

ICON = 100  # px, longer side: uncaptioned images up to this size are inline icons (route shields, line badges)


def gather():
    """Every Wikivoyage photo: photos (title, source, meta), files, image vectors and the index's chunks."""
    chunks = json.loads((config.ROOT / "index/wikivoyage/image_chunks.json").read_text())
    vectors = np.load(config.ROOT / "index/wikivoyage/image_embeddings.npy")
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(photomap.size, [config.ROOT / c["image"] for c in chunks]))
    keep = [i for i, (c, s) in enumerate(zip(chunks, sizes)) if c["text"] or (s and max(s) > ICON)]
    photomap.step(f"{len(chunks) - len(keep):,} small uncaptioned images (icons, route shields) left out")
    chunks, vectors = [chunks[i] for i in keep], vectors[keep]
    photos, files = [], []
    for c in chunks:
        meta = {"caption": c["text"], "article": ", ".join(c["articles"]), "url": c["url"]}
        photos.append({
            "title": c["text"] or c["title"],
            "source": c["image"].removeprefix("data/wikivoyage/"),
            "meta": {k: v for k, v in meta.items() if v},
        })
        files.append(config.ROOT / c["image"])
    return photos, files, vectors, chunks


if __name__ == "__main__":
    photos, files, vectors, chunks = gather()
    photomap.step(f"{len(photos):,} Wikivoyage photos, {sum(bool(c['text']) for c in chunks):,} with a caption")
    photomap.make(config.ROOT / "maps/wikivoyage.photomap", photos, files, vectors)
