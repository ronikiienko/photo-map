"""The Wikivoyage photo map: maps/wikivoyage.photomap, from what embed_images.py already made (no new embedding).
    python src/map_wikivoyage.py
Per photo: its caption as the title (else its article's title), plus the caption, article(s) and article link as
metadata, so all of them are searchable in the viewer.
"""

import json

import numpy as np

import config
import photomap

chunks = json.loads((config.ROOT / "index/wikivoyage/image_chunks.json").read_text())
vectors = np.load(config.ROOT / "index/wikivoyage/image_embeddings.npy")
photos, files = [], []
for c in chunks:
    meta = {"caption": c["text"], "article": ", ".join(c["articles"]), "url": c["url"]}
    photos.append({
        "title": c["text"] or c["title"],
        "source": c["image"].removeprefix("data/wikivoyage/"),
        "meta": {k: v for k, v in meta.items() if v},
    })
    files.append(config.ROOT / c["image"])
photomap.step(f"{len(photos):,} Wikivoyage photos, {sum(bool(c['text']) for c in chunks):,} with a caption")
photomap.make(config.ROOT / "maps/wikivoyage.photomap", photos, files, vectors)
