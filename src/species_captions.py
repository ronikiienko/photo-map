"""Captions for the Rare Species map, made from what each photo comes with: its common name, scientific name and taxonomy.
    python src/species_captions.py [bundle]        default: maps/rare-species.photomap (a Gemma map; BioCLIP's vectors are in another space)
"caption" (shown above a photo when zoomed in, and used by name_regions.py) = "Purple wartyback (Cyclonaias tuberculata)";
captions.bin = the vectors of a fuller text (name, scientific name, class, order, family), embedded like the Wikivoyage captions: float16, one row per photo.
"""

import json
import sys
from pathlib import Path

import numpy as np

import config

RANKS = ("class", "order", "family")


def caption(meta):
    return f"{meta['common']} ({meta['sciName']})" if meta.get("sciName") and meta.get("common") else meta.get("common") or meta.get("sciName", "")


def text(meta):
    ranks = ", ".join(meta[r] for r in RANKS if meta.get(r))
    return f"{caption(meta)}, {ranks}" if ranks else caption(meta)


def add(bundle):
    bundle = Path(bundle)
    photos = json.loads((bundle / "photos.json").read_text())
    model = config.load_model()
    vectors = model.encode([text(p["meta"]) for p in photos], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    vectors.astype(np.float16).tofile(bundle / "captions.bin")
    for p in photos:
        p["caption"] = caption(p["meta"])
    (bundle / "photos.json").write_text(json.dumps(photos, ensure_ascii=False))
    print(f"{bundle.name}: {len(photos):,} captions, e.g. {photos[0]['caption']!r}; embedded text: {text(photos[0]['meta'])!r}", flush=True)


if __name__ == "__main__":
    add(sys.argv[1] if len(sys.argv) > 1 else config.ROOT / "maps/rare-species.photomap")
