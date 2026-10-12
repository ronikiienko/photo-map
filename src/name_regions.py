"""Name a map's regions from a vocabulary of plain labels -> region_names.json (the viewer prefers it to regions.json).
    python src/name_regions.py [bundle] [--no-map-average]        default: maps/wikivoyage.photomap

For every level of the map's regions, each region (FINEST levels, MIN_PHOTOS photos or more) gets the phrase that fits it best:
- a region's vector is the average of its photos' image vectors, and separately of the vectors of its photos' captions where there are captions;
- the map's average vector is subtracted from these (image average from image, caption average from caption) and the vocabulary's average from the
  phrases ("centering"): what is left is how a region differs from a typical photo. Without it one all-purpose word ("outdoors") wins everywhere;
- a phrase's score is its cosine with the region's vector (the better of the image and the caption side) + COMMON * log10(common), where `common` is
  1000 / 100 / 10 for the round the phrase was written in (vocab/README.md): a basic word beats an unusual one when the fit is close.
The best phrase is the name. Only maps embedded by the vocabulary's model (Gemma) can be named; a BioCLIP map is skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np

import config
import map_search
import photomap
import vocab_embed

COMMON = 0.01
MIN_PHOTOS = 10  # smaller regions stay unnamed
FINEST = 6  # levels below this (regions of about 10 photos) stay unnamed
CHUNK = 1000


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def name(bundle, map_average=True):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["embedding"]["model"] != config.MODEL_ID:
        print(f"{bundle.name}: made with {manifest['embedding']['model']}, the vocabulary is for {config.MODEL_ID}: no names", flush=True)
        return
    n = manifest["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    data = map_search.load(bundle)
    items, V = vocab_embed.load()
    names = [i["name"] for i in items]
    V = unit(V - V.mean(0))
    bonus = (COMMON * np.log10([i["common"] for i in items]))[:, None]
    mean_img = data["img"].mean(0) if map_average else 0  # map_average=False: only for comparing, the names get worse
    mean_cap = data["cap"][data["has"]].mean(0) if map_average and data["has"].any() else 0
    result = []
    for li, lab in enumerate(photomap.cluster(xy)):
        out = []
        members = photomap.members(lab, lab.max() + 1)
        todo = [r for r, m in enumerate(members) if len(m) >= MIN_PHOTOS and li < FINEST]
        for c in range(0, len(todo), CHUNK):
            part = todo[c : c + CHUNK]
            img = unit(np.stack([data["img"][members[r]].mean(0) for r in part]) - mean_img)
            score = V @ img.T
            with_cap = [k for k, r in enumerate(part) if data["has"][members[r]].any()]
            if with_cap:
                cap = np.stack([data["cap"][[i for i in members[part[k]] if data["has"][i]]].mean(0) for k in with_cap])
                score[:, with_cap] = np.maximum(score[:, with_cap], V @ unit(cap - mean_cap).T)
            best = (score + bonus).argmax(0)
            for k, r in enumerate(part):
                pts = xy[members[r]]
                d = ((pts - pts.mean(0)) ** 2).sum(1)
                x, y = pts[d.argmin()]  # the label sits at the photo nearest the centre
                out.append({"name": names[best[k]], "x": float(x), "y": float(y), "spread": float(np.sqrt(d.mean())), "photos": len(pts)})
        out.sort(key=lambda l: -l["photos"])
        result.append(out)
        photomap.step(f"level {li + 1}: {len(out):,} of {lab.max() + 1:,} regions named, {len({l['name'] for l in out})} distinct names")
    (bundle / "region_names.json").write_text(json.dumps({"levels": result}, ensure_ascii=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name(args[0] if args else config.ROOT / "maps/wikivoyage.photomap", "--no-map-average" not in sys.argv)
