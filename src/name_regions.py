"""Name a map's regions from a vocabulary of plain labels -> region_names.json (the viewer prefers it to regions.json).
    python src/name_regions.py [bundle] [--up N] [--no-map-average]        default: maps/wikivoyage.photomap

For every level of the map's regions, each region (FINEST levels, MIN_PHOTOS photos or more) gets the phrase that fits it best:
- a region's vector is the average of its photos' image vectors, and separately of the vectors of its photos' captions where there are captions;
- the average vector of the region N levels up that contains it (--up N, default 0: the map's average, 1: its parent, 2: its grandparent; the coarsest level always uses the
  map's) is subtracted from these (image average from image, caption average from caption) and the vocabulary's average from the phrases ("centering"):
  what is left is how a region differs from the bigger one around it, so a style every region of a big area shares ("illustration") is named once, by the
  big region. With 0 (the default) it is how a region differs from a typical photo (without any centering one all-purpose word, "outdoors", wins everywhere). Tried on
  Flickr30k and Wikivoyage: 1 and 2 give 2-3 times as many distinct names, but many are nonsense (names for what the vocabulary can't express: "goat" for blue
  hair), which is worse than a repeated plain one;
- a phrase's score is its cosine with the region's vector (the better of the image and the caption side) + COMMON * log10(common), where `common` is
  1000 / 100 / 10 for the round the phrase was written in (vocab/README.md): a basic word beats an unusual one when the fit is close.
The best phrase is the name. Only maps embedded by the vocabulary's model (Gemma) can be named; a BioCLIP map is skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np

import config
import photomap
import vocab_embed

COMMON = 0.01
MIN_PHOTOS = 10  # smaller regions stay unnamed
FINEST = 6  # levels below this (regions of about 10 photos) stay unnamed
CHUNK = 1000


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def load_vectors(bundle):
    """A map's vectors: its photos' image vectors, the caption vectors (captions.bin, if the map has captions) and who has a caption."""
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    img = np.fromfile(bundle / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    photos = json.loads((bundle / "photos.json").read_text())
    has = np.array([bool(p.get("caption", "")) for p in photos])
    cap_file = bundle / "captions.bin"
    cap = np.fromfile(cap_file, np.float16).reshape(n, -1).astype(np.float32) if has.any() and cap_file.exists() else np.zeros_like(img)
    if not cap.any():
        has = np.zeros(n, bool)
    return {"img": img, "cap": cap, "has": has}


def name(bundle, map_average=True, up=0):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["embedding"]["model"] != config.MODEL_ID:
        print(f"{bundle.name}: made with {manifest['embedding']['model']}, the vocabulary is for {config.MODEL_ID}: no names", flush=True)
        return
    n = manifest["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    data = load_vectors(bundle)
    items, V = vocab_embed.load()
    names = [i["name"] for i in items]
    V = unit(V - V.mean(0))
    bonus = (COMMON * np.log10([i["common"] for i in items]))[:, None]
    mean_img = data["img"].mean(0) if map_average else np.zeros(config.DIM, np.float32)  # map_average=False: only for comparing, the names get worse
    mean_cap = data["cap"][data["has"]].mean(0) if map_average and data["has"].any() else np.zeros(config.DIM, np.float32)
    result = []
    labs = photomap.cluster(xy)
    all_members = [photomap.members(lab, lab.max() + 1) for lab in labs]
    cached = {}

    def parent_mean(li, p, side):
        """The average of the vectors (side: "img" or "cap", the photos with a caption only) of region p of level li; None if it has none."""
        if (li, p, side) not in cached:
            idx = all_members[li][p]
            idx = idx if side == "img" else [i for i in idx if data["has"][i]]
            cached[(li, p, side)] = data[side][idx].mean(0) if len(idx) else None
        return cached[(li, p, side)]

    def centre(vectors, regions, li, side, map_mean):
        """vectors (one per region of `regions`) minus the average of the region `up` levels up (the coarsest there is, if fewer); the map's where there is none or it is the same set of photos."""
        above = max(li - up, 0)
        base = np.stack([map_mean if up == 0 or li == 0 or (m := parent_mean(above, labs[above][regions[k][0]], side)) is None else m for k in range(len(regions))])
        diff = vectors - base
        same = np.linalg.norm(diff, axis=1) < 1e-4  # a region with exactly its parent's photos: nothing to subtract
        diff[same] = (vectors - map_mean)[same]
        return unit(diff)

    for li, lab in enumerate(labs):
        out = []
        members = all_members[li]
        todo = [r for r, m in enumerate(members) if len(m) >= MIN_PHOTOS and li < FINEST]
        for c in range(0, len(todo), CHUNK):
            part = todo[c : c + CHUNK]
            img = centre(np.stack([data["img"][members[r]].mean(0) for r in part]), [members[r] for r in part], li, "img", mean_img)
            score = V @ img.T
            with_cap = [k for k, r in enumerate(part) if data["has"][members[r]].any()]
            if with_cap:
                cap = np.stack([data["cap"][[i for i in members[part[k]] if data["has"][i]]].mean(0) for k in with_cap])
                score[:, with_cap] = np.maximum(score[:, with_cap], V @ centre(cap, [members[part[k]] for k in with_cap], li, "cap", mean_cap).T)
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
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    name(args[0] if args else config.ROOT / "maps/wikivoyage.photomap", "--no-map-average" not in sys.argv, int(sys.argv[sys.argv.index("--up") + 1]) if "--up" in sys.argv else 0)
