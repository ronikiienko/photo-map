"""Name one level of a map's regions (default: level 6, regions of about 28 photos) with a small language model -> region_names.json (the viewer prefers it to regions.json).
    python src/name_finest.py [bundle] [model] [level]   default: maps/wikivoyage.photomap, Qwen3.5-2B-Q4_K_M (a GGUF in .cache/llama.cpp/models/), level 6 (1 = coarsest)
A region is named from its photos' captions (name_compare.prompt_for: the commonest ones, short), with the descriptive LABEL_PROMPT, and the answer is used as it comes.
Regions with fewer than MIN_PHOTOS photos or MIN_CAPTIONS captions stay unnamed, and so does a region the model answers 'none' for. Only the finest
level is named for now: the other levels of the file are empty.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import config
import photomap
from name_compare import LABEL_PROMPT, Server, prompt_for

MIN_PHOTOS, MIN_CAPTIONS = 10, 5

if __name__ == "__main__":
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else config.ROOT / "maps/wikivoyage.photomap"
    gguf = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.5-2B-Q4_K_M"
    level = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    texts = [p["meta"].get("caption", "") for p in json.loads((bundle / "photos.json").read_text())]
    levels = photomap.cluster(xy)
    lab = levels[level - 1]
    members = photomap.members(lab, lab.max() + 1)
    todo = [r for r, m in enumerate(members) if len(m) >= MIN_PHOTOS and sum(bool(texts[i]) for i in m) >= MIN_CAPTIONS]
    photomap.step(f"{len(levels)} levels; level {level} has {lab.max() + 1:,} regions, {len(todo):,} to name with {gguf}")
    out, start = [], time.time()
    with Server(gguf) as server:
        server.ask(prompt_for(texts, members[todo[0]], LABEL_PROMPT), max_tokens=14, drop_bad=False)  # warm-up
        for k, r in enumerate(todo):
            answer = server.ask(prompt_for(texts, members[r], LABEL_PROMPT), max_tokens=14, drop_bad=False)  # whatever it answers is the name
            if answer not in ("none", ""):
                pts = xy[members[r]]
                c = pts.mean(0)
                d = ((pts - c) ** 2).sum(1)
                x, y = pts[d.argmin()]  # the label sits at the photo nearest the centre, as in regions.json
                out.append({"name": answer, "x": float(x), "y": float(y), "spread": float(np.sqrt(d.mean())), "photos": len(pts)})
            if (k + 1) % 250 == 0:
                photomap.step(f"{k + 1:,} / {len(todo):,} regions ({(k + 1) / (time.time() - start):.1f}/s), {len(out):,} named")
    out.sort(key=lambda l: -l["photos"])
    (bundle / "region_names.json").write_text(json.dumps({"levels": [out if l == level - 1 else [] for l in range(len(levels))]}, ensure_ascii=False))
    photomap.step(f"region_names.json: {len(out):,} names")
