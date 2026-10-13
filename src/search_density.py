"""Prototype: where on the map does a query concentrate? Top hits (image + caption vectors) are smoothed over the 2D
layout; the peak is where to fly. Three ways to score a peak: the hits' weight there (count), the share of the photos
around it that are hits (share), and in between (middle: weight / sqrt(photos around)).
    python src/search_density.py ["query" ...]
Needs the caption vectors from search_test.py.
"""

import json
import sys

import numpy as np

import config
import photomap
from search_test import BUNDLE, CACHE, QUERIES, z

HITS = 200  # top photos used
H = 120.0  # kernel width, in map units (the map is 10000 across)


def kernel(a, b):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * H * H))


if __name__ == "__main__":
    queries = sys.argv[1:] or QUERIES
    n = json.loads((BUNDLE / "manifest.json").read_text())["photos"]
    img = np.fromfile(BUNDLE / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    cap = np.load(CACHE).astype(np.float32)
    xy = np.fromfile(BUNDLE / "layout.bin", np.float32).reshape(n, 2)
    photos = json.loads((BUNDLE / "photos.json").read_text())
    texts = [p.get("caption", "") for p in photos]
    has = np.array([bool(t) for t in texts])
    levels = photomap.cluster(xy)
    named = json.loads((BUNDLE / "regions.json").read_text())["levels"]
    counting = [{int(lab[int(np.argmin(((xy - [r["x"], r["y"]]) ** 2).sum(1)))]): r["name"] for r in named[lv]} for lv, lab in enumerate(levels)]
    model = config.load_model()
    q = model.encode(queries, prompt=config.QUERY_PROMPT, normalize_embeddings=True)
    for query, v in zip(queries, q):
        s_img = z(img @ v)
        s_cap = np.where(has, z(np.where(has, cap @ v, (cap @ v)[has].mean())), s_img)
        s = (s_img + s_cap) / 2
        hit = np.argsort(-s)[:HITS]
        w = s[hit] - s[hit].min()  # the best photos weigh most
        print(f"\n### {query}")
        K = kernel(xy[hit], xy[hit])  # hit x hit
        weight = K @ w  # hits' weight around each hit
        near = np.array([np.exp(-((xy - xy[i]) ** 2).sum(1) / (2 * H * H)).sum() for i in hit])  # photos around each hit
        for name, score in (("count", weight), ("share", weight / near), ("middle", weight / np.sqrt(near))):
            best = hit[int(np.argmax(score))]
            around = hit[np.argsort(-w * (((xy[hit] - xy[best]) ** 2).sum(1) < (2 * H) ** 2))[:3]]
            region = next((f"{counting[lv][levels[lv][best]]} (level {lv + 1})" for lv in range(len(levels) - 1, -1, -1) if levels[lv][best] in counting[lv]), "-")
            inside = ((xy[hit] - xy[best]) ** 2).sum(1) < (2 * H) ** 2
            print(f"  {name:6} at ({xy[best][0]:5.0f},{xy[best][1]:5.0f}) hits within: {inside.sum():3}, photos around: {near[np.argmax(score)]:6.0f}  region: {region}")
            print("         " + " | ".join((texts[i] or photos[i]["title"])[:34] for i in around))
