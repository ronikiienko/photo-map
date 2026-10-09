"""Map of Simple Wikipedia: one dot per article, similar articles close together.
    python src/map_simplewiki.py   ->  reports/simplewiki_map.html (open in a browser)
Article vector = mean of its chunk vectors, re-normalized. Then: subtract the average article (so the shared
"I'm an encyclopedia text" part doesn't dominate), PCA to 50 numbers, UMAP to 2D, k-means for labelled regions.
CPU only; runs fine while the GPU is busy.
"""

import base64
import json
import re
import time

import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
CORPUS = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "simplewiki"  # or wikivoyage
INDEX = ROOT / "index" / CORPUS
REGIONS = 40
NEIGHBOURS = 10
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


# 1. Which article each chunk belongs to. Only the titles are needed, so pull them out of chunks.json
#    with a regex instead of loading every chunk's text into memory.
raw = (INDEX / "chunks.json").read_text()
titles = [json.loads(f'"{t}"') for t in re.findall(r'\{"title": "((?:[^"\\]|\\.)*)"', raw)]
del raw
vectors = np.load(INDEX / "embeddings.npy", mmap_mode="r")
assert len(titles) == len(vectors), (len(titles), len(vectors))
names, article_of = np.unique(np.array(titles, dtype=object), return_inverse=True)
del titles
step(f"{len(vectors):,} chunks belong to {len(names):,} articles")

# 2. Article vector = mean of its chunks, then length 1 again
sums = np.zeros((len(names), vectors.shape[1]), dtype=np.float32)
for i in range(0, len(vectors), 20_000):
    np.add.at(sums, article_of[i : i + 20_000], np.asarray(vectors[i : i + 20_000], dtype=np.float32))
n_chunks = np.bincount(article_of)
art = sums / np.linalg.norm(sums, axis=1, keepdims=True)
del sums
step("article vectors done")

# 3. Remove the shared average, PCA to 50 numbers
art -= art.mean(axis=0)
art /= np.linalg.norm(art, axis=1, keepdims=True)
svd = TruncatedSVD(50, random_state=0)
pcs = svd.fit_transform(art).astype(np.float32)
del art
pcs /= np.linalg.norm(pcs, axis=1, keepdims=True)
step(f"PCA to 50 numbers keeps {svd.explained_variance_ratio_.sum():.0%} of the variation")

# 4. UMAP to 2D (also gives each article's nearest neighbours)
reducer = umap.UMAP(n_neighbors=NEIGHBOURS + 1, min_dist=0.05, metric="cosine", low_memory=True, random_state=0, verbose=True)
xy = reducer.fit_transform(pcs)
nn = reducer._knn_indices[:, 1 : NEIGHBOURS + 1]  # [:, 0] is the article itself
nn_sim = np.einsum("ij,ikj->ik", pcs, pcs[nn])
step("UMAP done")

# 5. Regions: k-means, each named after its most typical articles (skipping one-chunk stubs when possible)
km = MiniBatchKMeans(REGIONS, random_state=0, n_init=3, batch_size=4096).fit(pcs)
labels = []
for r in range(REGIONS):
    members = np.where(km.labels_ == r)[0]
    centre = km.cluster_centers_[r] / np.linalg.norm(km.cluster_centers_[r])
    pool = members[n_chunks[members] > 1] if (n_chunks[members] > 1).sum() >= 3 else members
    typical = pool[np.argsort(-(pcs[pool] @ centre))[:3]]
    labels.append({"text": " · ".join(names[typical]), "x": float(np.median(xy[members, 0])),
                   "y": float(np.median(xy[members, 1])), "size": int(len(members))})
step("regions labelled")

# 6. Write a self-contained HTML page (data inline, no internet needed)
lo, hi = xy.min(axis=0), xy.max(axis=0)
q = np.round((xy - lo) / (hi - lo) * 10000).astype(int)  # positions as integers 0..10000 (smaller file)
for l in labels:
    l["x"] = round((l["x"] - lo[0]) / (hi[0] - lo[0]) * 10000)
    l["y"] = round((l["y"] - lo[1]) / (hi[1] - lo[1]) * 10000)
packed = lambda a: base64.b64encode(a.tobytes()).decode()  # numbers as binary: smaller, much faster to load
data = {"t": names.tolist(), "x": packed(q[:, 0].astype(np.uint16)), "y": packed(q[:, 1].astype(np.uint16)),
        "c": packed(np.minimum(n_chunks, 65535).astype(np.uint16)), "nn": packed(nn.ravel().astype(np.uint32)),
        "s": packed(np.clip(np.round(nn_sim.ravel() * 100), 0, 255).astype(np.uint8)), "k": NEIGHBOURS, "labels": labels, "name": {"simplewiki": "Simple Wikipedia", "wikivoyage": "Wikivoyage"}[CORPUS],
        "base": {"simplewiki": "https://simple.wikipedia.org/wiki/", "wikivoyage": "https://en.wikivoyage.org/wiki/"}[CORPUS]}
template = (ROOT / "src" / "map_template.html").read_text()
out = ROOT / "reports" / f"{CORPUS}_map.html"
out.write_text(template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
step(f"MAP_DONE: {out} ({out.stat().st_size / 1e6:.0f} MB)")
