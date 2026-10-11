"""Map of the Wikivoyage photos: one dot per photo, similar-looking photos close together.
    python src/map_photos.py   ->  reports/wikivoyage_photo_map.html (open in a browser)
Same recipe as map_simplewiki.py: subtract the average photo, PCA to 50 numbers, UMAP to 2D, k-means regions.
The page loads the photos from data/wikivoyage/. Flatpak Firefox cannot read them from a file:// page, so serve it:
    python -m http.server 8765 --bind 127.0.0.1   then open http://127.0.0.1:8765/reports/wikivoyage_photo_map.html
"""

import base64
import json
import time

import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
INDEX = ROOT / "index" / "wikivoyage"
REGIONS = 40
NEIGHBOURS = 10
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


photos = json.loads((INDEX / "image_chunks.json").read_text())
vectors = np.load(INDEX / "image_embeddings.npy").astype(np.float32)
ok = np.where(np.linalg.norm(vectors, axis=1) > 0)[0]  # broken images were stored as all-zero rows
photos, vectors = [photos[i] for i in ok], vectors[ok]
step(f"{len(photos):,} photos")

vectors -= vectors.mean(axis=0)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
svd = TruncatedSVD(50, random_state=0)
pcs = svd.fit_transform(vectors).astype(np.float32)
del vectors
pcs /= np.linalg.norm(pcs, axis=1, keepdims=True)
step(f"PCA to 50 numbers keeps {svd.explained_variance_ratio_.sum():.0%} of the variation")

reducer = umap.UMAP(n_neighbors=NEIGHBOURS + 1, min_dist=0.05, metric="cosine", low_memory=True, random_state=0, verbose=True)
xy = reducer.fit_transform(pcs)
nn = reducer._knn_indices[:, 1 : NEIGHBOURS + 1]
nn_sim = np.einsum("ij,ikj->ik", pcs, pcs[nn])
step("UMAP done")

# Regions, named after the captions of their most typical photos
km = MiniBatchKMeans(REGIONS, random_state=0, n_init=3, batch_size=4096).fit(pcs)
labels = []
for r in range(REGIONS):
    members = np.where(km.labels_ == r)[0]
    centre = km.cluster_centers_[r] / np.linalg.norm(km.cluster_centers_[r])
    typical = members[np.argsort(-(pcs[members] @ centre))[:3]]
    labels.append({"text": " · ".join((photos[i]["text"] or photos[i]["title"])[:30] for i in typical),
                   "x": float(np.median(xy[members, 0])), "y": float(np.median(xy[members, 1])), "size": int(len(members))})
step("regions labelled")

lo, hi = xy.min(axis=0), xy.max(axis=0)
q = np.round((xy - lo) / (hi - lo) * 10000).astype(int)
for l in labels:
    l["x"] = round((l["x"] - lo[0]) / (hi[0] - lo[0]) * 10000)
    l["y"] = round((l["y"] - lo[1]) / (hi[1] - lo[1]) * 10000)
packed = lambda a: base64.b64encode(a.tobytes()).decode()
data = {"cap": [p["text"] for p in photos], "img": [p["image"] for p in photos], "art": [p.get("articles") or [p["title"]] for p in photos],
        "x": packed(q[:, 0].astype(np.uint16)), "y": packed(q[:, 1].astype(np.uint16)), "r": packed(km.labels_.astype(np.uint8)),
        "nn": packed(nn.ravel().astype(np.uint32)), "s": packed(np.clip(np.round(nn_sim.ravel() * 100), 0, 255).astype(np.uint8)),
        "k": NEIGHBOURS, "labels": labels}
template = (ROOT / "src" / "map_photos_template.html").read_text()
out = ROOT / "reports" / "wikivoyage_photo_map.html"
out.write_text(template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
step(f"MAP_DONE: {out} ({out.stat().st_size / 1e6:.0f} MB)")
