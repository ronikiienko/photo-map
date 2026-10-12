"""Photo map bundles for the photo-map viewer (photo-map/): made here, the viewer only shows them.
    python src/photomap.py regions maps/<name>.photomap     (re)name the regions of a bundle
Region names come from the photos' captions (meta "caption") where there are any, checked against the photos by the
model, and from a big general vocabulary where captions don't give one.
Data-specific scripts (map_wikivoyage.py) gather photos, vectors and text, then call make().

A bundle, maps/<name>.photomap/:
    manifest.json     name, photo count, model
    photos.json       per photo: file (originals/NNNNNN.ext), source, width, height, title, meta (strings: caption,
                      article, ... shown in search)
    originals/        the images, flat: photo i is originals/<i, 6 digits>.<ext> (hard links where possible)
    embeddings.bin    per photo: vector, float16
    layout.bin        per photo: x, y (float32, 0 ... 10000); similar photos close together
    regions.json      named regions per zoom level, coarse first: {"levels": [[{name, x, y, spread, photos}]]}
    pages_bc7.bin     baked by the viewer the first time it opens the bundle
"""

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

WORLD = 10_000
VOCAB = config.ROOT / "data" / "vocab"
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


def size(path):
    """Width, height as shown (EXIF rotation applied); None if unreadable."""
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).size if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else im.size
    except Exception:
        return None


def make(bundle, photos, files, vectors, model=config.MODEL_ID):
    """photos: per photo {title, source, meta}; files: the image files; vectors: n x dim (normalized).
    Unreadable images are left out. Writes a new bundle (an old one at the same path is replaced)."""
    bundle = Path(bundle)
    new = bundle.with_name(bundle.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    (new / "originals").mkdir(parents=True)
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(size, files))
    keep = [i for i, s in enumerate(sizes) if s]
    step(f"{len(keep):,} readable images ({len(files) - len(keep)} left out)")

    out = []
    for i in keep:
        out.append({"file": f"originals/{len(out):06d}{Path(files[i]).suffix.lower()}", "width": sizes[i][0], "height": sizes[i][1], **photos[i]})

    def place(k):
        src, dst = files[keep[k]], new / out[k]["file"]
        try:
            os.link(src, dst)  # same disk: no copy needed
        except OSError:
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(place, range(len(out))))
    step("originals in place")
    vectors = np.asarray(vectors, np.float32)[keep]
    (new / "photos.json").write_text(json.dumps(out, ensure_ascii=False))
    vectors.astype(np.float16).tofile(new / "embeddings.bin")
    xy = lay_out(vectors)
    xy.tofile(new / "layout.bin")
    regions(new, vectors, xy, [p["meta"].get("caption", "") for p in out])
    manifest = {"format": 1, "name": bundle.name.removesuffix(".photomap"), "created": time.strftime("%Y-%m-%d %H:%M"),
                "photos": len(out), "world": WORLD, "embedding": {"model": model, "dim": vectors.shape[1], "dtype": "float16"}}
    (new / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(bundle, ignore_errors=True)
    new.rename(bundle)
    step(f"MAP_DONE: {bundle} (the viewer bakes it the first time it opens it)")


def lay_out(vectors):
    """Centre, PCA to 50, UMAP to 2D, scaled to 0 ... WORLD."""
    import umap
    from sklearn.decomposition import TruncatedSVD

    n = len(vectors)
    step(f"layout of {n:,} photos (PCA + UMAP)")
    v = vectors - vectors.mean(axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    pcs = TruncatedSVD(min(50, n - 1, v.shape[1] - 1), random_state=0).fit_transform(v).astype(np.float32)
    pcs /= np.maximum(np.linalg.norm(pcs, axis=1, keepdims=True), 1e-9)
    xy = umap.UMAP(n_neighbors=min(11, n - 1), min_dist=0.05, metric="cosine", low_memory=True, random_state=0).fit_transform(pcs)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    step("layout done")
    return ((xy - lo) / np.maximum(hi - lo, 1e-9) * WORLD).astype(np.float32)


# ---- regions: connected areas of the map at several zoom levels, each named ----

STEP, LEAST = 3, 10  # each level has 3x the regions of the one above, down to ~10 photos a region
MIN_SHARE, MIN_PHOTOS = 0.01, 10  # coarsest regions under 1% of the photos, other regions under 10 photos: no name
TOP = 10  # every photo votes for its 10 best-fitting words


def cluster(xy):
    """Per level, a region number per photo. Small cells (k-means, ~8 photos each) are merged, always with a
    neighbouring cell (Delaunay), into one tree (Ward: the merge that keeps groups most compact first); cutting it at
    several heights gives the levels. Every region is one connected piece of the map, finer ones nest in coarser ones."""
    from scipy.sparse import coo_matrix
    from scipy.spatial import Delaunay
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

    n = len(xy)
    km = MiniBatchKMeans(int(np.clip(n // 8, 20, 12_000)), random_state=0, n_init=1, batch_size=8192).fit(xy)
    used, cell_of = np.unique(km.labels_, return_inverse=True)
    centers = km.cluster_centers_[used]
    m = len(centers)
    tri = Delaunay(centers)
    a = np.concatenate([tri.simplices[:, [i, j]] for i, j in ((0, 1), (1, 2), (0, 2))])
    adj = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(m, m))
    tree = AgglomerativeClustering(n_clusters=1, linkage="ward", connectivity=adj + adj.T, compute_full_tree=True).fit(centers)

    sizes = [max(4, round(np.sqrt(n) / 25))]
    while sizes[-1] * STEP * LEAST <= n and sizes[-1] * STEP < m:
        sizes.append(sizes[-1] * STEP)
    parent, merged, levels = np.arange(2 * m - 1), 0, []
    for k in sorted(sizes, reverse=True):  # apply merges until k groups are left
        while m - merged > k:
            x, y = tree.children_[merged]
            parent[x] = parent[y] = m + merged
            merged += 1
        root = parent.copy()
        while not np.array_equal(root, root[root]):
            root = root[root]
        levels.append(np.unique(root[:m], return_inverse=True)[1][cell_of])
    return levels[::-1]


def vocabulary():
    """One big general vocabulary: Places365 scenes ('broadleaf forest') and all ~20k Open Images labels."""
    words = []
    for line in (VOCAB / "places365.txt").read_text().splitlines():
        parts = line.split()[0].split("/")[2:]  # '/f/forest/broadleaf 0' -> ['forest', 'broadleaf']
        words.append(" ".join(parts[1:] + parts[:1]).replace("_", " "))
    for line in (VOCAB / "openimages_all.csv").read_text().splitlines():
        words.append(line.split(",", 1)[1].strip().strip('"').lower())
    return list(dict.fromkeys(w for w in words if w))


def word_vectors(words):
    """Text vectors (as search queries), cached in data/vocab/text_cache.*; the model loads only for new words."""
    names, vecs = VOCAB / "text_cache.json", VOCAB / "text_cache.f32"
    cached = json.loads(names.read_text()) if names.exists() else []
    known = np.fromfile(vecs, np.float32).reshape(len(cached), -1) if cached else np.zeros((0, 768), np.float32)
    index = {w: i for i, w in enumerate(cached)}
    new = [w for w in words if w not in index]
    if new:
        step(f"embedding {len(new):,} new words")
        model = config.load_model()
        extra = model.encode(new, prompt=config.QUERY_PROMPT, batch_size=128, normalize_embeddings=True).astype(np.float32)
        cached, known = cached + new, np.concatenate([known, extra])
        names.write_text(json.dumps(cached))
        known.tofile(vecs)
        index = {w: i for i, w in enumerate(cached)}
    return known[[index[w] for w in words]]


def forms(word):
    """A word with plural/singular endings, so 'lake' and 'lakes' count as the same name."""
    if not word:
        return set()
    out = {word, word + "s", word + "es"}
    if word.endswith("es"):
        out.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        out.add(word[:-1])
    return out


def votes(vectors, wvecs):
    """Each photo's TOP best words (photo and word vectors both centred, so generic words don't win everywhere)."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.nn.functional.normalize(torch.tensor(vectors, device=dev) - torch.tensor(vectors.mean(0), device=dev), dim=1)
    w = torch.tensor(wvecs, device=dev)
    w = torch.nn.functional.normalize(w - w.mean(0), dim=1)
    return torch.cat([(p[i : i + 8192] @ w.T).topk(TOP, dim=1).indices for i in range(0, len(p), 8192)]).cpu().numpy()


CAPTION_MIN = 3  # a caption phrase needs at least 3 of a region's captions ...
COVERAGE = 0.5  # ... and half as many as the region's most common phrase: a name has to describe the region, not a few of its photos
SUBSUMED = 0.7  # a phrase gives way to a longer one found in 70% of its captions
FIT_MIN = 0.0  # ... and has to fit the region's photos better than the average photo
STOP = set("""a an the and or of in on at to for from with by as is are was were be been this that these those it its
into over under near after before during between about above below up down out off via per no not new old de la le el di
da del des du van von der den und one two three view views see seen here there some many most also very just looking
towards toward inside outside front back top left right side part photo image picture which where while""".split())


def phrases(text):
    """The words and 2-3 word phrases of a caption that could be a name (no leading or trailing filler words)."""
    w = [x for x in re.findall(r"[a-z][a-z'-]+", text.lower()) if len(x) > 2]
    return {" ".join(w[i : i + k]) for k in (1, 2, 3) for i in range(len(w) - k + 1) if w[i] not in STOP and w[i + k - 1] not in STOP}


def members(lab, R):
    """The photos of each region 0 ... R-1 (photos labelled -1 belong to none)."""
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(R + 1))
    return [order[bounds[r] : bounds[r + 1]] for r in range(R)]


def caption_candidates(levels, texts, centred):
    """Per level, per region: [(score, phrase)], best first. A phrase scores by how common it is in the region's
    captions compared with all captions (share here * log(share here / share everywhere)), a little more for longer
    phrases; it has to be in CAPTION_MIN captions and fit the region's photos (the model's check: the phrase's text
    vector points more towards the region's photos than towards the average photo)."""
    from collections import Counter

    grams = [phrases(t) for t in texts]
    has = np.array([bool(t) for t in texts])
    df, ncap = Counter(g for s in grams for g in s), has.sum()
    out = []
    for lab in levels:
        per = {}
        for r, idx in enumerate(members(np.where(has, lab, -1), lab.max() + 1)):
            if len(idx) < CAPTION_MIN:
                continue
            count = Counter(g for i in idx for g in grams[i])
            count = {g: v for g, v in count.items() if v >= max(CAPTION_MIN, 0.03 * len(idx)) and df[g] < 0.2 * ncap}
            top = max(count.values(), default=0)
            count = {g: v for g, v in count.items() if v >= COVERAGE * top}
            covered = set()  # parts of a longer phrase that takes most of their uses: 'main' in 'main street'
            for h, v in count.items():
                w = h.split()
                for sub in (" ".join(w[i : i + k]) for k in range(1, len(w)) for i in range(len(w) - k + 1)):
                    if sub in count and v >= SUBSUMED * count[sub]:
                        covered.add(sub)
            share = {g: v / len(idx) for g, v in count.items() if g not in covered}
            score = {g: s * s * np.log(s * ncap / df[g]) * (1 + 0.3 * g.count(" ")) for g, s in share.items()}
            per[r] = sorted(((v, g) for g, v in score.items()), reverse=True)[:20]
        out.append(per)
    every = sorted({g for per in out for c in per.values() for _, g in c})
    vec = dict(zip(every, word_vectors(every)))
    kept = dropped = 0
    for lab, per in zip(levels, out):
        means = np.stack([centred[idx].mean(0) for idx in members(lab, lab.max() + 1)])
        for r, c in per.items():
            per[r] = [(v, g) for v, g in c if means[r] @ vec[g] > FIT_MIN]
            kept, dropped = kept + len(per[r]), dropped + len(c) - len(per[r])
    step(f"caption names: {len(every):,} phrases; {dropped:,} of {kept + dropped:,} candidates don't fit their photos")
    return out


def vocabulary_candidates(levels, best, words, min_share):
    """Per level, per region: [(score, word)] from the photos' votes (votes()): the words its photos vote for most
    compared with the whole map (share here * log(share here / share everywhere)), with at least 3 votes."""
    from scipy.sparse import coo_matrix

    n, W = len(best), len(words)
    glob = np.bincount(best.ravel(), minlength=W) / n
    out = []
    for lab in levels:
        R = lab.max() + 1
        size = np.bincount(lab, minlength=R)
        counts = coo_matrix((np.ones(best.size), (np.repeat(lab, TOP), best.ravel())), shape=(R, W)).tocsr()
        per = {}
        for r in range(R):
            row = counts.getrow(r)
            share = row.data / size[r]
            ok = (row.data >= 3) & (share >= min_share)
            idx, share = row.indices[ok], share[ok]
            lift = share * np.log(share / glob[idx])
            per[r] = [(lift[j], words[idx[j]]) for j in np.argsort(-lift)[:30]]
        out.append(per)
    return out


def name(levels, sources):
    """Per level: {region: name}, from candidate lists (caption_candidates, vocabulary_candidates); the first source is
    used first, later ones only for regions still without a name. Within a source, best scores are assigned first; a
    name is used once among regions sharing a parent and never repeats a name of a region this one sits in. Small
    regions, and regions with exactly the photos of the one around them, get none."""
    n = len(levels[0])
    names = []
    for lv, lab in enumerate(levels):
        R = lab.max() + 1
        size = np.bincount(lab, minlength=R)
        rep = np.zeros(R, int)
        rep[lab] = np.arange(n)  # one photo of each region; regions nest, so its regions above are the region's
        parent = levels[lv - 1][rep] if lv else np.zeros(R, int)
        same = size == np.bincount(levels[lv - 1])[parent] if lv else np.zeros(R, bool)
        above = [set().union(*(forms(names[j].get(levels[j][rep[r]], "")) for j in range(lv))) for r in range(R)]
        eligible = (size >= (MIN_SHARE * n if lv == 0 else MIN_PHOTOS)) & ~same
        chosen, used, counts = {}, {}, []
        for source in sources:
            cands = sorted((v, r, w) for r, c in source[lv].items() if eligible[r] and r not in chosen for v, w in c)[::-1]
            before = len(chosen)
            for _, r, w in cands:
                f = forms(w)
                group = used.setdefault(parent[r], set())
                if r in chosen or f & group or f & above[r]:
                    continue
                chosen[r] = w
                group |= f
            counts.append(len(chosen) - before)
        names.append(chosen)
        step(f"level {lv + 1}: {len(chosen):,} of {R:,} regions named ({' + '.join(f'{c:,}' for c in counts)})")
    return names


def regions(bundle, vectors, xy, texts=None):
    """Names for the map's regions at several zoom levels -> regions.json. With texts (a caption per photo, may be
    empty), names come from the captions alone (a region without a good caption phrase stays unnamed); without them, from the general vocabulary."""
    bundle = Path(bundle)
    texts = texts if texts and any(texts) else None
    levels = cluster(xy)
    step(f"regions: levels of {[int(l.max()) + 1 for l in levels]}")
    if texts:
        centred = vectors - vectors.mean(0)
        sources = [caption_candidates(levels, texts, centred / np.linalg.norm(centred, axis=1, keepdims=True))]
    else:  # no captions: the model's own vocabulary is all there is
        words = vocabulary()
        sources = [vocabulary_candidates(levels, votes(vectors, word_vectors(words)), words, 0.0)]
    names = name(levels, sources)
    out = []
    for lab, named in zip(levels, names):
        level = []
        for r, nm in named.items():
            pts = xy[lab == r]
            c = pts.mean(0)
            d = ((pts - c) ** 2).sum(1)
            x, y = pts[d.argmin()]  # label at the photo nearest the centre
            level.append({"name": nm, "x": float(x), "y": float(y), "spread": float(np.sqrt(d.mean())), "photos": len(pts)})
        out.append(sorted(level, key=lambda l: -l["photos"]))
    (bundle / "regions.json").write_text(json.dumps({"levels": out}, ensure_ascii=False))
    step(f"regions.json: {sum(map(len, out)):,} names on {len(out)} levels")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "regions":
        sys.exit(__doc__)
    b = Path(sys.argv[2])
    n = json.loads((b / "manifest.json").read_text())["photos"]
    vecs = np.fromfile(b / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    texts = [p["meta"].get("caption", "") for p in json.loads((b / "photos.json").read_text())]
    regions(b, vecs, np.fromfile(b / "layout.bin", np.float32).reshape(n, 2), texts)
