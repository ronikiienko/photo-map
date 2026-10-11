"""Prototype: names for the regions of a photo map, at 3 levels of detail (coarse / middle / fine).
    python src/name_regions.py maps/<name>.photomap   ->  reports/regions_<name>_<level>.png + regions_<name>.txt
1. Regions: k-means on the 2D map positions into small cells, then Ward clustering of neighbouring cells (Delaunay
   neighbours only) into one tree. Every region is one connected piece of the map, and the levels nest.
2. Names: words and phrases embedded as text queries. A region is named by the word pointing most in the direction
   its photos differ from the average photo; each name is used once per level and never repeats the name of the
   region it sits in. The coarse level chooses from broad themes only; finer levels also from Places365 scenes,
   Open Images objects and words that are common in the map's own titles and descriptions.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

import config

VOCAB = config.ROOT / "data" / "vocab"
PROMPT = "task: search result | query: "
BROAD = [
    "nature", "landscape", "mountains", "water", "sea", "coast", "beach", "islands", "rivers and lakes", "forest",
    "desert", "snow and ice", "sky", "night", "sunset", "plants and flowers", "animals", "birds", "wildlife",
    "underwater", "insects", "city", "streets", "buildings", "architecture", "skyscrapers", "old town", "villages",
    "countryside", "farmland", "roads", "transport", "trains and railways", "boats and ships", "aircraft", "cars",
    "bridges", "religion", "churches", "temples", "mosques", "castles and fortresses", "ruins", "monuments",
    "museums", "art", "sculpture", "paintings", "interiors", "people", "crowds", "festivals", "food", "markets",
    "shops", "sports", "maps", "flags", "signs and text", "documents", "diagrams", "logos and symbols", "history",
    "space", "planets", "stars and galaxies", "rockets", "astronauts", "science and technology", "machines",
    "industry", "military", "parks and gardens", "caves", "volcanoes", "waterfalls", "harbours", "airports",
    "the sun", "the moon", "earth from space", "satellite images", "aerial views", "spacesuits", "spacecraft",
    "control rooms", "laboratories", "offices", "meetings", "computers", "engineering", "construction", "vehicles",
    "portraits", "groups of people", "children", "clouds and weather", "storms", "fields and meadows", "hills",
    "glaciers", "lakes", "rivers", "islands and atolls", "coral reefs", "fish", "mammals", "pets", "cattle and farm animals",
    "horses", "elephants", "monkeys", "bears", "big cats", "reptiles", "jungle", "savanna", "canyons", "rock formations",
    "houses", "palaces", "towers", "squares and plazas", "nightlife", "restaurants and cafes", "hotels", "beaches and resorts",
    "skiing and winter sports", "hiking", "camping", "music and performance", "dance", "traditional dress", "crafts",
    "textiles", "ceramics", "statues", "graves and cemeteries", "memorials", "stations", "ports", "lighthouses",
]
STOP = set("""a an the and or of in on at to for from with by as is are was were be been this that these those it its
into over under near after before during between about above below up down out off via per vs no not new old de la le
el di da del des du van von der den und photo photos image images picture view views file jpg png""".split())
DROP = {"indoor", "outdoor", "interior", "exterior", "home", "public", "shop", "natural", "urban", "indoor_procenium",
        "indoor_seats", "inside", "outside", "office", "house", "vehicle", "station", "platform"}
FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
start = time.time()


def step(msg):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)


def title_words(photos, n):
    """Words and word pairs common in the map's own titles and descriptions (in at least 0.5% of photos, at most 20%)."""
    count = {}
    for p in photos:
        text = " ".join([p["title"]] + [v for k, v in p["meta"].items() if k in ("description", "comment")]).lower()
        toks = [t for t in "".join(c if c.isalpha() else " " for c in text).split() if len(t) > 2 and t not in STOP]
        for g in set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}:
            count[g] = count.get(g, 0) + 1
    lo, hi = max(4, n * 0.005), n * 0.2
    return sorted(g for g, c in count.items() if lo <= c <= hi)


def vocabulary():
    words = []
    for line in (VOCAB / "places365.txt").read_text().splitlines():
        parts = line.split()[0].split("/")[2:]  # '/f/forest/broadleaf 0' -> ['forest', 'broadleaf']
        base, quals = parts[0], [q for q in parts[1:] if q not in DROP]
        words.append(" ".join(quals + [base]).replace("_", " "))  # 'broadleaf forest'
    objects = [line.split(",", 1)[1].split("(")[0].strip().lower() for line in (VOCAB / "openimages.csv").read_text().splitlines()]
    words = list(dict.fromkeys(w for w in words + objects + BROAD if w))
    return words, set(objects) - set(words[: len(words) - len(objects)])


def word_vectors(words):
    """Text vectors, cached per word in data/vocab/ (the model only loads when there are new words)."""
    cache_words, cache_vecs = VOCAB / "words.json", VOCAB / "word_vectors.npy"
    known = dict(zip(json.loads(cache_words.read_text()), np.load(cache_vecs))) if cache_words.exists() else {}
    new = [w for w in words if w not in known]
    if new:
        model = config.load_model()
        known.update(zip(new, model.encode(new, prompt=PROMPT, batch_size=64, normalize_embeddings=True)))
        cache_words.write_text(json.dumps(list(known)))
        np.save(cache_vecs, np.stack(list(known.values())))
    return np.stack([known[w] for w in words])


def regions(xy, sizes):
    """One region label per photo for each size; regions are connected and nest (coarser = unions of finer)."""
    n = len(xy)
    cells = min(2000, max(20, n // 20))
    km = MiniBatchKMeans(cells, random_state=0, n_init=3, batch_size=4096).fit(xy)
    used = np.unique(km.labels_)
    centers = km.cluster_centers_[used]
    cell_of = np.searchsorted(used, km.labels_)
    tri = Delaunay(centers)
    a = np.concatenate([tri.simplices[:, [i, j]] for i, j in ((0, 1), (1, 2), (0, 2))])
    adj = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(len(centers),) * 2)
    adj = adj + adj.T
    out = []
    for k in sizes:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=adj).fit(centers).labels_
        out.append(lab[cell_of])
    return out


OBJECT_WEIGHT = 0.85  # object names (Open Images) lose to scene and theme words unless clearly better
MIN_SHARE = (0.01, 0.002, 0.002)  # regions smaller than this share of all photos get no name, per level


def forms(word):
    """A word with its plural/singular endings off, so 'lake' and 'lakes' count as the same name."""
    if not word:
        return set()
    out = {word}
    if word.endswith("es"):
        out.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        out.add(word[:-1])
    return out | {word + "s", word + "es"}


def name(vecs, labels, words, wvecs, objects):
    """Per level: {region: name}, plus the top candidates of each region for the report. Best matches are assigned
    first; tiny regions, and regions that are the same photos as the one they sit in, get no name of their own."""
    mean_all = vecs.mean(0)
    broad = np.isin(words, BROAD)
    weight = np.where(np.isin(words, list(objects)), OBJECT_WEIGHT, 1.0)
    names, report = [], []
    for level, lab in enumerate(labels):
        regs = np.unique(lab)
        diffs = np.stack([vecs[lab == r].mean(0) - mean_all for r in regs])
        scores = diffs @ wvecs.T * weight  # regions x words
        if level == 0:
            scores[:, ~broad] = -np.inf  # coarse regions get broad themes
        above = {r: set() for r in regs}  # names of the regions this one sits in (as forms(), see below)
        same = set()  # regions with exactly the photos of the one they sit in
        for up in range(level):
            for r in regs:
                inside = labels[up][lab == r]
                parent = np.bincount(inside).argmax()
                above[r] |= forms(names[up][parent])
                if (labels[up] == parent).sum() == len(inside):
                    same.add(r)
        size = np.array([(lab == r).sum() for r in regs])
        named = [i for i in range(len(regs)) if size[i] >= MIN_SHARE[level] * len(lab) and regs[i] not in same]
        cands = sorted(((scores[i, j], i, j) for i in named for j in np.argsort(-scores[i])[:60]), reverse=True)
        chosen, used = {}, set()
        for _, i, j in cands:
            f = forms(words[j])
            if i in chosen or f & used or f & above[regs[i]]:
                continue
            chosen[i] = words[j]
            used |= f
        names.append({r: chosen.get(i) for i, r in enumerate(regs)})
        for i, r in enumerate(regs):
            top = ", ".join(f"{words[j]} {scores[i, j]:.3f}" for j in np.argsort(-scores[i])[:6])
            report.append(f"level {level + 1} region {r:3d} ({(lab == r).sum():5d} photos) -> {names[-1][r] or '-':<28} | {top}")
    return names, report


def render(xy, labels, names, level, path):
    W, M = 2400, 40
    lo, hi = xy.min(0), xy.max(0)
    px = (xy - lo) / (hi - lo).max() * (W - 2 * M) + M
    img = Image.new("RGB", (W, W), (21, 23, 28))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    lab = labels[level]
    colours = {r: tuple(int(c) for c in rng.integers(70, 200, 3)) for r in np.unique(lab)}
    for (x, y), r in zip(px, lab):
        d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=colours[r])
    sizes = {0: 40, 1: 26, 2: 16}
    for lv in range(level + 1):  # this level's names, with the coarser ones on top
        font = ImageFont.truetype(FONT, sizes[lv] if lv == level else sizes[lv] + 6)
        fill = (255, 255, 255) if lv == level else (255, 220, 120)
        for r, nm in names[lv].items():
            if nm is None:
                continue
            pts = px[labels[lv] == r]
            cx, cy = pts[np.argmin(((pts[:, None] - pts[None, :: max(1, len(pts) // 400)]) ** 2).sum(-1).sum(1))]  # most central photo
            d.text((cx, cy), nm, font=font, fill=fill, anchor="mm", stroke_width=3 if lv < level else 2, stroke_fill=(0, 0, 0))
    img.save(path)


if __name__ == "__main__":
    bundle = Path(sys.argv[1] if len(sys.argv) > 1 else config.ROOT / "maps" / "wikivoyage.photomap")
    stem = bundle.stem
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    vecs = np.fromfile(bundle / "embeddings.bin", np.float16).reshape(n, 768).astype(np.float32)
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    sizes = [max(4, round(np.sqrt(n) / 25)), max(8, round(np.sqrt(n) / 8)), max(14, round(np.sqrt(n) / 2.5))]
    step(f"{stem}: {n} photos, levels of {sizes} regions")

    photos = json.loads((bundle / "photos.json").read_text())
    own = title_words(photos, n)
    general, objects = vocabulary()
    words = list(dict.fromkeys(general + own))
    wvecs = word_vectors(words)
    step(f"{len(words)} words embedded ({len(own)} from the map's own titles)")
    labels = regions(xy, sizes)
    step("regions found")
    names, report = name(vecs, labels, words, wvecs, objects)
    out = config.ROOT / "reports"
    (out / f"regions_{stem}.txt").write_text("\n".join(report) + "\n")
    for level in range(3):
        render(xy, labels, names, level, out / f"regions_{stem}_{level + 1}.png")
    step(f"REGIONS_DONE: reports/regions_{stem}_1..3.png, reports/regions_{stem}.txt")
