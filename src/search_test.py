"""Test: does searching the caption vectors too help? For some queries, the top photos by the image vectors alone,
by the caption vectors alone, and by both together.
    python src/search_test.py
The caption vectors (one per photo with a caption, as text) are made once and kept in index/wikivoyage/.
"""

import json

import numpy as np

import config
import photomap

BUNDLE = config.ROOT / "maps/wikivoyage.photomap"
CACHE = config.ROOT / "index/wikivoyage/bundle_caption_vectors.npy"
QUERIES = [
    "windmill", "night market", "old church", "airport", "Assumption Cathedral", "Niagara Falls",
    "street food", "wooden houses", "snow-covered mountain village", "road sign in Cyrillic",
    "ancient ruins", "bridge at sunset", "hot spring", "train station platform", "tourist information map",
]
SHOW = 8


def z(s):
    return (s - s.mean()) / s.std()


if __name__ == "__main__":
    n = json.loads((BUNDLE / "manifest.json").read_text())["photos"]
    img = np.fromfile(BUNDLE / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    photos = json.loads((BUNDLE / "photos.json").read_text())
    texts = [p["meta"].get("caption", "") for p in photos]
    has = np.array([bool(t) for t in texts])
    model = config.load_model()
    model.max_seq_length = 128  # captions are 8 tokens on average; the few very long ones are cut
    if CACHE.exists():
        cap = np.load(CACHE).astype(np.float32)
    else:
        photomap.step(f"embedding {has.sum():,} captions")
        cap = np.zeros((n, img.shape[1]), np.float32)
        todo = np.flatnonzero(has)
        order = todo[np.argsort([len(texts[i]) for i in todo])]  # similar lengths together: less padding
        cap[order] = model.encode([texts[i] for i in order], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        np.save(CACHE, cap.astype(np.float16))
        photomap.step("caption vectors saved")
    q = model.encode(QUERIES, prompt=config.QUERY_PROMPT, normalize_embeddings=True)
    for query, v in zip(QUERIES, q):
        s_img = z(img @ v)
        s_cap = np.where(has, z(np.where(has, cap @ v, (cap @ v)[has].mean())), s_img)  # no caption: image score stands in
        s_both = (s_img + s_cap) / 2
        print(f"\n### {query}")
        for name, s in (("image", s_img), ("caption", s_cap), ("both", s_both)):
            top = np.argsort(-s)[:SHOW]
            print(f"  {name}:")
            for i in top:
                print(f"    {s[i]:5.1f}  {(texts[i] or '(no caption) ' + photos[i]['title'])[:70]}")
        a, b, c = (set(np.argsort(-s)[:50]) for s in (s_img, s_cap, s_both))
        print(f"  top-50 overlap: image/caption {len(a & b)}, image/both {len(a & c)}, caption/both {len(b & c)}")
