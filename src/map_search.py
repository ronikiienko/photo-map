"""Semantic search for a map in the viewer: type a query in the viewer's search box and press Enter; the viewer (open on
the same bundle) tints the photos that match. The viewer leaves the query in <bundle>/query.txt; this answers in
<bundle>/search.json ({"query", "file": "search.u16"}), search.u16 holding two cosines per photo (all the image ones,
then all the caption ones; 0..1 as 0..65535, caption 0 = no caption), and keeps query.ready fresh so the viewer knows
it runs. The viewer blends the two with its slider.
    python src/map_search.py [bundle]          runs until Ctrl-C
A photo's similarity is a blend of its image vector's and its caption vector's cosine with the query (caption vectors:
search_test.py). A photo matches from CUT up; how far it is above CUT (to TOP) is its colour in the viewer, a fixed
scale. Measured on the Wikivoyage map: the best photos of real queries are 0.76-0.89, of nonsense ones
('zxqv blorf') 0.65-0.70, and 70-550 photos are above 0.72 for real queries.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import config
from search_test import CACHE

# (CUT, TOP and MIN_HITS live in the viewer's main.rs now)


if __name__ == "__main__":
    bundle = Path(sys.argv[1] if len(sys.argv) > 1 else config.ROOT / "maps/wikivoyage.photomap")
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    img = np.fromfile(bundle / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    photos = json.loads((bundle / "photos.json").read_text())
    texts = [p["meta"].get("caption", "") for p in photos]
    has = np.array([bool(t) for t in texts])
    cap = np.load(CACHE).astype(np.float32)
    model = config.load_model()
    ask, ready = bundle / "query.txt", bundle / "query.ready"
    seen = ask.stat().st_mtime_ns if ask.exists() else 0  # a query from before this started is old news
    print("ready: type a query in the viewer's search box and press Enter", flush=True)
    while True:
        ready.touch()
        time.sleep(0.25)
        if not ask.exists() or ask.stat().st_mtime_ns == seen:
            continue
        seen = ask.stat().st_mtime_ns
        query = ask.read_text().strip()
        if not query:
            continue
        v = model.encode([query], prompt=config.QUERY_PROMPT, normalize_embeddings=True)[0]
        a, b = img @ v, cap @ v
        # both cosines go to the viewer (u16: 0..65535 = 0..1; 0 for a caption that isn't there); it blends them
        # with its slider and applies CUT / TOP
        both = np.concatenate([a, np.where(has, b, 0)]).clip(0, 1)
        (bundle / "search.tmp").write_bytes(np.round(both * 65535).astype("<u2").tobytes())
        (bundle / "search.tmp").replace(bundle / "search.u16")  # before search.json: that is what the viewer waits for
        found = {"query": query, "file": "search.u16"}
        tmp = bundle / "search.json.tmp"
        tmp.write_text(json.dumps(found))
        tmp.replace(bundle / "search.json")  # whole or not there
        print(f"{query!r}: best image {a.max():.2f}, best caption {b[has].max():.2f}", flush=True)
