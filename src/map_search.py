"""Semantic search for a map in the viewer: type a query in the viewer's search box and press Enter, or drop a photo on
the window (or both: the photo and the words are searched together); the viewer (open on the same bundle) tints the
photos that match. The viewer leaves the query in <bundle>/query.txt as JSON {"text", "image": a file path or null}; this answers in
<bundle>/search.json ({"query", "file": "search.u16"}), search.u16 holding two cosines per photo (all the image ones,
then all the caption ones; 0..1 as 0..65535, caption 0 = no caption), and keeps query.ready fresh so the viewer knows
it runs. The viewer takes the better of the two and applies its cutoff.
One helper serves whatever map the viewer has open: the viewer writes the map's path to logs/search.bundle when it
opens another, and the helper (model already loaded) loads that map's vectors; the last two maps stay loaded.
    python src/map_search.py [bundle]          runs until Ctrl-C (the viewer starts it itself)
A photo's similarity is the better of its image vector's and its caption vector's cosine with the query (caption
vectors: search_test.py). A photo matches from the viewer's cutoff up (0.72 to start); how far above it is its colour. Measured on the Wikivoyage map: the best photos of real queries are 0.76-0.89, of nonsense ones
('zxqv blorf') 0.65-0.70, and 70-550 photos are above 0.72 for real queries.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from search_test import CACHE

# (CUT, TOP and MIN_HITS live in the viewer's main.rs now)


CONTROL = config.ROOT / "logs/search.bundle"  # the viewer writes the path of the map it has open here
KEEP = 2  # maps kept loaded: going back and forth between two is instant


def load(bundle):
    """A map's vectors: its photos' image vectors, the caption vectors (the Wikivoyage map has them) and who has a caption."""
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    img = np.fromfile(bundle / "embeddings.bin", np.float16).reshape(n, -1).astype(np.float32)
    photos = json.loads((bundle / "photos.json").read_text())
    has = np.array([bool(p["meta"].get("caption", "")) for p in photos])
    # caption vectors exist for the Wikivoyage map (search_test.py); a map with no captions is searched by its images alone
    cap = np.load(CACHE).astype(np.float32) if has.any() else np.zeros_like(img)
    if len(cap) != n:  # caption vectors of another map
        print(f"{bundle.name}: no caption vectors ({len(cap)} != {n} rows): images only", flush=True)
        cap, has = np.zeros_like(img), np.zeros(n, bool)
    return {"img": img, "cap": cap, "has": has}


BIOCLIP = "imageomics/bioclip-2"
models = {}  # name -> (text(query), photo(PIL image)) -> vector: each model is loaded when a map made with it is opened


def gemma():
    model = config.load_model(images=True)
    model[0].processor.image_processor.max_soft_tokens = 140  # as when the photos were embedded
    return (
        lambda text: model.encode([text], prompt=config.QUERY_PROMPT, normalize_embeddings=True)[0],
        lambda im: model.encode([{"image": im}], normalize_embeddings=True)[0],
    )


def bioclip():
    import open_clip

    net, _, preprocess = open_clip.create_model_and_transforms("hf-hub:" + BIOCLIP)
    net = net.half().cuda().eval()  # as when the photos were embedded
    tokenizer = open_clip.get_tokenizer("hf-hub:" + BIOCLIP)

    def text(t):
        with torch.no_grad():
            return net.encode_text(tokenizer([t]).cuda(), normalize=True)[0].float().cpu().numpy()

    def photo(im):
        with torch.no_grad():
            return net.encode_image(preprocess(im)[None].half().cuda(), normalize=True)[0].float().cpu().numpy()

    return text, photo


def model_for(bundle):
    """The query encoders of the model the map's photos were embedded with (its manifest says which)."""
    name = json.loads((bundle / "manifest.json").read_text())["embedding"]["model"]
    key = BIOCLIP if name == BIOCLIP else "gemma"
    if key not in models:
        print(f"loading {key} for {bundle.name}", flush=True)
        models[key] = bioclip() if key == BIOCLIP else gemma()
    return models[key]


if __name__ == "__main__":
    first = Path(sys.argv[1] if len(sys.argv) > 1 else config.ROOT / "maps/wikivoyage.photomap").resolve()
    loaded = {}  # bundle -> load(bundle), the KEEP most recently used
    now = {"bundle": None, "seen": 0}  # the map being served; its query.txt time when it was taken up (older: old news)

    def take_up(bundle):
        if bundle not in loaded:
            loaded[bundle] = {**load(bundle), "model": model_for(bundle)}
        loaded[bundle] = loaded.pop(bundle)  # most recent last
        while len(loaded) > KEEP:
            del loaded[next(iter(loaded))]
        ask = bundle / "query.txt"
        now["seen"] = ask.stat().st_mtime_ns if ask.exists() else 0
        now["bundle"] = bundle  # from here the heartbeat is for this map: the viewer's "search ready"
        print(f"serving {bundle.name}", flush=True)

    take_up(first)
    control_seen = CONTROL.stat().st_mtime_ns if CONTROL.exists() else 0  # a switch from before this started is old news
    print("ready: type a query in the viewer's search box and press Enter", flush=True)

    parent = os.getppid()

    def beat():  # the viewer calls the helper dead when this is 3 s old: keep it fresh during a slow search too
        while True:
            if now["bundle"]:
                (now["bundle"] / "query.ready").touch()
            if os.getppid() != parent:  # whoever started this is gone (the viewer crashed or was killed): stop too
                os._exit(0)
            time.sleep(0.5)

    threading.Thread(target=beat, daemon=True).start()
    while True:
        time.sleep(0.25)
        if CONTROL.exists() and CONTROL.stat().st_mtime_ns != control_seen:  # the viewer opened another map
            control_seen = CONTROL.stat().st_mtime_ns
            wanted = Path(CONTROL.read_text().strip())
            if wanted != now["bundle"] and (wanted / "manifest.json").exists():
                try:
                    take_up(wanted)
                except Exception as e:
                    print(f"can't serve {wanted}: {e}", flush=True)
        bundle = now["bundle"]
        data = loaded[bundle]
        img, cap, has = data["img"], data["cap"], data["has"]
        embed_text, embed_photo = data["model"]
        ask = bundle / "query.txt"
        if not ask.exists() or ask.stat().st_mtime_ns == now["seen"]:
            continue
        now["seen"] = ask.stat().st_mtime_ns
        raw = ask.read_text().strip()
        try:
            asked = json.loads(raw)  # {"text": ..., "image": path or null} from the viewer
        except json.JSONDecodeError:
            asked = {"text": raw}
        text, image = (asked.get("text") or "").strip(), asked.get("image")
        if not text and not image:
            continue
        try:
            parts = []
            if text:
                parts.append(embed_text(text))
            if image:  # a photo is embedded like the map's photos: no prompt
                with Image.open(image) as im:
                    parts.append(embed_photo(im.convert("RGB")))
        except Exception as e:  # unreadable photo etc.: tell the viewer, keep running
            print(f"{asked}: {e}", flush=True)
            tmp = bundle / "search.json.tmp"
            tmp.write_text(json.dumps({"error": f"couldn't search: {e}"[:120]}))
            tmp.replace(bundle / "search.json")
            continue
        v = np.sum(parts, axis=0)  # text and photo together: the vector between them
        v /= np.linalg.norm(v)
        query = " + ".join(filter(None, [text, Path(image).name if image else ""]))
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
        print(f"{query!r}: best image {a.max():.2f}" + (f", best caption {b[has].max():.2f}" if has.any() else ""), flush=True)
