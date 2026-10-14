"""About 1000 public-domain photos from NASA's Image and Video Library (images-api.nasa.gov: no key, no tight rate
limit), spread over topics so a photo map shows distinct groups.
    python src/fetch_nasa.py   ->  data/nasa/images/<title> (<nasa_id>).jpg   (1280 px "medium" versions)
The title goes into the file name: photo-map uses the name as the title when the file carries none.
Rerun to resume: files already on disk are skipped.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nasa" / "images"
PER_TOPIC = 50
TOPICS = [
    "galaxy", "nebula", "mars surface", "moon surface", "jupiter", "saturn", "sun solar flare", "comet asteroid",
    "hurricane from space", "aurora", "rocket launch", "space shuttle", "spacewalk", "space station interior",
    "rover", "aircraft flight", "wind tunnel", "telescope mirror", "mission control", "earth city lights",
]
HEADERS = {"User-Agent": "photo-map test (personal project)"}


def get(url, timeout=60):
    url = urllib.parse.quote(url, safe=":/?=&~%")  # some NASA ids contain spaces
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return r.read()


def search(topic):
    q = urllib.parse.urlencode({"q": topic, "media_type": "image", "page_size": PER_TOPIC, "page": 1})
    items = json.loads(get(f"https://images-api.nasa.gov/search?{q}"))["collection"]["items"]
    out = []
    for it in items:
        data, links = it["data"][0], it.get("links") or []
        href = next((l["href"] for l in links if l.get("rel") == "preview"), None)
        if href:
            out.append((data["nasa_id"], data.get("title", ""), href.replace("~thumb.", "~medium."), it["href"]))
    return out


def file_name(nasa_id, title):
    title = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", title).strip()[:80].strip()
    return f"{title} ({nasa_id}).jpg" if title else f"{nasa_id}.jpg"


def best_file(manifest_url):
    """When there's no ~medium (often for JPL's PIA images): the best size the item does have."""
    files = [f.replace("http://", "https://") for f in json.loads(get(manifest_url))]
    for size in ("~medium.", "~large.", "~orig.", "~small."):
        for f in files:
            if size in f and f.lower().endswith((".jpg", ".jpeg", ".png")):
                return f
    return None


def download(item):
    nasa_id, title, url, manifest = item
    path = OUT / file_name(nasa_id, title)
    if path.exists():
        return "skipped"
    err = None
    for attempt in range(3):
        try:
            data = get(url)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(path)
            return "ok"
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (403, 404):  # that size doesn't exist: look up which ones do (once)
                url = best_file(manifest) if url.endswith(("~medium.jpg", "~medium.png")) else None
                if not url:
                    break
            else:
                time.sleep(1 + attempt)
        except Exception as e:
            err = e
            time.sleep(1 + attempt)
    print(f"FAILED {nasa_id}: {err}", flush=True)
    return "failed"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    items, seen = [], set()
    for topic in TOPICS:
        found = [x for x in search(topic) if x[0] not in seen]
        seen.update(x[0] for x in found)
        items += found
        print(f"[{time.time() - start:4.0f}s] {topic}: {len(found)} new, {len(items)} total", flush=True)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(8) as pool:
        for k, done in enumerate(as_completed([pool.submit(download, x) for x in items]), 1):
            counts[done.result()] += 1
            if k % 100 == 0 or k == len(items):
                print(f"[{time.time() - start:4.0f}s] downloaded {k}/{len(items)}: {counts}", flush=True)
    size = sum(p.stat().st_size for p in OUT.glob("*.jpg")) / 1e6
    print(f"FETCH_DONE: {len(list(OUT.glob('*.jpg')))} images, {size:.0f} MB in {OUT}", flush=True)
