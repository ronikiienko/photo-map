"""Download a random sample of the Met's public-domain images (web-large size) with their titles.

Usage: python met_download.py OUT_DIR COUNT
Writes OUT_DIR/images/<objectID>.jpg and OUT_DIR/objects.jsonl (one line per photo). Safe to stop and start again.
"""
import json, os, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://collectionapi.metmuseum.org/public/collection/v1.1/search?hasImages=true&isPublicDomain=true&q=*"
OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
HEADERS = {"User-Agent": "photomap-research (ronikiienko@gmail.com)"}
PER_SECOND = 16  # the Met asks for at most 80 requests a second; each photo is two requests

out, want = Path(sys.argv[1]), int(sys.argv[2])
images = out / "images"
images.mkdir(parents=True, exist_ok=True)
log = out / "objects.jsonl"


def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403:  # the Met's firewall has blocked us: carrying on would only make it worse
                print("blocked (403), stopping", flush=True)
                os._exit(1)
            time.sleep(1 + t * 2)
        except Exception:
            time.sleep(1 + t * 2)
    return None


ids_file = out / "ids.json"
if ids_file.exists():
    ids = json.loads(ids_file.read_text())
else:
    ids = json.loads(get(API))["objectIDs"]
    random.Random(2026).shuffle(ids)
    ids_file.write_text(json.dumps(ids))

have = {int(p.stem) for p in images.glob("*.jpg")}
lock = threading.Lock()
state = {"ok": len(have), "tried": 0, "last": time.time()}
tokens = threading.Semaphore(0)


def refill():
    while True:
        for _ in range(PER_SECOND // 4):
            tokens.release()
        time.sleep(0.25)


def work(oid):
    if state["ok"] >= want or oid in have:
        return
    tokens.acquire(); tokens.acquire()
    meta = get(OBJECT.format(oid))
    if not meta:
        return
    d = json.loads(meta)
    url = d.get("primaryImageSmall")
    if not url or not d.get("isPublicDomain"):
        return
    data = get(url)
    if not data:
        return
    (images / f"{oid}.jpg").write_bytes(data)
    row = {k: d.get(k) for k in ("objectID", "title", "artistDisplayName", "department", "objectName", "culture", "period", "objectDate", "medium", "classification", "objectURL")}
    with lock:
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        state["ok"] += 1
        if state["ok"] % 250 == 0:
            print(f"{state['ok']}/{want}", flush=True)


threading.Thread(target=refill, daemon=True).start()
with ThreadPoolExecutor(16) as pool:
    list(pool.map(work, ids[: want * 2]))
print("done", state["ok"], flush=True)
