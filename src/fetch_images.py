"""Get each Wikivoyage article's image (picked by parse_wikivoyage.py): from the Kiwix ZIM if it has it,
else downloaded from Wikimedia as a ~960 px thumbnail.
    data/wikivoyage/images/NNNNN.*      one file per article that has an image
    data/wikivoyage/images.json         [{title, file, caption, path}], written at the end
Rerun to resume: files already on disk are skipped.
"""

import html
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import config
import pyarrow.parquet as pq
from libzim.reader import Archive

DIR = config.ROOT / "data" / "wikivoyage"
ZIM = DIR / "wikivoyage_en_all_maxi_2026-09.zim"  # Kiwix copy of Wikivoyage, images included
OUT = DIR / "images"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "EmbeddingTest/0.1 (personal research; ronikiienko@gmail.com)"}
WORKERS = 1  # Wikimedia throttles parallel downloads (HTTP 429); one at a time is not throttled

articles = []
for a in pq.read_table(DIR / "articles.parquet").to_pylist():
    a["image"] = html.unescape(a["image"])  # a few names come as "Molua&#039;s Church.jpg"
    if a["image"] and not any(c in a["image"] for c in "|#<>[]{}"):  # skip template leftovers like {{{1|...}}}
        articles.append(a)


def get(url, tries=6, post=None):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=post, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = int(e.headers.get("Retry-After") or 0) or 2 ** attempt * 5  # 429 / 5xx: back off
            print(f"  HTTP {e.code}, waiting {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(2 ** attempt)
    return None


# Step 1: file name -> thumbnail URL, 50 files per API request
urls_file = DIR / "image_urls.json"
urls = json.loads(urls_file.read_text()) if urls_file.exists() else {}
names = sorted({a["image"] for a in articles} - urls.keys())
for i in range(0, len(names), 50):
    q = urllib.parse.urlencode({"action": "query", "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 960, "format": "json",
                                "formatversion": 2, "titles": "|".join("File:" + n for n in names[i : i + 50])})
    for attempt in range(5):  # the API sometimes answers with an error (busy, rate limit) instead of results
        # POST: 50 long, percent-encoded file names can be too long for a URL (HTTP 414)
        reply = json.loads(get("https://en.wikivoyage.org/w/api.php", post=q.encode()) or "{}")
        if "query" in reply:
            break
        print(f"  API said {reply.get('error', reply)}; retrying", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        continue  # give up on these 50 files
    data = reply["query"]
    renamed = {n["to"]: n["from"] for n in data.get("normalized", [])}
    for p in data["pages"]:
        asked = renamed.get(p["title"], p["title"]).removeprefix("File:")
        info = p.get("imageinfo", [{}])[0]
        urls[asked] = info.get("thumburl") or info.get("url")  # None: file was deleted
    if (i // 50) % 40 == 0:
        print(f"urls: {min(i + 50, len(names))}/{len(names)}", flush=True)
        urls_file.write_text(json.dumps(urls))
urls_file.write_text(json.dumps(urls))
print(f"urls done: {sum(1 for u in urls.values() if u)} of {len(urls)} files found", flush=True)


# Step 2: copy what the Kiwix ZIM already has. Its images are small (~250-400 px) but need no download;
# in our test their vectors were 0.94 similar to the 960 px originals. Images already downloaded are kept.
def zim_name(name):
    for _ in range(3):  # ZIM names can be percent-encoded twice ("Exterior%252C_...")
        name = urllib.parse.unquote(name)
    name = name.replace(" ", "_")
    return name[:1].upper() + name[1:]


on_disk = {int(p.stem): p for p in OUT.iterdir() if p.stem.isdigit() and p.suffix != ".part"}
if ZIM.exists():
    zim = Archive(ZIM)
    media = {}
    for i in range(zim.all_entry_count):
        e = zim._get_entry_by_id(i)
        if e.path.startswith("_assets_/"):
            media[zim_name(e.path.rsplit("/", 1)[-1])] = e
    copied = 0
    for n, a in enumerate(articles):
        e = media.get(zim_name(a["image"]))
        if n in on_disk or e is None:
            continue
        item = e.get_item()
        ext = {"image/webp": "webp", "image/jpeg": "jpg", "image/png": "png", "image/gif": "gif"}.get(item.mimetype)
        if ext:
            on_disk[n] = OUT / f"{n:05d}.{ext}"
            on_disk[n].write_bytes(bytes(item.content))
            copied += 1
    print(f"zim: copied {copied} images; {len(articles) - len(on_disk)} still missing -> download", flush=True)


# Step 3: download whatever is still missing
def fetch(n_a):
    n, a = n_a
    path = on_disk.get(n)
    if path is None:
        url = urls.get(a["image"])
        if not url:
            return None
        path = OUT / f"{n:05d}.{url.split('?')[0].rsplit('.', 1)[-1].lower()}"  # extension, without ?utm_... junk
        data = get(url)
        if data is None:
            return None
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").rename(path)
        downloaded.append(n)
    return {"title": a["title"], "file": a["image"], "caption": a["caption"], "path": str(path.relative_to(config.ROOT))}


done, downloaded, start = [], [], time.time()
missing = len(articles) - len(on_disk)
with ThreadPoolExecutor(WORKERS) as pool:
    for k, item in enumerate(pool.map(fetch, enumerate(articles)), 1):
        if item:
            done.append(item)
        if k % 500 == 0:  # files already on disk are skipped instantly; the time goes into downloads
            rate = len(downloaded) / (time.time() - start)
            print(f"images: {k}/{len(articles)} checked, downloaded {len(downloaded)} of ~{missing} missing, {rate:.1f}/s", flush=True)
(DIR / "images.json").write_text(json.dumps(done, ensure_ascii=False, indent=0))
print(f"FETCH_DONE: {len(done)} images", flush=True)
