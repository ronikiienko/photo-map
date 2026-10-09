"""Collect every image used in the articles of a Kiwix ZIM (not just one per article), each unique image once.
    python src/collect_zim_images.py           # count only
    python src/collect_zim_images.py --write   # extract images, write data/<corpus>/images_all.json (after fetch_images.py)
Skipped: images shown narrower than MIN_WIDTH px, and Wikimedia's own interface symbols (UI_ICONS).
Flags, logos and route icons are kept: each unique image is embedded once, however many pages use it.
Where fetch_images.py already has a bigger copy of an article's lead photo (960 px download), that copy is used.
"""

import collections
import json
import re
import sys
import urllib.parse

import config
import lxml.html
from libzim.reader import Archive

DIR = config.ROOT / "data" / config.CORPUS
ZIM = next(DIR.glob("*.zim"))
OUT = DIR / "images_zim"
MIN_WIDTH = 20  # inline flags are shown ~20 px wide
UI_ICONS = re.compile(r"handicap|wheelchair|wi-?fi|oojs|(commons|wikipedia|wikivoyage|wikidata|wiktionary)[-_]logo|"
                      r"disambig|pencil|edit[-_]icon|magnify|info[-_]icon|ambox|question_book|padlock|"
                      r"symbol_(support|oppose)|gnome-|crystal_clear|nuvola|star_full|star_empty|star_half|exclamationdiamond", re.I)
write = "--write" in sys.argv


def zim_name(name):
    for _ in range(3):  # names can be percent-encoded twice
        name = urllib.parse.unquote(name)
    name = name.replace(" ", "_")
    return name[:1].upper() + name[1:]


def resolve(page, src):
    parts = page.split("/")[:-1]
    for p in src.split("/"):
        if p == "..":
            parts and parts.pop()
        elif p not in ("", "."):
            parts.append(p)
    return "/".join(parts)


zim = Archive(ZIM)
used = collections.defaultdict(list)  # zim image path -> [(article title, caption)]
width = {}  # zim image path -> widest it is shown
pages = 0
for i in range(zim.all_entry_count):
    e = zim._get_entry_by_id(i)
    if e.is_redirect or e.path.startswith("_") or not e.get_item().mimetype.startswith("text/html"):
        continue
    pages += 1
    root = lxml.html.fromstring(bytes(e.get_item().content))
    seen = set()
    for img in root.iter("img"):
        src = img.get("src")
        if not src or int(img.get("width") or 0) < MIN_WIDTH or UI_ICONS.search(src):
            continue
        path = resolve(e.path, src)
        width[path] = max(width.get(path, 0), int(img.get("width") or 0))
        if path in seen:
            continue
        seen.add(path)
        fig = next((a for a in img.iterancestors() if a.tag == "figure"), None)
        cap = fig.find(".//figcaption") if fig is not None else None
        caption = " ".join(cap.text_content().split()) if cap is not None else (img.get("alt") or "")
        used[path].append((e.title, caption))
    if pages % 5000 == 0:
        print(f"  {pages} pages read, {len(used)} unique images so far", flush=True)

per_page = sum(len(v) for v in used.values())
print(f"COUNT: {pages} pages, {per_page} image uses, {len(used)} unique images (>= {MIN_WIDTH} px wide)", flush=True)
small = [p for p in used if width[p] < 100]
print(f"  of which shown smaller than 100 px: {len(small)}", flush=True)
for path in sorted(small, key=lambda p: -len(used[p]))[:25]:
    print(f"  small, {width[path]:3d} px, used on {len(used[path]):5d} pages: {path.rsplit('/', 1)[-1][:60]}", flush=True)

if write:
    # lead photos from fetch_images.py: 960 px copies, some of them not in the ZIM at all (e.g. listing photos)
    fetched = json.loads((DIR / "images.json").read_text()) if (DIR / "images.json").exists() else []
    big = {zim_name(i["file"]): i["path"] for i in fetched}
    OUT.mkdir(exist_ok=True)
    items, names = [], set()
    for n, (path, uses) in enumerate(sorted(used.items())):
        name = zim_name(path.rsplit("/", 1)[-1])
        if name in names:  # same file stored twice in the ZIM
            continue
        names.add(name)
        if name in big:
            file = big[name]
        else:
            item = None
            for p in (path, urllib.parse.unquote(path), urllib.parse.unquote(urllib.parse.unquote(path))):
                try:  # the HTML writes names percent-encoded ("Krak%C3%B3w"), the ZIM stores them decoded ("Kraków")
                    item = zim.get_entry_by_path(p).get_item()
                    break
                except KeyError:
                    pass
            if item is None:
                continue
            ext = {"image/webp": "webp", "image/jpeg": "jpg", "image/png": "png", "image/gif": "gif"}.get(item.mimetype)
            if not ext:
                continue
            target = OUT / f"{n:06d}.{ext}"
            target.write_bytes(bytes(item.content))
            file = str(target.relative_to(config.ROOT))
        caption = next((c for _, c in uses if c), "")
        items.append({"title": uses[0][0], "file": name, "caption": caption, "path": file,
                      "articles": sorted({t for t, _ in uses})})
    from_zim = len(items)
    extra = {}  # downloaded photos the ZIM doesn't have; one entry per file, even if several articles use it
    for i in fetched:
        name = zim_name(i["file"])
        if name not in names:
            extra.setdefault(name, {**i, "file": name, "articles": []})["articles"].append(i["title"])
    items += extra.values()
    (DIR / "images_all.json").write_text(json.dumps(items, ensure_ascii=False, indent=0))
    print(f"COLLECT_DONE: {len(items)} images in images_all.json: {from_zim} from the ZIM "
          f"({sum(1 for i in items[:from_zim] if 'images_zim' not in i['path'])} of them as 960 px copies), "
          f"{len(extra)} only downloaded", flush=True)
