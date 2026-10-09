"""Turn a Kiwix Wikipedia ZIM into a folder of parquet parts (title, url, text, image, caption).
    python src/parse_zim.py data/history/wikipedia_en_history_maxi_2026-07.zim data/history/articles
HTML -> plain text: keep headings, paragraphs, lists; drop references, infoboxes, navigation boxes, tables,
and the References / See also / External links sections. image = the article's first real picture (ZIM path).
Runs on several CPU cores; each worker reads its own share of the ZIM and saves its own part
(so no process ever holds all the text: it can be gigabytes). pq.read_table(folder) reads all parts.
"""

import os
import sys
import time
from multiprocessing import Pool

import lxml.html
import pyarrow as pa
import pyarrow.parquet as pq
from libzim.reader import Archive

ZIM, OUT = sys.argv[1], sys.argv[2]
WORKERS = 6
DROP = ("style", "script", "table", "figure", "sup", "math", "noscript", "link", "meta")
DROP_CLASSES = ("mw-editsection", "reflist", "references", "navbox", "hatnote", "metadata", "noprint", "gallery",
                "shortdescription", "thumb", "mw-empty-elt", "infobox", "sidebar", "ambox", "toc")
END_SECTIONS = {"references", "notes", "citations", "sources", "bibliography", "further reading", "external links",
                "see also", "footnotes", "notes and references", "works cited", "general and cited sources"}
BLOCKS = ("p", "li", "h2", "h3", "h4", "h5", "dd", "dt", "blockquote")


def first_image(root, path):
    """First picture at least 100 px wide (skips icons and flags). Returns (zim path, caption)."""
    for img in root.iter("img"):
        if int(img.get("width") or 0) >= 100 and img.get("src"):
            src = img.get("src")
            base = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
            parts = []
            for p in (base + src).split("/"):
                if p == "..":
                    parts and parts.pop()
                elif p not in ("", "."):
                    parts.append(p)
            fig = next((a for a in img.iterancestors() if a.tag in ("figure", "table")), None)
            cap = fig.find(".//figcaption") if fig is not None else None
            caption = " ".join(cap.text_content().split()) if cap is not None else ""
            return "/".join(parts), caption
    return "", ""


def to_text(html, path):
    root = lxml.html.fromstring(html)
    body = root.find_class("mw-parser-output")
    root = body[0] if body else root
    image = first_image(root, path)
    for el in list(root.iter("section")):  # cut the reference-like sections (their h2 may sit inside a wrapper div)
        heading = el.find(".//h2")
        if heading is not None and heading.text_content().strip().lower() in END_SECTIONS and el.getparent() is not None:
            el.getparent().remove(el)
    for el in list(root.iter(*DROP)):
        if el.getparent() is not None:
            el.drop_tree()
    for cls in DROP_CLASSES:
        for el in root.find_class(cls):
            if el.getparent() is not None:
                el.drop_tree()
    lines = []
    for el in root.iter(*BLOCKS):
        if any(a.tag in BLOCKS for a in el.iterancestors()):
            continue  # nested list item: already part of its parent's text
        line = " ".join(el.text_content().split())
        if line:
            lines.append("\n" + line if el.tag.startswith("h") else line)
    return "\n".join(lines).strip(), image


def work(share):
    k, n = share
    z = Archive(ZIM)
    rows = []
    for i in range(k, z.all_entry_count, n):
        e = z._get_entry_by_id(i)
        if e.is_redirect or e.path.startswith("_"):
            continue
        item = e.get_item()
        if not item.mimetype.startswith("text/html"):
            continue
        text, (image, caption) = to_text(bytes(item.content), e.path)
        if len(text) >= 200:
            rows.append({"title": e.title, "url": "https://en.wikipedia.org/wiki/" + e.path, "text": text,
                         "image": image, "caption": caption})
        if k == 0 and len(rows) % 2000 == 0 and rows:
            print(f"  worker 0: {len(rows)} articles", flush=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/part-{k}.parquet")
    return len(rows), sum(len(r["text"]) for r in rows), sum(1 for r in rows if r["image"])


if __name__ == "__main__":
    if len(sys.argv) > 3:  # test: print one article
        z = Archive(ZIM)
        e = z.get_entry_by_path(sys.argv[3])
        text, image = to_text(bytes(e.get_item().content), e.path)
        print(text[:3000], "\n...\n", text[-800:], f"\n\n{len(text)} chars, image: {image}")
        sys.exit()
    start = time.time()
    os.makedirs(OUT, exist_ok=True)
    with Pool(WORKERS) as pool:
        n, chars, images = map(sum, zip(*pool.map(work, [(k, WORKERS) for k in range(WORKERS)])))
    print(f"PARSE_DONE: {n} articles, {chars / 1e6:.0f}M chars, {images} with an image, {time.time() - start:.0f}s", flush=True)
