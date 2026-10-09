"""Turn the Wikivoyage XML dump into data/wikivoyage/articles.parquet (title, url, text, image, caption).
Wikitext -> plain text: listings ({{see|name=...|content=...}}) become readable lines, other templates
(maps, banners, climate charts...) are dropped. image = the article's first photo, or its banner if it has none.
"""

import bz2
import re
import xml.etree.ElementTree as ET

import config
import mwparserfromhell as mw
import pyarrow as pa
import pyarrow.parquet as pq

DUMP = config.ROOT / "data" / "wikivoyage" / "enwikivoyage-20261001-pages-articles.xml.bz2"
NS = "{http://www.mediawiki.org/xml/export-0.11/}"
LISTINGS = {"see", "do", "buy", "eat", "drink", "sleep", "listing", "go"}
UNITS = {"eur": "EUR", "usd": "USD", "km": "km", "mi": "mi", "c": "°C", "f": "°F", "m": "m", "ft": "ft", "kg": "kg"}
NOT_A_PHOTO = re.compile(r"\.svg$|map|locator|flag|logo|icon", re.I)


def arg(t, name):
    return plain(str(t.get(name).value)) if t.has(name) else ""


def plain(wikitext):
    """Wikitext -> plain text, with listings and a few inline templates turned into words."""
    code = mw.parse(wikitext)
    for t in code.filter_templates(recursive=False):
        name = str(t.name).strip().lower()
        if name in LISTINGS:
            head = arg(t, "name") + (f" ({arg(t, 'alt')})" if arg(t, "alt") else "")
            parts = [arg(t, "content")] + [f"{k.title()}: {arg(t, k)}" for k in ("address", "directions", "hours", "price") if arg(t, k)]
            text = f"{head}: " + " ".join(p.rstrip(".") + "." for p in parts if p)
        elif name == "marker":
            text = arg(t, "name")
        elif name in UNITS and t.params:
            text = f"{plain(str(t.params[0].value))} {UNITS[name]}"
        elif name == "convert" and len(t.params) >= 2:
            text = f"{t.params[0].value} {t.params[1].value}".strip()
        elif name in ("lang", "iata") and t.params:
            text = plain(str(t.params[-1].value))
        elif name == "regionlist":  # region pages: regionNname / regionNdescription pairs
            names = {str(p.name).strip(): plain(str(p.value)) for p in t.params}
            text = "\n".join(f"{v}: {names.get(k.replace('name', 'description'), '')}" for k, v in names.items() if k.endswith("name") and v)
        else:
            text = ""
        try:
            code.replace(t, text)
        except ValueError:
            pass
    for link in code.filter_wikilinks(recursive=False):  # files, categories, other-language links: drop
        if re.match(r"\s*(file|image|category|[a-z]{2,3}):", str(link.title), re.I):
            code.remove(link)
    text = code.strip_code(normalize=True, collapse=True)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def pick_image(wikitext):
    """First photo in the article (from [[File:...]] or a listing's image=), else the banner. Returns (file, caption)."""
    code = mw.parse(wikitext)
    banner = ("", "")
    for node in code.filter(recursive=True, forcetype=(mw.nodes.Wikilink, mw.nodes.Template)):
        if isinstance(node, mw.nodes.Wikilink):
            m = re.match(r"\s*(?:file|image):(.+)", str(node.title), re.I)
            if m and not NOT_A_PHOTO.search(m.group(1)):
                caption = str(node.text).split("|")[-1] if node.text else ""
                return m.group(1).strip(), plain(caption)
        else:
            name = str(node.name).strip().lower()
            if name == "pagebanner" and node.params and not banner[0]:
                banner = (str(node.params[0].value).strip(), arg(node, "caption"))
            elif name in LISTINGS and node.has("image") and arg(node, "image") and not NOT_A_PHOTO.search(arg(node, "image")):
                return arg(node, "image"), arg(node, "name")
    return banner


rows, seen = [], 0
for _, el in ET.iterparse(bz2.open(DUMP)):
    if el.tag != NS + "page":
        continue
    if el.findtext(NS + "ns") == "0" and el.find(NS + "redirect") is None:
        title, wikitext = el.findtext(NS + "title"), el.findtext(f"{NS}revision/{NS}text") or ""
        seen += 1
        if not re.search(r"\{\{\s*disamb", wikitext, re.I):
            text = plain(wikitext)
            if len(text) >= 200:
                image, caption = pick_image(wikitext)
                url = "https://en.wikivoyage.org/wiki/" + title.replace(" ", "_")
                rows.append({"title": title, "url": url, "text": text, "image": image, "caption": caption})
        if seen % 2000 == 0:
            print(f"parsed {seen} articles, kept {len(rows)}", flush=True)
    el.clear()

pq.write_table(pa.Table.from_pylist(rows), config.ROOT / "data" / "wikivoyage" / "articles.parquet")
with_image = sum(1 for r in rows if r["image"])
print(f"PARSE_DONE: {len(rows)} articles, {sum(len(r['text']) for r in rows) / 1e6:.0f}M chars, {with_image} with an image", flush=True)
