"""Small square thumbnails of the Wikivoyage photos, for the photo map (it would be slow with the full photos).
    python src/make_thumbs.py   ->  data/wikivoyage/thumbs/<same sub-path>.webp, 128x128 centre crops
Skips thumbnails that already exist, so it can be stopped and restarted.
"""

import json
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "wikivoyage"
OUT = SRC / "thumbs"
SIZE = 128


def thumb_path(image):  # "data/wikivoyage/images/06384.jpg" -> thumbs/images/06384.webp
    return OUT / Path(image).relative_to("data/wikivoyage").with_suffix(".webp")


def make(image):
    out = thumb_path(image)
    if out.exists():
        return True
    try:
        with Image.open(ROOT / image) as im:
            im.draft("RGB", (SIZE * 2, SIZE * 2))  # JPEGs decode at reduced size: much faster
            im = ImageOps.fit(im.convert("RGB"), (SIZE, SIZE), Image.Resampling.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, "WEBP", quality=75)
        return True
    except Exception:
        return False  # the map falls back to the full photo


if __name__ == "__main__":
    images = [p["image"] for p in json.loads((ROOT / "index" / "wikivoyage" / "image_chunks.json").read_text())]
    failed = done = 0
    with Pool(10) as pool:
        for ok in pool.imap_unordered(make, images, chunksize=64):
            done += 1
            failed += not ok
            if done % 10_000 == 0:
                print(f"{done:,} / {len(images):,} thumbnails ({failed} failed)", flush=True)
    size = sum(f.stat().st_size for f in OUT.rglob("*.webp"))
    print(f"THUMBS_DONE: {done - failed:,} thumbnails, {failed} failed, {size / 1e6:.0f} MB", flush=True)
