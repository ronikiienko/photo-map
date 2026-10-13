"""Unpack the wikiart parquet shards into one image file per painting.

Usage: python wikiart_unpack.py WIKIART_DIR
Reads WIKIART_DIR/parquet/*.parquet and WIKIART_DIR/labels.json; writes WIKIART_DIR/images/<artist> - <style> - <genre> [shard-row].<ext>.
Safe to stop and start again (files that exist are skipped).
"""
import json, sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
labels = json.loads((root / "labels.json").read_text())
out = root / "images"
out.mkdir(exist_ok=True)


def words(label):
    return label.replace("_", " ").replace("-", " ").title()


shards = sorted((root / "parquet").glob("*.parquet"))
total = 0
for n, shard in enumerate(shards, 1):
    tag = shard.name.split("-")[1]
    rows = pq.read_table(shard).to_pylist()
    for i, r in enumerate(rows):
        data = r["image"]["bytes"]
        ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
        name = f"{words(labels['artist'][r['artist']])} - {words(labels['style'][r['style']])} - {words(labels['genre'][r['genre']])} [{tag}-{i:04d}]{ext}"
        path = out / name
        if not path.exists():
            path.write_bytes(data)
        total += 1
    print(f"shard {n}/{len(shards)}: {total} paintings", flush=True)
print("finished", total, flush=True)
