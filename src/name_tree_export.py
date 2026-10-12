"""Make maps/wikivoyage-qwen.photomap from the names name_tree.py found (logs/name_tree.json): the same photos and
layout as maps/wikivoyage.photomap (linked, not copied), but regions.json holds only the language model's names.
    python src/name_tree_export.py
"""

import json

import numpy as np

import config
import photomap

if __name__ == "__main__":
    src, dst = config.ROOT / "maps/wikivoyage.photomap", config.ROOT / "maps/wikivoyage-qwen.photomap"
    n = json.loads((src / "manifest.json").read_text())["photos"]
    xy = np.fromfile(src / "layout.bin", np.float32).reshape(n, 2)
    levels = photomap.cluster(xy)
    found = json.loads((config.ROOT / "logs/name_tree.json").read_text())
    out = []
    for lab, names in zip(levels, found):
        level = []
        for r, nm in names.items():
            pts = xy[lab == int(r)]
            c = pts.mean(0)
            d = ((pts - c) ** 2).sum(1)
            x, y = pts[d.argmin()]
            level.append({"name": nm, "x": float(x), "y": float(y), "spread": float(np.sqrt(d.mean())), "photos": len(pts)})
        out.append(sorted(level, key=lambda l: -l["photos"]))
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("manifest.json", "regions.json") and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["name"] = "wikivoyage (qwen names)"
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (dst / "regions.json").write_text(json.dumps({"levels": out}, ensure_ascii=False))
    photomap.step(f"{dst.name}: {sum(map(len, out))} names on {len(out)} levels")
