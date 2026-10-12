"""Ask a bigger model for the regions the 2B model gave a very common name (level 6 of the Wikivoyage map), side by side.
    python src/model_vs.py [model] [per_name]       default: Qwen3.5-4B-Q4_K_M, 5 regions for each of the 8 commonest 2B names
Same prompt as the 2B run (name_compare.LABEL_PROMPT), same captions. Prints 2B | bigger model | two captions.
"""

import json
import sys
from collections import Counter

import numpy as np

import config
import photomap
from name_compare import LABEL_PROMPT, Server, prompt_for

if __name__ == "__main__":
    gguf = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-4B-Q4_K_M"
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    bundle = config.ROOT / "maps/wikivoyage.photomap"
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    texts = [p["meta"].get("caption", "") for p in json.loads((bundle / "photos.json").read_text())]
    lab = photomap.cluster(xy)[5]
    members = photomap.members(lab, lab.max() + 1)
    at = {(float(x), float(y)): i for i, (x, y) in enumerate(xy)}
    named = json.loads((bundle / "region_names.json").read_text())["levels"][5]
    by_name = {}
    for r in named:
        by_name.setdefault(r["name"], []).append(int(lab[at[(r["x"], r["y"])]]))
    rng = np.random.default_rng(5)
    common = [name for name, _ in Counter({k: len(v) for k, v in by_name.items()}).most_common(8)]
    rows = []
    with Server(gguf) as server:
        for name in common:
            for r in rng.choice(by_name[name], per, replace=False):
                caps = [t for t in (texts[i] for i in members[r]) if t]
                rows.append((name, server.ask(prompt_for(texts, members[r], LABEL_PROMPT), max_tokens=14, drop_bad=False), len(members[r]), caps[:2]))
    for old, new, size, caps in rows:
        print(f"{old:<18} | {new:<24} | {size:>3} photos | {' / '.join(c[:34] for c in caps)}")
    same = sum(o == n for o, n, _, _ in rows)
    print(f"\n{same} of {len(rows)} unchanged; distinct answers: 2B {len(set(r[0] for r in rows))}, {gguf.split('-')[1]} {len(set(r[1] for r in rows))}")
