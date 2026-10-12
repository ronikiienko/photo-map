"""Trial: name a region bottom-up with a small language model, from the finest regions to the one on top.
    python src/name_tree.py <level 2-7> <name of a region at that level in regions.json>
The finest regions (>= 10 photos) are named from their captions; every region above, from the labels of its named
children (with their photo counts). Prints the tree next to the names regions.json has.
"""

import json
import sys
import time
from collections import Counter

import numpy as np
import torch

import config
import photomap
from name_trial import BAD, MODEL, PROMPT, region_captions

LABELS = """Below are the names of groups of pictures, with the number of pictures in each group:

{labels}

What is the main subject of the whole set? Answer with one or two plain words (a noun such as a kind of building, place or object), in lowercase, and nothing else. If there is no main subject, answer: none"""


def ask(model, tok, prompt):
    msg = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(msg, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    answer = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip().lower().strip(".")
    return None if not answer or answer == "none" or set(answer.split()) & BAD else answer


if __name__ == "__main__":
    top_level, top_name = int(sys.argv[1]) - 1, sys.argv[2]
    bundle = config.ROOT / "maps/wikivoyage.photomap"
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    texts = [p["meta"].get("caption", "") for p in json.loads((bundle / "photos.json").read_text())]
    levels = photomap.cluster(xy)
    named = json.loads((bundle / "regions.json").read_text())["levels"]
    counting = []  # per level: {region: name} from regions.json
    for lv, lab in enumerate(levels):
        counting.append({int(lab[int(np.argmin(((xy - [r["x"], r["y"]]) ** 2).sum(1)))]): r["name"] for r in named[lv]})
    top = next(r for r, nm in counting[top_level].items() if nm == top_name)
    inside = levels[top_level] == top  # the photos of the region
    last = len(levels) - 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    sizes = [np.bincount(l[inside], minlength=l.max() + 1) for l in levels]
    model_name = [dict() for _ in levels]  # per level: region -> label
    t0 = time.time()
    for lv in range(last, top_level - 1, -1):
        todo = [r for r in np.flatnonzero(sizes[lv]) if sizes[lv][r] >= photomap.MIN_PHOTOS and (lv > top_level or r == top)]
        for r in todo:
            if lv == last:
                idx = np.flatnonzero(inside & (levels[lv] == r))
                prompt = PROMPT.format(captions=region_captions(texts, idx))
            else:
                idx = np.flatnonzero(inside & (levels[lv] == r))
                kids = Counter()
                for k in np.unique(levels[lv + 1][idx]):
                    if k in model_name[lv + 1]:
                        kids[model_name[lv + 1][k]] += int(sizes[lv + 1][k])
                if not kids:
                    continue
                prompt = LABELS.format(labels="\n".join(f"{nm} x{c}" for nm, c in kids.most_common()))
            label = ask(model, tok, prompt)
            if label:
                model_name[lv][r] = label
        print(f"level {lv + 1}: {len(model_name[lv])} of {len(todo)} regions named  [{time.time() - t0:.0f}s]", flush=True)

    # the tree: the top region, its children, their children
    def show(lv, r, depth):
        photos = int(sizes[lv][r])
        print(f"{'    ' * depth}{photos:5} photos  model: {model_name[lv].get(r, '-'):18} counting: {counting[lv].get(r, '-')}")
        if depth < 2 and lv < last:
            kids = sorted({int(k) for k in np.unique(levels[lv + 1][inside & (levels[lv] == r)]) if sizes[lv + 1][k] >= photomap.MIN_PHOTOS}, key=lambda k: -sizes[lv + 1][k])
            for k in kids[:10]:
                show(lv + 1, k, depth + 1)

    print()
    show(top_level, top, 0)
    json.dump([{int(r): nm for r, nm in d.items()} for d in model_name], open(config.ROOT / "logs/name_tree.json", "w"))
