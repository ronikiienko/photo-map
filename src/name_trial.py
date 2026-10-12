"""Trial: can a small language model name a region from its captions?
    python src/name_trial.py [per_level]
Takes a few regions of maps/wikivoyage.photomap at levels 2, 4 and 6, gives the model all of a region's captions
(repeats collapsed to counts) and prints its answer next to the name regions.json has, with the time it took.
"""

import json
import sys
import time
from collections import Counter

import numpy as np
import torch

import config
import photomap

MODEL = "Qwen/Qwen3.5-0.8B"
BAD = {"travel", "photo", "photos", "photography", "picture", "pictures", "image", "images", "caption", "captions", "map"}  # words of the prompt: not a name
LEVELS = (1, 3, 5)  # 0-based: levels 2, 4, 6
PROMPT = """Below are the captions of a group of pictures ("x3" means three pictures have that caption):

{captions}

What is the main subject that most of these captions are about? Answer with one or two plain words (a noun such as a kind of building, place or object), in lowercase, and nothing else. If there is no main subject, answer: none"""


def region_captions(texts, idx):
    count = Counter(texts[i] for i in idx if texts[i])
    return "\n".join(f"{t} x{n}" if n > 1 else t for t, n in count.most_common())


if __name__ == "__main__":
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    bundle = config.ROOT / "maps/wikivoyage.photomap"
    n = json.loads((bundle / "manifest.json").read_text())["photos"]
    xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
    texts = [p["meta"].get("caption", "") for p in json.loads((bundle / "photos.json").read_text())]
    levels = photomap.cluster(xy)
    named = json.loads((bundle / "regions.json").read_text())["levels"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    photomap.step(f"model loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    rng = np.random.default_rng(1)
    for lv in LEVELS:
        lab = levels[lv]
        members = photomap.members(lab, lab.max() + 1)
        current = {}
        for r in named[lv]:  # the photo a name is placed at -> its region
            current[int(lab[int(np.argmin(((xy - [r["x"], r["y"]]) ** 2).sum(1)))])] = r["name"]
        ok = [r for r, m in enumerate(members) if 10 <= sum(bool(texts[i]) for i in m) <= 5000]
        print(f"\n=== level {lv + 1} ({lab.max() + 1} regions, {len(ok)} to pick from)")
        for r in sorted(rng.choice(ok, min(per, len(ok)), replace=False)):
            caps = region_captions(texts, members[r])
            msg = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(captions=caps)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(msg, return_tensors="pt").to("cuda")
            if ids.input_ids.shape[1] > 8000:  # runs out of memory on this GPU
                print(f"{len(members[r]):5} photos {ids.input_ids.shape[1]:6} tok  skipped (too long)")
                continue
            t0 = time.time()
            try:
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=8, do_sample=False)
            except torch.OutOfMemoryError:
                print(f"{len(members[r]):5} photos {ids.input_ids.shape[1]:6} tok  out of memory")
                torch.cuda.empty_cache()
                continue
            answer = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            raw = answer
            answer = "none" if set(answer.lower().strip(".").split()) & BAD else answer
            top = ", ".join(c.rsplit(" x", 1)[0] for c in caps.split("\n")[:3])
            print(f"{len(members[r]):5} photos {ids.input_ids.shape[1]:6} tok {time.time() - t0:5.2f}s  model: {answer!r:16} (raw {raw!r:22}) counting: {current.get(r, '-')!r:20} e.g. {top[:70]}")
