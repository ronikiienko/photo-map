"""Speed of Qwen3.5-0.8B (the region-naming model) on this GPU: the same prompts of 300 ... 6000 tokens, 8 new tokens each.
    python src/qwen_bench.py            as installed (flash-linear-attention's fast kernels if it is installed)
    python src/qwen_bench.py --no-fla   the slow fallback path, to see what the kernels bring
Run it with the viewer closed: the model takes 1.5 GB of the 4 GB.
"""

import json
import sys
import time

if "--no-fla" in sys.argv:
    sys.modules["fla"] = None  # makes transformers think flash-linear-attention is missing

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

MODEL = "Qwen/Qwen3.5-0.8B"
SIZES = (300, 1000, 3000, 6000)  # prompt tokens, about

if __name__ == "__main__":
    photos = json.loads((config.ROOT / "maps/wikivoyage.photomap/photos.json").read_text())
    captions = [p["meta"]["caption"] for p in photos if p["meta"].get("caption")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    print(f"loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    rng = np.random.default_rng(0)
    for size in SIZES:
        text = ""
        while len(tok(text).input_ids) < size:
            text += "\n".join(captions[i] for i in rng.choice(len(captions), 40)) + "\n"
        ids = tok(text, return_tensors="pt", truncation=True, max_length=size).to("cuda")
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=2, do_sample=False)  # warm-up (compiles the kernels the first time)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                t = time.time()
                model.generate(**ids, max_new_tokens=8, do_sample=False)
                torch.cuda.synchronize()
                times.append(time.time() - t)
        print(f"{ids.input_ids.shape[1]:5} tokens: {np.median(times) * 1000:6.0f} ms  ({ids.input_ids.shape[1] / np.median(times):6.0f} tok/s)  peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
