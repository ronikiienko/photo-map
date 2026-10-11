"""Experiment: name regions by letting every photo vote for its best label (vs. the region-average direction).
    python src/vote_regions.py maps/<name>.photomap
Uses the bundle's regions from the clustering in name_regions.py and the text vectors cached by photo-map in
data/vocab/text_cache.*. Photos and words are both centred (average photo / average word subtracted) so a few
generic words don't win everywhere. A region's name: the label its photos vote for most, relative to how often that
label wins over the whole map (and at least a few votes); coarser levels count the votes of all photos inside.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from name_regions import regions

bundle = Path(sys.argv[1])
n = json.loads((bundle / "manifest.json").read_text())["photos"]
vecs = np.fromfile(bundle / "embeddings.bin", np.float16).reshape(n, 768).astype(np.float32)
xy = np.fromfile(bundle / "layout.bin", np.float32).reshape(n, 2)
cached = json.loads((config.ROOT / "data/vocab/text_cache.json").read_text())
allvecs = np.fromfile(config.ROOT / "data/vocab/text_cache.f32", np.float32).reshape(len(cached), 768)
vocab = set()  # only the general vocabulary (as photo-map builds it), not older cached words
for line in (config.ROOT / "data/vocab/places365.txt").read_text().splitlines():
    parts = line.split()[0].split("/")[2:]
    vocab.add(" ".join(parts[1:] + parts[:1]).replace("_", " "))
for line in (config.ROOT / "data/vocab/openimages_all.csv").read_text().splitlines():
    vocab.add(line.split(",", 1)[1].strip().strip('"').lower())
keep = [i for i, w in enumerate(cached) if w in vocab]
words, wvecs = [cached[i] for i in keep], allvecs[keep]
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"{len(words)} words, each photo votes for its top {TOP}")

# every photo's best label, centred cosine, on the GPU
dev = "cuda"
p = torch.tensor(vecs, device=dev)
p = torch.nn.functional.normalize(p - p.mean(0), dim=1)
w = torch.tensor(wvecs, device=dev)
w = torch.nn.functional.normalize(w - w.mean(0), dim=1)
best = torch.cat([(p[i : i + 8192] @ w.T).topk(TOP, dim=1).indices for i in range(0, n, 8192)]).cpu().numpy()  # n x TOP
glob = np.bincount(best.ravel(), minlength=len(words)) / n
print("most common winners overall:", ", ".join(f"{words[j]} {glob[j]:.3f}" for j in np.argsort(-glob)[:12]))

sizes = [max(4, round(np.sqrt(n) / 25)), max(8, round(np.sqrt(n) / 8)), max(14, round(np.sqrt(n) / 2.5))]
labels = regions(xy, sizes)
for level, lab in enumerate(labels):
    used, out = set(), []
    rows = []
    for r in np.unique(lab):
        votes = np.bincount(best[lab == r].ravel(), minlength=len(words)) / (lab == r).sum()
        lift = votes * np.log((votes + 1e-9) / (glob + 1e-9))  # common here and rarer elsewhere
        rows.append((r, (lab == r).sum(), votes, lift))
    for r, size, votes, lift in sorted(rows, key=lambda x: -x[1]):
        name = next((words[j] for j in np.argsort(-lift) if words[j] not in used and votes[j] * size >= 3), "-")
        used.add(name)
        top = ", ".join(f"{words[j]} {votes[j]:.2f}" for j in np.argsort(-votes)[:4])
        out.append(f"{name} ({size})")
    print(f"level {level + 1}:", " | ".join(out))
